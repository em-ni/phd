import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def load_tum_trajectory(filepath):
    """
    Loads TUM trajectory and returns timestamps and poses.
    Format: timestamp x y z qx qy qz qw
    """
    data = np.loadtxt(filepath, comments='#')
    # Check if timestamps are decreasing
    if data[0, 0] > data[-1, 0]:
        print(f"  [INFO] Timestamps are decreasing. Flipping data.")
        data = np.flip(data, axis=0)
    
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8] # qx, qy, qz, qw
    
    return timestamps, positions, quaternions

def interpolate_pose(target_time, timestamps, positions, quaternions):
    """
    Interpolates pose at target_time.
    """
    # Find indices
    idx = np.searchsorted(timestamps, target_time)
    
    if idx == 0:
        return positions[0], quaternions[0]
    if idx >= len(timestamps):
        return positions[-1], quaternions[-1]
        
    t0 = timestamps[idx-1]
    t1 = timestamps[idx]
    ratio = (target_time - t0) / (t1 - t0)
    
    # Position interpolation (Linear)
    p0 = positions[idx-1]
    p1 = positions[idx]
    p_interp = (1 - ratio) * p0 + ratio * p1
    
    # Rotation interpolation (SLERP)
    # Create rotation objects
    r0 = R.from_quat(quaternions[idx-1])
    r1 = R.from_quat(quaternions[idx])
    
    # Slerp
    key_rots = R.from_quat([quaternions[idx-1], quaternions[idx]])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    r_interp = slerp([ratio])[0]
    q_interp = r_interp.as_quat()
    
    return p_interp, q_interp

def process_sequence(name, output_root):
    txt_path = f"dataset/phantom/{name}.txt"
    video_path = f"dataset/phantom/{name}.mp4"
    
    print(f"\nProcessing {name}...")
    
    if not os.path.exists(txt_path) or not os.path.exists(video_path):
        print(f"  [ERROR] Missing files for {name}")
        return

    # 1. Load Trajectory
    timestamps, positions, quaternions = load_tum_trajectory(txt_path)
    
    # Scale positions from meters to millimeters
    print(f"  [INFO] Scaling positions by 1000 (m -> mm)")
    positions = positions * 1000.0
    
    # --- SENSOR TO CAMERA TRANSFORM ---
    # Phantom moves in -X (local). Simulator expects +Y (local).
    # We need a transform T_sc such that T_camera = T_sensor * T_sc
    # mapping Camera +Y to Sensor -X.
    # Rotation +90 deg around Z maps [0, 1, 0] (Y) to [-1, 0, 0] (-X).
    # R_sc = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    
    print(f"  [INFO] Applying Sensor-to-Camera transform (Motion -X -> +Y)")
    r_sc = R.from_euler('z', 90, degrees=True).as_matrix()
    t_sc = np.eye(4)
    t_sc[:3, :3] = r_sc
    
    # Convert all to matrices first
    N = len(positions)
    all_rots = R.from_quat(quaternions).as_matrix()
    all_matrices = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    all_matrices[:, :3, :3] = all_rots
    all_matrices[:, :3, 3] = positions
    
    # Apply T_sc
    # T_camera = T_sensor @ T_sc
    all_matrices = all_matrices @ t_sc
    
    # Update positions and quaternions for the alignment step
    positions = all_matrices[:, :3, 3]
    quaternions = R.from_matrix(all_matrices[:, :3, :3]).as_quat()
    
    # --- ALIGNMENT ---
    # Target Start Pose (from seq_1764114794)
    # Matrix:
    # [[-0.69604685 -0.21296049  0.68568697 10.031978]
    #  [-0.6556269  -0.20079842 -0.72789653 20.91881]
    #  [ 0.29269806 -0.95620491  0.00014248 37.472145]
    #  [ 0.0         0.0         0.0        1.0]]
    
    target_start_matrix = np.array([
        [-0.69604685, -0.21296049,  0.68568697, 10.031978],
        [-0.6556269,  -0.20079842, -0.72789653, 20.91881],
        [ 0.29269806, -0.95620491,  0.00014248, 37.472145],
        [ 0.0,         0.0,         0.0,        1.0]
    ])
    
    # Current Start Pose
    current_start_pos = positions[0]
    current_start_quat = quaternions[0]
    current_start_rot = R.from_quat(current_start_quat).as_matrix()
    
    current_start_matrix = np.eye(4)
    current_start_matrix[:3, :3] = current_start_rot
    current_start_matrix[:3, 3] = current_start_pos
    
    # Compute Alignment Transform: T_align * T_current = T_target
    # T_align = T_target * inv(T_current)
    t_align = target_start_matrix @ np.linalg.inv(current_start_matrix)
    
    print(f"  [INFO] Aligning coordinate system...")
    print(f"  Alignment Matrix:\n{t_align}")
    
    # Apply to all poses
    # 1. Convert all to matrices
    N = len(positions)
    all_rots = R.from_quat(quaternions).as_matrix()
    all_matrices = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    all_matrices[:, :3, :3] = all_rots
    all_matrices[:, :3, 3] = positions
    
    # 2. Transform
    # New = T_align * Old
    # We can do this efficiently
    new_matrices = t_align @ all_matrices
    
    # 3. Extract back
    positions = new_matrices[:, :3, 3]
    new_rots = new_matrices[:, :3, :3]
    quaternions = R.from_matrix(new_rots).as_quat()
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    print(f"  Trajectory duration: {end_time - start_time:.2f}s")
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [ERROR] Could not open video")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {frame_count} frames, {fps} fps, {width}x{height}")
    
    # 3. Process Frames
    valid_frames = []
    valid_poses = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate target time
        # Assuming video starts at the same time as the trajectory
        current_time = start_time + (i / fps)
        
        if current_time > end_time:
            print(f"  [INFO] Video exceeds trajectory duration at frame {i}. Stopping.")
            break
            
        # Interpolate pose
        pos, quat = interpolate_pose(current_time, timestamps, positions, quaternions)
        
        # Store
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        valid_frames.append(frame_rgb)
        
        # Pose format: x, y, z, qx, qy, qz, qw
        pose = np.concatenate([pos, quat])
        valid_poses.append(pose)
        
        if i % 100 == 0:
            print(f"  Processed {i}/{frame_count} frames...", end='\r')
            
    cap.release()
    print(f"  Processed {len(valid_frames)} frames.          ")
    
    # 4. Save
    # Create output directory
    # Use timestamp for folder name to match convention
    seq_name = f"seq_{int(start_time)}"
    # Append suffix to distinguish if needed, but int timestamp is usually unique enough.
    # However, to be safe and descriptive, maybe I should use the name?
    # The user said "sequence format accepted by the model". The existing ones are seq_TIMESTAMP.
    # I'll stick to seq_TIMESTAMP.
    
    seq_dir = os.path.join(output_root, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    
    print(f"  Saving to {seq_dir}...")
    
    # Save trajectory
    traj_arr = np.array(valid_poses, dtype=np.float32)
    np.save(os.path.join(seq_dir, "trajectory.npy"), traj_arr)
    
    # Save video
    # Use memmap if too large? 
    # For 2612 frames, it's ~7GB. np.save might handle it if RAM is sufficient.
    # Let's try direct save first.
    try:
        video_arr = np.array(valid_frames, dtype=np.uint8)
        np.save(os.path.join(seq_dir, "video.npy"), video_arr)
        print("  Saved successfully.")
    except Exception as e:
        print(f"  [ERROR] Saving video failed: {e}")

if __name__ == "__main__":
    OUTPUT_ROOT = "dataset/sequences"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    process_sequence("lb", OUTPUT_ROOT)
    process_sequence("rb", OUTPUT_ROOT)
