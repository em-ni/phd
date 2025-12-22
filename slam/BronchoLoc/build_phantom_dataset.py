#!/usr/bin/env python
"""
Build Phantom Dataset

This script processes phantom bronchoscopy recordings and creates
sequence folders in the same format as the simulator recordings.

It uses the transformations computed by align_phantom_traj.py to align
the sensor trajectories with the 3D lung model.

Usage:
    python build_phantom_dataset.py

Requirements:
    For each video (e.g., lb.mp4), there must be:
    - dataset/phantom/lb.txt (TUM trajectory from magnetic sensor)
    - dataset/phantom/lb_transform.json (computed by align_phantom_traj.py)

Output:
    Creates sequence folders in dataset/sequences/seq_phantom_<name>/
    with trajectory.npy and video.npy
"""

import os
import glob
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.signal import savgol_filter


# ===========================================================================
# SMOOTHING FUNCTIONS
# ===========================================================================
def smooth_positions_savgol(positions, window_length=300, polyorder=3):
    """
    Smooth trajectory positions using Savitzky-Golay filter.
    This reduces magnetic sensor noise while preserving trajectory shape.
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    if window_length > len(positions):
        window_length = len(positions) // 2 * 2 + 1
        if window_length <= polyorder:
            return positions.copy()
    
    smoothed = np.zeros_like(positions)
    for i in range(3):
        smoothed[:, i] = savgol_filter(positions[:, i], window_length, polyorder)
    return smoothed


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
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    
    return timestamps, positions, quaternions


def interpolate_pose(target_time, timestamps, positions, quaternions):
    """
    Interpolates pose at target_time.
    """
    idx = np.searchsorted(timestamps, target_time)
    
    if idx == 0:
        return positions[0], quaternions[0]
    if idx >= len(timestamps):
        return positions[-1], quaternions[-1]
        
    t0 = timestamps[idx-1]
    t1 = timestamps[idx]
    ratio = (target_time - t0) / (t1 - t0)
    
    # Position interpolation (Linear)
    p_interp = (1 - ratio) * positions[idx-1] + ratio * positions[idx]
    
    # Rotation interpolation (SLERP)
    key_rots = R.from_quat([quaternions[idx-1], quaternions[idx]])
    slerp = Slerp([0, 1], key_rots)
    q_interp = slerp([ratio])[0].as_quat()
    
    return p_interp, q_interp


def process_sequence(name, phantom_dir, output_root):
    """
    Process a single phantom sequence.
    
    Args:
        name: Sequence name (e.g., 'lb')
        phantom_dir: Path to phantom data folder
        output_root: Output folder for sequences
    """
    txt_path = os.path.join(phantom_dir, f"{name}_gt.txt")
    
    # Try different video formats
    video_path = None
    for ext in ['.mp4', '.mkv', '.avi']:
        candidate = os.path.join(phantom_dir, f"{name}{ext}")
        if os.path.exists(candidate):
            video_path = candidate
            break
    
    transform_path = os.path.join(phantom_dir, f"{name}_transform.json")
    
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    # Check required files
    if not os.path.exists(txt_path):
        print(f"  [ERROR] Trajectory not found: {txt_path}")
        return False
    
    if video_path is None or not os.path.exists(video_path):
        print(f"  [ERROR] Video not found for: {name} (tried .mp4, .mkv, .avi)")
        return False
    
    if not os.path.exists(transform_path):
        print(f"  [ERROR] Transform not found: {transform_path}")
        print(f"  Run: python align_phantom_traj.py {name}")
        return False
    
    # 1. Load Transform
    print(f"  Loading transform: {transform_path}")
    with open(transform_path, 'r') as f:
        transform_data = json.load(f)
    
    scale = transform_data['scale']
    R_mat = np.array(transform_data['rotation_matrix'])
    t_vec = np.array(transform_data['translation'])
    rmse = transform_data['alignment_rmse_mm']
    
    print(f"    Scale: {scale:.4f}")
    print(f"    RMSE: {rmse:.2f} mm")
    
    # 2. Load Trajectory
    print(f"  Loading trajectory: {txt_path}")
    timestamps, positions, quaternions = load_tum_trajectory(txt_path)
    print(f"    {len(timestamps)} poses")
    
    # Scale positions from meters to millimeters
    positions = positions * 1000.0
    
    # 3. Apply Similarity Transformation
    # p_model = scale * R @ p_sensor + t
    print(f"  Applying similarity transformation...")
    positions_aligned = scale * (positions @ R_mat.T) + t_vec
    
    # For quaternions, we need to apply the rotation component
    # q_model = R_align * q_sensor
    R_align = R.from_matrix(R_mat)
    original_rots = R.from_quat(quaternions)
    aligned_rots = R_align * original_rots
    quaternions_aligned = aligned_rots.as_quat()
    
    # 4. Smooth Trajectory (Reduces magnetic sensor noise)
    # NOTE: Only positions are smoothed - orientation is not used in GT
    print(f"  Smoothing trajectory (window=300)...")
    positions_aligned = smooth_positions_savgol(positions_aligned, window_length=300)
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration = end_time - start_time
    print(f"    Trajectory duration: {duration:.2f}s")
    
    # 4. Open Video
    print(f"  Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [ERROR] Could not open video")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    {frame_count} frames, {fps:.1f} fps, {width}x{height}")
    
    # 5. Process Frames
    print(f"  Processing frames...")
    valid_frames = []
    valid_poses = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate target time
        current_time = start_time + (i / fps)
        
        if current_time > end_time:
            print(f"  [INFO] Video exceeds trajectory at frame {i}. Stopping.")
            break
            
        # Interpolate pose
        pos, quat = interpolate_pose(
            current_time, timestamps, 
            positions_aligned, quaternions_aligned
        )
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        valid_frames.append(frame_rgb)
        
        # Pose format: x, y, z, qx, qy, qz, qw
        pose = np.concatenate([pos, quat])
        valid_poses.append(pose)
        
        if i % 200 == 0:
            print(f"    {i}/{frame_count} frames...", end='\r')
            
    cap.release()
    print(f"    Processed {len(valid_frames)} frames.          ")
    
    # 6. Save
    seq_name = f"seq_phantom_{name}"
    seq_dir = os.path.join(output_root, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    
    print(f"  Saving to: {seq_dir}")
    
    # Save trajectory
    traj_arr = np.array(valid_poses, dtype=np.float32)
    np.save(os.path.join(seq_dir, "trajectory.npy"), traj_arr)
    print(f"    trajectory.npy: {traj_arr.shape}")
    
    # Save video
    try:
        video_arr = np.array(valid_frames, dtype=np.uint8)
        np.save(os.path.join(seq_dir, "video.npy"), video_arr)
        print(f"    video.npy: {video_arr.shape} ({video_arr.nbytes / 1e9:.2f} GB)")
    except MemoryError:
        print(f"  [ERROR] Out of memory saving video. Try using memmap.")
        return False
    
    print(f"  Done!")
    return True


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    phantom_dir = os.path.join(base_dir, "dataset", "phantom", "data")
    output_root = os.path.join(base_dir, "dataset", "sequences")
    
    os.makedirs(output_root, exist_ok=True)
    
    print("="*60)
    print("BUILD PHANTOM DATASET")
    print("="*60)
    print(f"Phantom folder: {phantom_dir}")
    print(f"Output folder: {output_root}")
    
    # Find all videos (.mp4 and .mkv)
    video_files = glob.glob(os.path.join(phantom_dir, "*.mp4")) + glob.glob(os.path.join(phantom_dir, "*.mkv"))
    
    if not video_files:
        print(f"\n[ERROR] No videos found in {phantom_dir}")
        return
    
    # Extract names
    all_names = [os.path.splitext(os.path.basename(v))[0] for v in video_files]
    
    # Filter: prefer _part1 version over base version
    # e.g. if both "3" and "3_part1" exist, only keep "3_part1"
    base_names = set()
    part1_names = set()
    for name in all_names:
        if '_part1' in name:
            part1_names.add(name)
            # Extract base name (e.g. "3" from "3_part1")
            base = name.replace('_part1', '')
            base_names.add(base)
    
    # Remove base names that have a _part1 version
    names = [n for n in all_names if n not in base_names]
    names = sorted(set(names))  # Remove duplicates and sort
    
    print(f"\nFound {len(all_names)} videos, processing {len(names)}: {names}")
    
    # Process each
    success = []
    failed = []
    
    for name in names:
        try:
            if process_sequence(name, phantom_dir, output_root):
                success.append(name)
            else:
                failed.append(name)
        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            failed.append(name)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful: {len(success)} - {success}")
    print(f"Failed: {len(failed)} - {failed}")
    
    if failed:
        print("\nTo fix failed sequences, run align_phantom_traj.py for each:")
        for name in failed:
            print(f"  python align_phantom_traj.py {name}")


if __name__ == "__main__":
    main()
