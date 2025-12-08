#!/usr/bin/env python
"""
Correspondence Annotation Tool - Slider Navigation

Navigate the video with keyboard, navigate the centerline with a SLIDER.
Much more responsive than keyboard events!

Usage:
    python annotate_simple.py <sequence_name>
    
    Example: python annotate_simple.py lb

Step 1: Video browser (OpenCV)
    LEFT/RIGHT or A/D  - Navigate frames
    1/5/0              - Set step size (1/5/10)
    ENTER              - Mark frame for annotation
    BACKSPACE          - Unmark last frame
    Q                  - Done, proceed to 3D navigation

Step 2: 3D point selection (PyVista) - for each marked frame
    SLIDER             - Move along centerline (drag it!)
    P                  - Confirm current point
    K                  - Skip this frame
    Q                  - Quit (saves progress)
    M                  - Toggle mesh visibility
"""

import os
import sys
import json
import numpy as np
import cv2
import pyvista as pv
from scipy.spatial.transform import Rotation as R


def load_trajectory(trajectory_path):
    """Load TUM format trajectory."""
    data = np.loadtxt(trajectory_path, comments='#')
    if data[0, 0] > data[-1, 0]:
        data = np.flip(data, axis=0)
    return data[:, 0], data[:, 1:4] * 1000, data[:, 4:8]


def get_sensor_pose(frame_idx, fps, timestamps, positions, quaternions):
    """Get interpolated sensor pose for a frame."""
    start_time = timestamps[0]
    current_time = start_time + (frame_idx / fps)
    
    idx = np.searchsorted(timestamps, current_time)
    
    if idx == 0:
        return positions[0], quaternions[0]
    if idx >= len(timestamps):
        return positions[-1], quaternions[-1]
    
    t0, t1 = timestamps[idx-1], timestamps[idx]
    ratio = (current_time - t0) / (t1 - t0)
    
    pos = (1 - ratio) * positions[idx-1] + ratio * positions[idx]
    
    from scipy.spatial.transform import Slerp
    r = R.concatenate([R.from_quat(quaternions[idx-1]), R.from_quat(quaternions[idx])])
    quat = Slerp([0, 1], r)([ratio])[0].as_quat()
    
    return pos, quat


def browse_video(video_path, existing_frames=None):
    """
    Browse video and mark frames for annotation.
    Returns list of frame indices.
    """
    print("\n" + "="*60)
    print("STEP 1: VIDEO BROWSER")
    print("="*60)
    print("Controls:")
    print("  LEFT/RIGHT or A/D  - Navigate frames")
    print("  1/5/0              - Set step size (1/5/10)")
    print("  ENTER              - Mark frame for annotation")
    print("  BACKSPACE          - Unmark last frame")
    print("  Q                  - Done, proceed to 3D picking")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    current_frame = 0
    marked_frames = list(existing_frames) if existing_frames else []
    step = 1
    
    cv2.namedWindow("Video Browser", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Browser", 1280, 960)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    
    while True:
        display = frame.copy()
        
        # Draw info
        info = [
            f"Frame: {current_frame}/{frame_count-1} (step={step})",
            f"Marked frames: {len(marked_frames)}",
            f"Frames: {marked_frames[:5]}..." if len(marked_frames) > 5 else f"Frames: {marked_frames}",
            "",
            "ENTER=Mark  |  BACKSPACE=Unmark  |  Q=Done",
            "LEFT/RIGHT=Navigate  |  1/5/0=Step size"
        ]
        
        # Check if current frame is marked
        is_marked = current_frame in marked_frames
        if is_marked:
            cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), 
                         (0, 255, 0), 10)
            info.insert(0, ">>> MARKED <<<")
        
        y = 40
        for line in info:
            cv2.putText(display, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 0), 4)
            cv2.putText(display, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
            y += 35
        
        cv2.imshow("Video Browser", display)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 13:  # ENTER
            if current_frame not in marked_frames:
                marked_frames.append(current_frame)
                marked_frames.sort()
                print(f"Marked frame {current_frame}")
        elif key == 8:  # BACKSPACE
            if marked_frames:
                removed = marked_frames.pop()
                print(f"Unmarked frame {removed}")
        elif key == ord('1'):
            step = 1
        elif key == ord('5'):
            step = 5
        elif key == ord('0'):
            step = 10
        elif key == 81 or key == ord('a'):  # LEFT
            current_frame = max(0, current_frame - step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
        elif key == 83 or key == ord('d'):  # RIGHT
            current_frame = min(frame_count - 1, current_frame + step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    
    return marked_frames, fps


def select_centerline_point(frame_idx, video_frame, mesh, centerline_points, 
                            existing_correspondences, frame_num, total_frames):
    """
    Navigate through centerline points using a SLIDER.
    Returns selected point or None if skipped.
    """
    n_points = len(centerline_points)
    
    # State stored in lists for callback access
    current_idx = [n_points // 2]  # Start in middle
    confirmed = [False]
    skipped = [False]
    quit_all = [False]
    mesh_visible = [True]
    marker_actor = [None]
    
    # Create plotter
    plotter = pv.Plotter(
        title=f"Frame {frame_idx} ({frame_num}/{total_frames}) - Use SLIDER, P=Confirm, K=Skip",
        window_size=(1400, 900)
    )
    
    # Add mesh (very transparent)
    plotter.add_mesh(mesh, color='lightblue', opacity=0.1, name='mesh')
    
    # Add centerline as points (rendered as spheres)
    centerline_poly = pv.PolyData(centerline_points)
    plotter.add_mesh(
        centerline_poly, 
        color='yellow', 
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.8, 
        name='centerline'
    )
    
    # Add existing correspondences
    for corr in existing_correspondences:
        sphere = pv.Sphere(radius=2.0, center=corr['model_pos'])
        plotter.add_mesh(sphere, color='lime')
    
    # Initial marker
    current_point = centerline_points[current_idx[0]]
    marker = pv.Sphere(radius=3.5, center=current_point)
    marker_actor[0] = plotter.add_mesh(marker, color='red', name='marker')
    
    def update_marker(value):
        """Callback for slider - update marker position."""
        idx = int(value)
        current_idx[0] = idx
        point = centerline_points[idx]
        
        # Remove old marker and add new one
        plotter.remove_actor('marker')
        new_marker = pv.Sphere(radius=3.5, center=point)
        marker_actor[0] = plotter.add_mesh(new_marker, color='red', name='marker')
    
    # Add slider widget
    plotter.add_slider_widget(
        callback=update_marker,
        rng=[0, n_points - 1],
        value=current_idx[0],
        title="Centerline Position",
        pointa=(0.1, 0.1),
        pointb=(0.9, 0.1),
        style='modern',
        fmt="%.0f"
    )
    
    def close_plotter():
        """Properly close the plotter from a callback."""
        # Use the interactor to properly terminate
        if plotter.iren is not None:
            plotter.iren.terminate_app()
    
    def confirm():
        confirmed[0] = True
        close_plotter()
    
    def skip():
        skipped[0] = True
        close_plotter()
    
    def quit_action():
        quit_all[0] = True
        close_plotter()
    
    def toggle_mesh():
        plotter.remove_actor('mesh')
        if mesh_visible[0]:
            mesh_visible[0] = False
        else:
            plotter.add_mesh(mesh, color='lightblue', opacity=0.1, name='mesh')
            mesh_visible[0] = True
    
    # Key bindings (only for confirm/skip/quit)
    plotter.add_key_event('p', confirm)
    plotter.add_key_event('k', skip)
    plotter.add_key_event('q', quit_action)
    plotter.add_key_event('m', toggle_mesh)
    
    # Add instructions
    plotter.add_text(
        f"Frame {frame_idx} ({frame_num}/{total_frames})\n\n"
        "DRAG the slider below to move\n"
        "along the centerline.\n\n"
        "P = Confirm this point\n"
        "K = Skip this frame\n"
        "M = Toggle mesh\n"
        "Q = Quit",
        position='upper_left',
        font_size=12,
        color='white'
    )
    
    # Show video frame in separate window for reference
    cv2.namedWindow(f"Reference: Frame {frame_idx}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Reference: Frame {frame_idx}", 800, 600)
    display = video_frame.copy()
    cv2.putText(display, f"Frame {frame_idx} - Find this location on 3D model", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(f"Reference: Frame {frame_idx}", display)
    cv2.waitKey(1)
    
    # Show plotter
    plotter.show()
    
    cv2.destroyWindow(f"Reference: Frame {frame_idx}")
    
    if quit_all[0]:
        return None, True  # Signal to quit
    elif confirmed[0]:
        selected_point = centerline_points[current_idx[0]]
        return selected_point.tolist(), False
    else:
        return None, False


def main():
    if len(sys.argv) < 2:
        print("Usage: python annotate_simple.py <sequence_name>")
        print("  Example: python annotate_simple.py lb")
        sys.exit(1)
    
    sequence_name = sys.argv[1]
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, f"dataset/phantom/{sequence_name}.mp4")
    trajectory_path = os.path.join(base_dir, f"dataset/phantom/{sequence_name}.txt")
    mesh_path = os.path.join(base_dir, "patient/lungs.obj")
    centerline_path = os.path.join(base_dir, "patient/centerline.vtk")
    output_path = os.path.join(base_dir, f"dataset/phantom/{sequence_name}_correspondences.json")
    
    # Validate paths
    for path, name in [
        (video_path, "Video"), 
        (trajectory_path, "Trajectory"), 
        (mesh_path, "Mesh"),
        (centerline_path, "Centerline")
    ]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            sys.exit(1)
    
    # Always start fresh - overwrite existing correspondences
    if os.path.exists(output_path):
        print(f"[INFO] Existing correspondences file will be OVERWRITTEN")
    
    # Load trajectory
    timestamps, positions, quaternions = load_trajectory(trajectory_path)
    
    # Step 1: Browse video and mark frames
    marked_frames, fps = browse_video(video_path, existing_frames=None)
    
    if not marked_frames:
        print("No frames marked. Exiting.")
        return
    
    print(f"\nMarked {len(marked_frames)} frames: {marked_frames}")
    
    # All marked frames are new (starting fresh)
    
    print(f"Will annotate {len(marked_frames)} frames: {marked_frames}")
    
    # Load 3D data
    print("\nLoading 3D data...")
    mesh = pv.read(mesh_path)
    centerline = pv.read(centerline_path)
    centerline_points = np.array(centerline.points)
    print(f"  Centerline: {len(centerline_points)} points")
    
    # Load video for frames
    cap = cv2.VideoCapture(video_path)
    
    # Step 2: Navigate centerline for each marked frame
    print("\n" + "="*60)
    print("STEP 2: CENTERLINE NAVIGATION (SLIDER)")
    print("="*60)
    print("For each marked frame:")
    print("  - DRAG the slider to move along centerline")
    print("  - P to confirm current point")
    print("  - K to skip, Q to quit")
    print("  - M to toggle mesh visibility")
    print("="*60)
    
    correspondences = []  # Start fresh
    
    for i, frame_idx in enumerate(marked_frames):
        print(f"\n--- Frame {frame_idx} ({i+1}/{len(marked_frames)}) ---")
        
        # Get video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, video_frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}, skipping")
            continue
        
        # Get sensor pose
        sensor_pos, sensor_quat = get_sensor_pose(
            frame_idx, fps, timestamps, positions, quaternions
        )
        
        # Select point on centerline
        selected_point, should_quit = select_centerline_point(
            frame_idx, video_frame, mesh, centerline_points,
            correspondences, i+1, len(marked_frames)
        )
        
        if should_quit:
            print("Quitting early...")
            break
        
        if selected_point is not None:
            corr = {
                'frame_idx': int(frame_idx),
                'sensor_pos': sensor_pos.tolist(),
                'sensor_quat': sensor_quat.tolist(),
                'model_pos': selected_point
            }
            correspondences.append(corr)
            print(f"  Added: frame {frame_idx} -> [{selected_point[0]:.1f}, {selected_point[1]:.1f}, {selected_point[2]:.1f}]")
            
            # Save after each addition
            with open(output_path, 'w') as f:
                json.dump(correspondences, f, indent=2)
        else:
            print(f"  Skipped frame {frame_idx}")
    
    cap.release()
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(correspondences, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved {len(correspondences)} correspondences to:")
    print(f"  {output_path}")
    print(f"{'='*60}")
    
    # --- COMPUTE TRANSFORMATION ---
    if len(correspondences) >= 3:
        compute_and_visualize_transform(
            sequence_name, base_dir, correspondences, 
            mesh_path, centerline_path, trajectory_path
        )
    else:
        print(f"\nNeed at least 3 correspondences (have {len(correspondences)})")


def umeyama_similarity(src_points, dst_points):
    """
    Compute similarity transform (scale, rotation, translation) 
    that minimizes ||dst - (s * R @ src + t)||^2
    
    This is the Umeyama algorithm for point cloud registration.
    """
    n = src_points.shape[0]
    
    if n < 3:
        raise ValueError("Need at least 3 correspondences for robust estimation")
    
    # Compute centroids
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    
    # Center the points
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    
    # Compute covariance matrix
    H = src_centered.T @ dst_centered / n
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation
    R_mat = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        S[-1] *= -1
        R_mat = Vt.T @ U.T
    
    # Scale
    src_var = np.mean(np.sum(src_centered**2, axis=1))
    s = np.sum(S) / src_var
    
    # Translation
    t = dst_mean - s * R_mat @ src_mean
    
    # Compute error
    transformed = s * (src_points @ R_mat.T) + t
    error = np.sqrt(np.mean(np.sum((transformed - dst_points)**2, axis=1)))
    
    return s, R_mat, t, error


def compute_and_visualize_transform(sequence_name, base_dir, correspondences,
                                     mesh_path, centerline_path, trajectory_path):
    """Compute the transformation and visualize the result."""
    from scipy.spatial.transform import Rotation as Rot
    
    n = len(correspondences)
    
    # Extract points
    src_points = np.array([c['sensor_pos'] for c in correspondences])
    dst_points = np.array([c['model_pos'] for c in correspondences])
    
    print("\n" + "="*60)
    print("COMPUTING SIMILARITY TRANSFORMATION")
    print("="*60)
    
    print("\nSource points (sensor, mm):")
    for i, p in enumerate(src_points):
        print(f"  {i+1}: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]")
    
    print("\nTarget points (3D model):")
    for i, p in enumerate(dst_points):
        print(f"  {i+1}: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]")
    
    # Compute transformation
    s, R_mat, t, error = umeyama_similarity(src_points, dst_points)
    
    # Convert rotation to different representations
    rot = Rot.from_matrix(R_mat)
    euler = rot.as_euler('xyz', degrees=True)
    quat = rot.as_quat()  # [x, y, z, w]
    
    print(f"\nScale factor: {s:.6f}")
    print(f"  (Sensor scale / Model scale = {1/s:.6f})")
    
    print(f"\nRotation matrix:")
    print(f"  [{R_mat[0,0]:>8.5f}, {R_mat[0,1]:>8.5f}, {R_mat[0,2]:>8.5f}]")
    print(f"  [{R_mat[1,0]:>8.5f}, {R_mat[1,1]:>8.5f}, {R_mat[1,2]:>8.5f}]")
    print(f"  [{R_mat[2,0]:>8.5f}, {R_mat[2,1]:>8.5f}, {R_mat[2,2]:>8.5f}]")
    
    print(f"\nRotation (Euler XYZ, degrees): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
    print(f"Rotation (Quaternion xyzw): [{quat[0]:.5f}, {quat[1]:.5f}, {quat[2]:.5f}, {quat[3]:.5f}]")
    
    print(f"\nTranslation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    
    print(f"\nAlignment RMSE: {error:.4f} mm")
    
    # Per-point verification
    transformed = s * (src_points @ R_mat.T) + t
    print("\nPer-point verification:")
    print("-" * 70)
    print(f"{'Point':>5} | {'Transformed':>30} | {'Target':>30} | {'Error':>8}")
    print("-" * 70)
    
    for i in range(len(src_points)):
        trans_str = f"[{transformed[i,0]:8.2f}, {transformed[i,1]:8.2f}, {transformed[i,2]:8.2f}]"
        tgt_str = f"[{dst_points[i,0]:8.2f}, {dst_points[i,1]:8.2f}, {dst_points[i,2]:8.2f}]"
        err = np.linalg.norm(transformed[i] - dst_points[i])
        print(f"{i+1:>5} | {trans_str:>30} | {tgt_str:>30} | {err:>8.2f}")
    
    print("-" * 70)
    
    # Build 4x4 transformation matrix (scaled)
    T = np.eye(4)
    T[:3, :3] = s * R_mat
    T[:3, 3] = t
    
    print("\nFull 4x4 transformation matrix (scale embedded in rotation):")
    print(T)
    
    # Save transformation
    transform_path = os.path.join(base_dir, f"dataset/phantom/{sequence_name}_transform.json")
    transform_data = {
        'scale': float(s),
        'rotation_matrix': R_mat.tolist(),
        'rotation_euler_xyz_deg': euler.tolist(),
        'rotation_quaternion_xyzw': quat.tolist(),
        'translation': t.tolist(),
        'transformation_matrix_4x4': T.tolist(),
        'alignment_rmse_mm': float(error),
        'num_correspondences': n,
        'correspondences': correspondences
    }
    
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    
    print(f"\nTransformation saved to: {transform_path}")
    
    # Print code snippet for build_phantom_dataset.py
    print("\n" + "="*60)
    print("CODE SNIPPET for build_phantom_dataset.py")
    print("="*60)
    print("""
# --- SIMILARITY TRANSFORMATION ---
# Computed from manual correspondences
scale = {:.6f}
R_align = np.array([
    [{:>10.6f}, {:>10.6f}, {:>10.6f}],
    [{:>10.6f}, {:>10.6f}, {:>10.6f}],
    [{:>10.6f}, {:>10.6f}, {:>10.6f}]
])
t_align = np.array([{:.4f}, {:.4f}, {:.4f}])

# Apply: p_model = scale * R_align @ p_sensor + t_align
positions = scale * (positions @ R_align.T) + t_align
""".format(
        s,
        R_mat[0,0], R_mat[0,1], R_mat[0,2],
        R_mat[1,0], R_mat[1,1], R_mat[1,2],
        R_mat[2,0], R_mat[2,1], R_mat[2,2],
        t[0], t[1], t[2]
    ))
    
    # --- VISUALIZATION ---
    visualize_alignment(sequence_name, base_dir, s, R_mat, t, 
                        src_points, dst_points, mesh_path, centerline_path, 
                        trajectory_path)


def visualize_alignment(sequence_name, base_dir, scale, R_mat, t, 
                        src_points, dst_points, mesh_path, centerline_path,
                        trajectory_path):
    """Visualize the transformed trajectory inside the 3D model."""
    
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    # Load mesh
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        return
    mesh = pv.read(mesh_path)
    
    # Load centerline
    centerline = None
    if os.path.exists(centerline_path):
        centerline = pv.read(centerline_path)
    
    # Load full trajectory
    data = np.loadtxt(trajectory_path, comments='#')
    if data[0, 0] > data[-1, 0]:
        data = np.flip(data, axis=0)
    positions = data[:, 1:4] * 1000  # Convert to mm
    
    # Apply transformation
    positions_aligned = scale * (positions @ R_mat.T) + t
    
    print(f"Trajectory: {len(positions)} points")
    print("Close the visualization window to finish.")
    
    # Create plotter
    plotter = pv.Plotter(title=f"Alignment Visualization: {sequence_name}")
    
    # Add mesh (semi-transparent)
    plotter.add_mesh(mesh, color='lightblue', opacity=0.2, label='Lungs')
    
    # Add centerline
    if centerline is not None:
        plotter.add_mesh(
            centerline, 
            color='gray', 
            point_size=2,
            render_points_as_spheres=True,
            opacity=0.3,
            label='Centerline'
        )
    
    # Add aligned trajectory as line
    # Subsample for performance
    step = max(1, len(positions_aligned) // 500)
    traj_points = positions_aligned[::step]
    if len(traj_points) > 1:
        line = pv.lines_from_points(traj_points)
        plotter.add_mesh(line, color='blue', line_width=3, label='Aligned Trajectory')
    
    # Add start and end markers
    start_sphere = pv.Sphere(radius=2.0, center=positions_aligned[0])
    end_sphere = pv.Sphere(radius=2.0, center=positions_aligned[-1])
    plotter.add_mesh(start_sphere, color='lime', label='Start')
    plotter.add_mesh(end_sphere, color='orange', label='End')
    
    # Add correspondence points
    # Transformed source points
    src_transformed = scale * (src_points @ R_mat.T) + t
    
    for i in range(len(src_points)):
        # Target point (green)
        tgt_sphere = pv.Sphere(radius=1.5, center=dst_points[i])
        plotter.add_mesh(tgt_sphere, color='green')
        
        # Transformed source point (red)
        src_sphere = pv.Sphere(radius=1.2, center=src_transformed[i])
        plotter.add_mesh(src_sphere, color='red')
        
        # Line connecting them to show error
        error_line = pv.Line(src_transformed[i], dst_points[i])
        plotter.add_mesh(error_line, color='yellow', line_width=2)
    
    # Add legend explanation
    plotter.add_text(
        f"Sequence: {sequence_name}\n"
        f"RMSE: {np.sqrt(np.mean(np.sum((src_transformed - dst_points)**2, axis=1))):.2f} mm\n\n"
        "Green spheres: Target (model)\n"
        "Red spheres: Transformed source\n"
        "Yellow lines: Alignment errors\n"
        "Blue line: Full trajectory",
        position='upper_left',
        font_size=10,
        color='white'
    )
    
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()
