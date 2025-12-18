#!/usr/bin/env python
"""
Align Phantom Trajectory

Computes the similarity transformation to align phantom sensor trajectories
with the 3D lung model.

Usage:
    # Full workflow: annotate correspondences and compute transform
    python align_phantom_traj.py <sequence_name>
    
    # Reuse existing correspondences (skip annotation)
    python align_phantom_traj.py <sequence_name> --reuse

Options:
    --reuse    Skip annotation and use existing correspondences from JSON file

The transformation ensures that all trajectory points remain INSIDE the mesh,
since a bronchoscope cannot be outside the airways.
"""

import os
import sys
import json
import argparse
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


def check_points_inside_mesh(points, mesh):
    """
    Check which points are inside the mesh.
    
    Returns:
        inside_mask: boolean array, True if point is inside
        distances: signed distances (negative = inside, positive = outside)
    """
    # Create a PolyData from points
    point_cloud = pv.PolyData(points)
    
    # Use select_enclosed_points to check containment
    # This works by casting rays and checking intersections
    selection = point_cloud.select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
    
    # Get the mask
    inside_mask = selection['SelectedPoints'].astype(bool)
    
    # Compute signed distances for more detailed info
    # Negative = inside, Positive = outside
    # Note: This uses the closest point on surface
    closest_cells, closest_points = mesh.find_closest_cell(points, return_closest_point=True)
    distances = np.linalg.norm(points - closest_points, axis=1)
    
    # Sign the distances: negative if inside
    signed_distances = np.where(inside_mask, -distances, distances)
    
    return inside_mask, signed_distances


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
        
        info = [
            f"Frame: {current_frame}/{frame_count-1} (step={step})",
            f"Marked frames: {len(marked_frames)}",
            f"Frames: {marked_frames[:5]}..." if len(marked_frames) > 5 else f"Frames: {marked_frames}",
            "",
            "ENTER=Mark  |  BACKSPACE=Unmark  |  Q=Done",
            "LEFT/RIGHT=Navigate  |  1/5/0=Step size"
        ]
        
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
    """Navigate through centerline points using a SLIDER."""
    n_points = len(centerline_points)
    
    current_idx = [n_points // 2]
    confirmed = [False]
    skipped = [False]
    quit_all = [False]
    mesh_visible = [True]
    
    plotter = pv.Plotter(
        title=f"Frame {frame_idx} ({frame_num}/{total_frames}) - Use SLIDER, P=Confirm, K=Skip",
        window_size=(1400, 900)
    )
    
    plotter.add_mesh(mesh, color='lightblue', opacity=0.1, name='mesh')
    
    centerline_poly = pv.PolyData(centerline_points)
    plotter.add_mesh(
        centerline_poly, color='yellow', point_size=5,
        render_points_as_spheres=True, opacity=0.8, name='centerline'
    )
    
    for corr in existing_correspondences:
        sphere = pv.Sphere(radius=1.0, center=corr['model_pos'])
        plotter.add_mesh(sphere, color='lime')
    
    current_point = centerline_points[current_idx[0]]
    marker = pv.Sphere(radius=1.75, center=current_point)
    plotter.add_mesh(marker, color='red', name='marker')
    
    def update_marker(value):
        idx = int(value)
        current_idx[0] = idx
        point = centerline_points[idx]
        plotter.remove_actor('marker')
        new_marker = pv.Sphere(radius=1.75, center=point)
        plotter.add_mesh(new_marker, color='red', name='marker')
    
    plotter.add_slider_widget(
        callback=update_marker, rng=[0, n_points - 1], value=current_idx[0],
        title="Centerline Position", pointa=(0.1, 0.1), pointb=(0.9, 0.1),
        style='modern', fmt="%.0f"
    )
    
    def close_plotter():
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
    
    plotter.add_key_event('p', confirm)
    plotter.add_key_event('k', skip)
    plotter.add_key_event('q', quit_action)
    plotter.add_key_event('m', toggle_mesh)
    
    plotter.add_text(
        f"Frame {frame_idx} ({frame_num}/{total_frames})\n\n"
        "DRAG the slider below to move\nalong the centerline.\n\n"
        "P = Confirm this point\nK = Skip this frame\nM = Toggle mesh\nQ = Quit",
        position='upper_left', font_size=12, color='white'
    )
    
    cv2.namedWindow(f"Reference: Frame {frame_idx}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Reference: Frame {frame_idx}", 800, 600)
    display = video_frame.copy()
    cv2.putText(display, f"Frame {frame_idx} - Find this location on 3D model", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(f"Reference: Frame {frame_idx}", display)
    cv2.waitKey(1)
    
    plotter.show()
    
    cv2.destroyWindow(f"Reference: Frame {frame_idx}")
    
    if quit_all[0]:
        return None, True
    elif confirmed[0]:
        selected_point = centerline_points[current_idx[0]]
        return selected_point.tolist(), False
    else:
        return None, False


def umeyama_similarity(src_points, dst_points):
    """Compute similarity transform using Umeyama algorithm."""
    n = src_points.shape[0]
    
    if n < 3:
        raise ValueError("Need at least 3 correspondences")
    
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    
    H = src_centered.T @ dst_centered / n
    U, S, Vt = np.linalg.svd(H)
    
    R_mat = Vt.T @ U.T
    
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        S[-1] *= -1
        R_mat = Vt.T @ U.T
    
    src_var = np.mean(np.sum(src_centered**2, axis=1))
    s = np.sum(S) / src_var
    
    t = dst_mean - s * R_mat @ src_mean
    
    transformed = s * (src_points @ R_mat.T) + t
    error = np.sqrt(np.mean(np.sum((transformed - dst_points)**2, axis=1)))
    
    return s, R_mat, t, error


def refine_transform_constrained(s_init, R_init, t_init, src_points, dst_points,
                                  trajectory_positions, mesh, 
                                  outside_penalty=1.0, max_iterations=200):
    """
    Refine transformation to minimize points outside mesh.
    
    Uses scipy.optimize to minimize:
        objective = correspondence_error + outside_penalty * num_outside_points
    
    Args:
        s_init, R_init, t_init: Initial transform from Umeyama
        src_points: Correspondence source points
        dst_points: Correspondence target points
        trajectory_positions: Full trajectory to check containment
        mesh: PyVista mesh for containment checking
        outside_penalty: Weight for outside points (higher = stricter)
        max_iterations: Max optimization iterations
    
    Returns:
        s, R_mat, t, error, n_outside
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as Rot
    
    # Subsample trajectory for faster optimization
    step = max(1, len(trajectory_positions) // 200)
    traj_sample = trajectory_positions[::step]
    
    # Initial parameters: [scale, euler_x, euler_y, euler_z, tx, ty, tz]
    euler_init = Rot.from_matrix(R_init).as_euler('xyz', degrees=False)
    x0 = np.array([s_init, euler_init[0], euler_init[1], euler_init[2],
                   t_init[0], t_init[1], t_init[2]])
    
    # Pre-compute mesh for faster containment checking
    def count_outside(positions, mesh):
        """Count points outside mesh."""
        if len(positions) == 0:
            return 0
        point_cloud = pv.PolyData(positions)
        try:
            selection = point_cloud.select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
            inside_mask = selection['SelectedPoints'].astype(bool)
            return (~inside_mask).sum()
        except:
            return len(positions)  # Assume all outside on error
    
    def objective(x):
        """Combined objective: correspondence error + outside penalty."""
        s = x[0]
        euler = x[1:4]
        t = x[4:7]
        
        # Build rotation matrix
        R_mat = Rot.from_euler('xyz', euler, degrees=False).as_matrix()
        
        # Correspondence error
        transformed_src = s * (src_points @ R_mat.T) + t
        corr_error = np.sqrt(np.mean(np.sum((transformed_src - dst_points)**2, axis=1)))
        
        # Containment penalty (subsample for speed)
        traj_transformed = s * (traj_sample @ R_mat.T) + t
        n_outside = count_outside(traj_transformed, mesh)
        outside_ratio = n_outside / len(traj_sample)
        
        # Combined objective
        total = corr_error + outside_penalty * outside_ratio * 100
        
        return total
    
    print(f"\n  Refining transformation to minimize outside points...")
    print(f"  Initial outside: {count_outside(s_init * (traj_sample @ R_init.T) + t_init, mesh)}/{len(traj_sample)}")
    
    # Bounds: scale > 0.1, others unbounded
    bounds = [(0.1, 2.0),  # scale
              (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),  # euler
              (None, None), (None, None), (None, None)]  # translation
    
    result = minimize(
        objective, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'disp': False}
    )
    
    # Extract results
    s = result.x[0]
    euler = result.x[1:4]
    t = result.x[4:7]
    R_mat = Rot.from_euler('xyz', euler, degrees=False).as_matrix()
    
    # Compute final metrics
    transformed_src = s * (src_points @ R_mat.T) + t
    error = np.sqrt(np.mean(np.sum((transformed_src - dst_points)**2, axis=1)))
    
    # Final outside count on full trajectory
    traj_transformed = s * (trajectory_positions @ R_mat.T) + t
    n_outside = count_outside(traj_transformed, mesh)
    
    print(f"  Refined outside: {n_outside}/{len(trajectory_positions)}")
    print(f"  Optimization converged: {result.success}")
    
    return s, R_mat, t, error, n_outside


def annotate_correspondences(sequence_name, base_dir, video_path, trajectory_path,
                              mesh_path, centerline_path, output_path):
    """Run the annotation workflow."""
    # Load trajectory
    timestamps, positions, quaternions = load_trajectory(trajectory_path)
    
    # Browse video and mark frames
    marked_frames, fps = browse_video(video_path, existing_frames=None)
    
    if not marked_frames:
        print("No frames marked. Exiting.")
        return None
    
    print(f"\nMarked {len(marked_frames)} frames: {marked_frames}")
    
    # Load 3D data
    print("\nLoading 3D data...")
    mesh = pv.read(mesh_path)
    centerline = pv.read(centerline_path)
    centerline_points = np.array(centerline.points)
    print(f"  Centerline: {len(centerline_points)} points")
    
    cap = cv2.VideoCapture(video_path)
    
    print("\n" + "="*60)
    print("STEP 2: CENTERLINE NAVIGATION (SLIDER)")
    print("="*60)
    
    correspondences = []
    
    for i, frame_idx in enumerate(marked_frames):
        print(f"\n--- Frame {frame_idx} ({i+1}/{len(marked_frames)}) ---")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, video_frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}, skipping")
            continue
        
        sensor_pos, sensor_quat = get_sensor_pose(
            frame_idx, fps, timestamps, positions, quaternions
        )
        
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
            
            with open(output_path, 'w') as f:
                json.dump(correspondences, f, indent=2)
        else:
            print(f"  Skipped frame {frame_idx}")
    
    cap.release()
    
    with open(output_path, 'w') as f:
        json.dump(correspondences, f, indent=2)
    
    print(f"\nSaved {len(correspondences)} correspondences to: {output_path}")
    
    return correspondences


def compute_and_validate_transform(sequence_name, base_dir, correspondences,
                                    mesh_path, centerline_path, trajectory_path,
                                    apply_constraint=False):
    """Compute transformation and validate that all points are inside mesh."""
    from scipy.spatial.transform import Rotation as Rot
    
    n = len(correspondences)
    
    src_points = np.array([c['sensor_pos'] for c in correspondences])
    dst_points = np.array([c['model_pos'] for c in correspondences])
    
    print("\n" + "="*60)
    print("STEP 1: UMEYAMA (INITIAL ESTIMATE)")
    print("="*60)
    
    # Compute initial transformation
    s, R_mat, t, error = umeyama_similarity(src_points, dst_points)
    
    rot = Rot.from_matrix(R_mat)
    euler = rot.as_euler('xyz', degrees=True)
    
    print(f"\nScale factor: {s:.6f}")
    print(f"Alignment RMSE: {error:.4f} mm")
    
    # Load trajectory and mesh
    data = np.loadtxt(trajectory_path, comments='#')
    if data[0, 0] > data[-1, 0]:
        data = np.flip(data, axis=0)
    positions = data[:, 1:4] * 1000  # Convert to mm
    
    mesh = pv.read(mesh_path)
    
    # Check initial containment
    positions_aligned = s * (positions @ R_mat.T) + t
    inside_mask, _ = check_points_inside_mesh(positions_aligned, mesh)
    n_outside_initial = (~inside_mask).sum()
    pct_inside_initial = 100 * inside_mask.sum() / len(inside_mask)
    
    print(f"\nInitial containment:")
    print(f"  Inside: {inside_mask.sum()}/{len(inside_mask)} ({pct_inside_initial:.1f}%)")
    print(f"  Outside: {n_outside_initial}")
    
    # If apply_constraint is set and there are points outside, refine the transformation
    if apply_constraint and n_outside_initial > 0:
        print("\n" + "="*60)
        print("STEP 2: CONSTRAINED REFINEMENT")
        print("="*60)
        
        s, R_mat, t, error, n_outside = refine_transform_constrained(
            s, R_mat, t, src_points, dst_points,
            positions, mesh,
            outside_penalty=2.0,  # Weight for outside penalty
            max_iterations=300
        )
        
        rot = Rot.from_matrix(R_mat)
        euler = rot.as_euler('xyz', degrees=True)
        
        print(f"\nRefined scale factor: {s:.6f}")
        print(f"Refined RMSE: {error:.4f} mm")
        
        # Update aligned positions
        positions_aligned = s * (positions @ R_mat.T) + t
    elif n_outside_initial > 0 and not apply_constraint:
        print("\n  [TIP] Use --constraint flag to refine transformation and minimize outside points")
    
    # Final validation
    print("\n" + "="*60)
    print("FINAL VALIDATION")
    print("="*60)
    
    inside_mask, signed_distances = check_points_inside_mesh(positions_aligned, mesh)
    
    n_inside = inside_mask.sum()
    n_outside = len(inside_mask) - n_inside
    pct_inside = 100 * n_inside / len(inside_mask)
    
    print(f"\n  Points INSIDE mesh:  {n_inside:>6} ({pct_inside:.1f}%)")
    print(f"  Points OUTSIDE mesh: {n_outside:>6} ({100-pct_inside:.1f}%)")
    
    if n_outside > 0:
        outside_distances = signed_distances[~inside_mask]
        print(f"\n  Outside point statistics:")
        print(f"    Max distance outside: {outside_distances.max():.2f} mm")
        print(f"    Mean distance outside: {outside_distances.mean():.2f} mm")
        print(f"\n  [WARNING] {n_outside} points are still outside the mesh!")
    else:
        print("\n  [OK] All trajectory points are inside the mesh!")
    
    # Per-point correspondence verification
    transformed = s * (src_points @ R_mat.T) + t
    print("\nPer-point verification:")
    print("-" * 70)
    for i in range(len(src_points)):
        err = np.linalg.norm(transformed[i] - dst_points[i])
        print(f"  Point {i+1}: Error = {err:.2f} mm")
    print("-" * 70)
    
    # Build 4x4 transformation matrix
    quat = rot.as_quat()
    T = np.eye(4)
    T[:3, :3] = s * R_mat
    T[:3, 3] = t
    
    # Save transformation with validation info
    transform_path = os.path.join(base_dir, "dataset", "phantom", "data", f"{sequence_name}_transform.json")
    transform_data = {
        'scale': float(s),
        'rotation_matrix': R_mat.tolist(),
        'rotation_euler_xyz_deg': euler.tolist(),
        'rotation_quaternion_xyzw': quat.tolist(),
        'translation': t.tolist(),
        'transformation_matrix_4x4': T.tolist(),
        'alignment_rmse_mm': float(error),
        'num_correspondences': n,
        'validation': {
            'total_points': int(len(inside_mask)),
            'points_inside': int(n_inside),
            'points_outside': int(n_outside),
            'percent_inside': float(pct_inside),
            'max_outside_distance_mm': float(signed_distances[~inside_mask].max()) if n_outside > 0 else 0.0
        },
        'correspondences': correspondences
    }
    
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    
    print(f"\nTransformation saved to: {transform_path}")
    
    # Visualize
    visualize_alignment(sequence_name, mesh, positions_aligned, 
                        src_points, dst_points, s, R_mat, t,
                        inside_mask, centerline_path)
    
    return transform_data


def visualize_alignment(sequence_name, mesh, positions_aligned, 
                        src_points, dst_points, scale, R_mat, t,
                        inside_mask, centerline_path):
    """Visualize alignment with inside/outside coloring."""
    
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    centerline = None
    if os.path.exists(centerline_path):
        centerline = pv.read(centerline_path)
    
    plotter = pv.Plotter(title=f"Alignment Visualization: {sequence_name}")
    
    # Add mesh
    plotter.add_mesh(mesh, color='lightblue', opacity=0.2, label='Lungs')
    
    # Add centerline
    if centerline is not None:
        plotter.add_mesh(centerline, color='gray', point_size=2,
                        render_points_as_spheres=True, opacity=0.3)
    
    # Add trajectory with color based on inside/outside
    step = max(1, len(positions_aligned) // 500)
    traj_points = positions_aligned[::step]
    traj_inside = inside_mask[::step]
    
    # Points inside = blue, outside = red
    colors = np.where(traj_inside, 0, 1)  # 0=inside, 1=outside
    
    if len(traj_points) > 1:
        traj_poly = pv.PolyData(traj_points)
        traj_poly['outside'] = colors.astype(float)
        plotter.add_mesh(traj_poly, scalars='outside', cmap=['blue', 'red'],
                        point_size=5, render_points_as_spheres=True,
                        show_scalar_bar=False)
    
    # Add start/end markers
    start_sphere = pv.Sphere(radius=1.0, center=positions_aligned[0])
    end_sphere = pv.Sphere(radius=1.0, center=positions_aligned[-1])
    plotter.add_mesh(start_sphere, color='lime', label='Start')
    plotter.add_mesh(end_sphere, color='orange', label='End')
    
    # Add correspondence points
    src_transformed = scale * (src_points @ R_mat.T) + t
    
    for i in range(len(src_points)):
        tgt_sphere = pv.Sphere(radius=0.75, center=dst_points[i])
        plotter.add_mesh(tgt_sphere, color='green')
        
        src_sphere = pv.Sphere(radius=0.6, center=src_transformed[i])
        plotter.add_mesh(src_sphere, color='magenta')
        
        error_line = pv.Line(src_transformed[i], dst_points[i])
        plotter.add_mesh(error_line, color='yellow', line_width=2)
    
    n_outside = (~inside_mask).sum()
    pct_inside = 100 * inside_mask.sum() / len(inside_mask)
    
    plotter.add_text(
        f"Sequence: {sequence_name}\n"
        f"Inside: {pct_inside:.1f}% | Outside: {n_outside} points\n\n"
        "Blue = Inside mesh\n"
        "Red = Outside mesh\n"
        "Green = Target correspondences\n"
        "Magenta = Transformed source",
        position='upper_left', font_size=10, color='white'
    )
    
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description='Align phantom trajectory with 3D model')
    parser.add_argument('sequence_name', help='Name of the sequence (e.g., lb)')
    parser.add_argument('--reuse', action='store_true',
                       help='Reuse existing correspondences from JSON file')
    parser.add_argument('--constraint', action='store_true',
                       help='Apply constrained optimization to minimize points outside mesh')
    parser.add_argument('--centerline', type=str, default=None,
                       help='Path to centerline .vtk file (default: patient/centerline.vtk)')
    args = parser.parse_args()
    
    sequence_name = args.sequence_name
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "dataset", "phantom", "data")
    patient_dir = os.path.join(base_dir, "patient")
    
    # Try different video formats
    video_path = None
    for ext in ['.mp4', '.mkv', '.avi']:
        candidate = os.path.join(data_dir, f"{sequence_name}{ext}")
        if os.path.exists(candidate):
            video_path = candidate
            break
    
    trajectory_path = os.path.join(data_dir, f"{sequence_name}_gt.txt")
    mesh_path = os.path.join(patient_dir, "lungs.obj")
    
    # Determine centerline path
    if args.centerline:
        centerline_path = args.centerline
    else:
        # Look up centerline from CSV
        csv_path = os.path.join(data_dir, "closest_centerline.csv")
        centerline_name = None
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2 and parts[0] == sequence_name:
                        centerline_name = parts[1]
                        break
        
        if centerline_name:
            centerline_path = os.path.join(patient_dir, "centerlines", f"{centerline_name}.vtp")
            print(f"Using centerline from CSV lookup: {centerline_name}")
        else:
            centerline_path = os.path.join(patient_dir, "centerline.vtk")
            print(f"No CSV mapping found for '{sequence_name}', using default centerline")
    
    corr_path = os.path.join(data_dir, f"{sequence_name}_correspondences.json")
    
    # Validate paths
    for path, name in [
        (trajectory_path, "Trajectory"), 
        (mesh_path, "Mesh"),
        (centerline_path, "Centerline")
    ]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            sys.exit(1)
    
    if args.reuse:
        # Reuse existing correspondences
        print("="*60)
        print(f"REUSING EXISTING CORRESPONDENCES: {sequence_name}")
        print("="*60)
        
        if not os.path.exists(corr_path):
            print(f"Error: Correspondences file not found: {corr_path}")
            print("Run without --reuse first to create correspondences.")
            sys.exit(1)
        
        with open(corr_path, 'r') as f:
            correspondences = json.load(f)
        
        print(f"Loaded {len(correspondences)} correspondences from: {corr_path}")
    else:
        # Full annotation workflow
        if video_path is None:
            print(f"Error: No video found for sequence '{sequence_name}' (tried .mp4, .mkv, .avi)")
            sys.exit(1)
        
        if os.path.exists(corr_path):
            print(f"[INFO] Existing correspondences will be OVERWRITTEN")
        
        correspondences = annotate_correspondences(
            sequence_name, base_dir, video_path, trajectory_path,
            mesh_path, centerline_path, corr_path
        )
        
        if correspondences is None:
            return
    
    # Compute and validate transformation
    if len(correspondences) >= 3:
        compute_and_validate_transform(
            sequence_name, base_dir, correspondences,
            mesh_path, centerline_path, trajectory_path,
            apply_constraint=args.constraint
        )
    else:
        print(f"\nNeed at least 3 correspondences (have {len(correspondences)})")


if __name__ == "__main__":
    main()
