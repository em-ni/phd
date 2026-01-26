import numpy as np
import glob
import os
import sys
import argparse
import pyvista as pv
import random
from scipy.spatial import cKDTree
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from constants import MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS, MAP_POINT_SPACING
from utils.utils import load_centerline_points, filter_connected_component, density_based_sample


def visualize_window_3d(positions, frame_indices, lung_path, centerline_path, seq_name, max_points=DEFAULT_MAX_MAP_POINTS):
    """
    Visualizes window frame positions in 3D context.
    Uses ball query centered at first frame (like ant_dataset.py).
    
    Args:
        positions: (N, 3) array of all trajectory positions
        frame_indices: list of indices for the window frames
        lung_path: path to lungs.obj
        centerline_path: path to centerline centerline .npz
        seq_name: sequence name for title
        max_points: max points after FPS (same as dataset)
    """
    p = pv.Plotter(title=f"Window 3D + FPS: {seq_name}")
    centerline_pts = None
    ball_points = None
    fps_points = None
    
    # Get window positions
    window_positions = positions[frame_indices]
    p0 = window_positions[0]  # First frame position (center of ball query)
    
    # 1. Load and plot Lungs (Ghostly)
    if os.path.exists(lung_path):
        lung_mesh = pv.read(lung_path)
        p.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
    
    # 2. Load Centerline and find points in ball
    centerline_pts = load_centerline_points(centerline_path)
    if centerline_pts is not None:
        # Plot full centerline (very faint)
        p.add_mesh(pv.PolyData(centerline_pts), color='black', opacity=0.1, 
                  point_size=2, render_points_as_spheres=True, label='Full Centerline')
        
        # Ball query: find points within MAP_QUERY_RADIUS of first frame
        tree = cKDTree(centerline_pts)
        ball_indices = tree.query_ball_point(p0, r=MAP_QUERY_RADIUS)
        ball_points = centerline_pts[ball_indices]
        
        # Apply DBSCAN filtering (same as dataset)
        if len(ball_points) > 0:
            connected_points, _ = filter_connected_component(p0, ball_points)
        else:
            connected_points = ball_points
        
        # Apply density-based downsampling (same as dataset)
        if len(connected_points) > 0:
            dists = np.linalg.norm(connected_points - p0, axis=1)
            start_idx = np.argmin(dists)
            fps_points, _ = density_based_sample(
                connected_points, 
                min_distance=MAP_POINT_SPACING, 
                start_idx=start_idx,
                max_points=max_points
            )
        else:
            fps_points = connected_points
        
        # Plot connected points (faint orange)
        if len(connected_points) > 0:
            p.add_mesh(pv.PolyData(connected_points), color='orange', opacity=0.3, 
                      point_size=4, render_points_as_spheres=True, label=f'Connected ({len(connected_points)})')
        
        # Plot FPS points (bright red - what model sees)
        if len(fps_points) > 0:
            p.add_mesh(pv.PolyData(fps_points), color='magenta', opacity=1.0, 
                      point_size=8, render_points_as_spheres=True, label=f'FPS Model Input ({len(fps_points)})')
    
    # 3. Draw Ball Wireframe (centered at first frame)
    sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=p0, theta_resolution=20, phi_resolution=20)
    p.add_mesh(sphere, style='wireframe', color='gray', opacity=0.5, label=f'Ball R={MAP_QUERY_RADIUS}mm')
    
    # 4. Plot full trajectory (Blue, faint)
    if len(positions) > 1:
        traj_line = pv.lines_from_points(positions)
        p.add_mesh(traj_line, color='blue', opacity=0.3, line_width=2, label='Full Trajectory')
    
    # 5. Plot window frame positions (Red, highlighted)
    p.add_mesh(pv.PolyData(window_positions), color='red', point_size=12, 
              render_points_as_spheres=True, label='Window Frames')
    
    # 6. Connect window frames with a line
    if len(window_positions) > 1:
        window_line = pv.lines_from_points(window_positions)
        p.add_mesh(window_line, color='red', line_width=4, label='Window Path')
    
    # 7. Add start/end labels
    p.add_point_labels([window_positions[0]], ["Start (Ball Center)"], point_size=8, 
                       text_color='green', always_visible=True)
    p.add_point_labels([window_positions[-1]], ["End"], point_size=8, 
                       text_color='darkred', always_visible=True)
    
    # 8. Print frame positions and closest centerline points FROM FPS POINTS
    print(f"\n--- Window Frame Positions ({seq_name}) ---")
    print(f"Ball center (frame 0): [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}]")
    print(f"Ball radius: {MAP_QUERY_RADIUS} mm")
    print(f"Points in ball (raw): {len(ball_points) if ball_points is not None else 0}")
    print(f"Points after DBSCAN: {len(connected_points) if 'connected_points' in dir() and connected_points is not None else 0}")
    print(f"Points after FPS: {len(fps_points) if fps_points is not None else 0} (max_points={max_points})")
    print()
    
    if fps_points is not None and len(fps_points) > 0:
        # Use fps_points for nearest neighbor search (what model sees)
        fps_tree = cKDTree(fps_points)
        for i, (frame_idx, pos) in enumerate(zip(frame_indices, window_positions)):
            dist, closest_idx = fps_tree.query(pos)
            closest_pt = fps_points[closest_idx]
            # Check if frame is inside the ball
            dist_to_center = np.linalg.norm(pos - p0)
            inside_ball = "IN" if dist_to_center <= MAP_QUERY_RADIUS else "OUT"
            print(f"  Frame {i:2d} (idx {frame_idx:4d}) [{inside_ball}]: Pos [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}] -> "
                  f"Closest FPS [{closest_pt[0]:8.2f}, {closest_pt[1]:8.2f}, {closest_pt[2]:8.2f}] (dist: {dist:.2f} mm)")
    else:
        for i, (frame_idx, pos) in enumerate(zip(frame_indices, window_positions)):
            print(f"  Frame {i:2d} (idx {frame_idx:4d}): Pos [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}]")
    print("-" * 80)
    
    # Camera focus on window center
    center = np.mean(window_positions, axis=0)
    p.camera.position = (center[0], center[1] - 100, center[2] + 30)
    p.camera.focal_point = center
    p.camera.up = (0, 0, 1)
    
    p.add_legend()
    p.add_axes()
    p.show()


def analyze_window_containment(data_root, frame_skip, window_size, num_samples=100):
    """
    Sample random windows and check how many frames are inside/outside the ball radius.
    """
    seq_dirs = sorted(glob.glob(os.path.join(data_root, "seq_*")))
    
    # Collect all valid windows across all sequences
    all_windows = []
    effective_len = (window_size - 1) * frame_skip + 1
    
    for seq_dir in seq_dirs:
        traj_path = os.path.join(seq_dir, "trajectory.npy")
        if not os.path.exists(traj_path):
            continue
        traj = np.load(traj_path)
        positions = traj[:, :3]
        
        # Find all valid window start positions
        for start_idx in range(0, len(positions) - effective_len + 1, effective_len):
            frame_indices = [start_idx + i * frame_skip for i in range(window_size)]
            all_windows.append((positions, frame_indices, os.path.basename(seq_dir)))
    
    print(f"\n[INFO] Found {len(all_windows)} valid windows across {len(seq_dirs)} sequences")
    
    # Sample random windows
    if num_samples > len(all_windows):
        num_samples = len(all_windows)
    sampled = random.sample(all_windows, num_samples)
    
    # Statistics
    total_frames = 0
    frames_inside = 0
    frames_outside = 0
    windows_with_out = 0
    out_per_window = []
    
    for positions, frame_indices, seq_name in tqdm(sampled, desc="Checking windows"):
        window_positions = positions[frame_indices]
        p0 = window_positions[0]  # Ball center
        
        window_out = 0
        for i, pos in enumerate(window_positions):
            dist_to_center = np.linalg.norm(pos - p0)
            total_frames += 1
            if dist_to_center <= MAP_QUERY_RADIUS:
                frames_inside += 1
            else:
                frames_outside += 1
                window_out += 1
        
        out_per_window.append(window_out)
        if window_out > 0:
            windows_with_out += 1
    
    # Report
    print("\n" + "="*60)
    print(f"WINDOW CONTAINMENT ANALYSIS (radius={MAP_QUERY_RADIUS}mm)")
    print("="*60)
    print(f"Windows sampled:       {num_samples}")
    print(f"Total frames:          {total_frames}")
    print(f"Frames INSIDE ball:    {frames_inside} ({100*frames_inside/total_frames:.1f}%)")
    print(f"Frames OUTSIDE ball:   {frames_outside} ({100*frames_outside/total_frames:.1f}%)")
    print("-"*60)
    print(f"Windows with any OUT:  {windows_with_out} ({100*windows_with_out/num_samples:.1f}%)")
    out_arr = np.array(out_per_window)
    print(f"Avg OUT per window:    {np.mean(out_arr):.2f}")
    print(f"Max OUT per window:    {np.max(out_arr)}")
    print("-"*60)
    print("Distribution of OUT frames per window:")
    for i in range(window_size + 1):
        count = np.sum(out_arr == i)
        if count > 0:
            print(f"  {i} OUT: {count} windows ({100*count/num_samples:.1f}%)")
    print("="*60)


def analyze_dataset(data_root, frame_skip=1, window_size=None, lung_path=None, centerline_path=None, visualize=False, random_window=False):
    seq_dirs = sorted(glob.glob(os.path.join(data_root, "seq_*")))
    print(f"Found {len(seq_dirs)} sequences in {data_root}")

    all_frame_distances = []
    
    # Validation: Check if we can extract a window
    window_saved = False
    if window_size is not None:
        save_dir = "./check/window"
        os.makedirs(save_dir, exist_ok=True)
        # Clear existing
        for f in glob.glob(os.path.join(save_dir, "*.png")):
            os.remove(f)
        print(f"[INFO] Window frames will be saved to {save_dir}")

    print("\n--- Individual Sequence Stats (First 5) ---")
    for i, seq_dir in enumerate(seq_dirs):
        traj_path = os.path.join(seq_dir, "trajectory.npy")
        vid_path = os.path.join(seq_dir, "video.npy")
        
        if not os.path.exists(traj_path):
            continue
            
        # Load trajectory
        traj = np.load(traj_path) # (T, 7)
        positions = traj[:, :3]   # (T, 3)
        
        # Calculate distances between frames separated by frame_skip
        # d[i] = dist(p[i], p[i+frame_skip])
        if len(positions) > frame_skip:
            diffs = positions[frame_skip:] - positions[:-frame_skip]
            dists = np.linalg.norm(diffs, axis=1)
            all_frame_distances.append(dists)
        else:
            dists = np.array([])
        
        # --- Collect valid windows ---
        if window_size is not None and os.path.exists(vid_path):
            effective_len = (window_size - 1) * frame_skip + 1
            if len(positions) >= effective_len:
                if 'valid_sequences' not in dir():
                    valid_sequences = []
                # Collect all valid starting positions in this sequence
                max_start = len(positions) - effective_len
                valid_sequences.append((seq_dir, vid_path, positions, effective_len, max_start))

        if i < 5:
            print(f"Seq: {os.path.basename(seq_dir)}")
            print(f"  Frames: {len(positions)}")
            if len(dists) > 0:
                print(f"  Mean Move (skip={frame_skip}): {np.mean(dists):.4f} mm")
                print(f"  Max Move  (skip={frame_skip}): {np.max(dists):.4f} mm")
    
    # --- Process selected window ---
    if window_size is not None and 'valid_sequences' in dir() and len(valid_sequences) > 0:
        import cv2
        
        if random_window:
            # Pick a random sequence and random start position
            seq_dir, vid_path, positions, effective_len, max_start = random.choice(valid_sequences)
            start_idx = random.randint(0, max_start)
            print(f"[INFO] Randomly selected sequence: {os.path.basename(seq_dir)}, start_idx={start_idx}")
        else:
            # Use first valid sequence, start at 0
            seq_dir, vid_path, positions, effective_len, max_start = valid_sequences[0]
            start_idx = 0
            print(f"[INFO] Using first valid sequence: {os.path.basename(seq_dir)}")
        
        # Load video and extract frames
        vid_data = np.load(vid_path, mmap_mode='r')
        frame_indices = [start_idx + j * frame_skip for j in range(window_size)]
        window_frames = vid_data[frame_indices]
        
        print(f"[INFO] Saving {len(window_frames)} frames...")
        
        # Setup Video Writer
        save_dir = "./check/window"
        os.makedirs(save_dir, exist_ok=True)
        h, w = window_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid_path = os.path.join(save_dir, "window_clip.mp4")
        out_vid = cv2.VideoWriter(out_vid_path, fourcc, 5.0, (w, h))
        
        for idx, frame in enumerate(window_frames):
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"frame_{idx:04d}.png"), frame)
            out_vid.write(frame)
        
        out_vid.release()
        print(f"[INFO] Saved frames and video to {save_dir}")
        
        # 3D Visualization
        if visualize and lung_path and centerline_path:
            visualize_window_3d(positions, frame_indices, lung_path, centerline_path, os.path.basename(seq_dir))
            
    if not all_frame_distances:
        print("No trajectory data found or sequences too short for frame_skip.")
        return

    # Aggregate
    all_dists = np.concatenate(all_frame_distances)
    
    print("\n" + "="*40)
    print(f"GLOBAL STATISTICS (Movement per sample, skip={frame_skip})")
    print("="*40)
    print(f"Total Samples Analyzed: {len(all_dists)}")
    print(f"Mean Movement:       {np.mean(all_dists):.4f} mm")
    print(f"Median Movement:     {np.median(all_dists):.4f} mm")
    print(f"Std Dev:             {np.std(all_dists):.4f} mm")
    print("-" * 40)
    print(f"Min Movement:        {np.min(all_dists):.4f} mm")
    print(f"Max Movement:        {np.max(all_dists):.4f} mm")
    print("-" * 40)
    print("Percentiles:")
    print(f"  90th: {np.percentile(all_dists, 90):.4f} mm")
    print(f"  95th: {np.percentile(all_dists, 95):.4f} mm")
    print(f"  99th: {np.percentile(all_dists, 99):.4f} mm")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/sequences')
    parser.add_argument('--frame_skip', type=int, default=40, help="Frame skipping interval")
    parser.add_argument('--window_size', type=int, default=10, help="Number of frames in one sample window")
    parser.add_argument('--lung_path', type=str, default='../patient/lungs.obj', help="Path to lungs mesh")
    parser.add_argument('--centerline_path', type=str, default='../dataset/static/centerline.npz', help="Path to centerline centerline")
    parser.add_argument('--visualize', action='store_true', default=True, help="Show 3D visualization of window")
    parser.add_argument('--sample_windows', type=int, default=0, 
                        help="If >0, sample N random windows and report containment stats")
    parser.add_argument('--random_window', action='store_true', 
                        help="Pick a random window for visualization instead of the first one")
    args = parser.parse_args()
    
    # Resolve paths - BASE_DIR is now the parent of check folder (BronchoLoc root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Handle relative paths - if relative, resolve from BASE_DIR (root)
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(BASE_DIR, args.data_root.lstrip('../').lstrip('./'))
    lung_path = os.path.join(BASE_DIR, args.lung_path.lstrip('../').lstrip('./')) if not os.path.isabs(args.lung_path) else args.lung_path
    centerline_path = os.path.join(BASE_DIR, args.centerline_path.lstrip('../').lstrip('./')) if not os.path.isabs(args.centerline_path) else args.centerline_path
    
    analyze_dataset(args.data_root, args.frame_skip, args.window_size, lung_path, centerline_path, args.visualize, args.random_window)
    
    # Run window containment analysis if requested
    if args.sample_windows > 0:
        analyze_window_containment(args.data_root, args.frame_skip, args.window_size, args.sample_windows)
    
    # Save window config for other scripts to use
    import json
    from constants import WINDOW_CONFIG_PATH
    config = {
        'window_size': args.window_size,
        'frame_skip': args.frame_skip
    }
    with open(WINDOW_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[INFO] Saved window config to {WINDOW_CONFIG_PATH}")
    print(f"       window_size={args.window_size}, frame_skip={args.frame_skip}")
