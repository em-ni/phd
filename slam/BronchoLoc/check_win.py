import numpy as np
import glob
import os
import argparse
import pyvista as pv
from scipy.spatial import cKDTree
from constants import MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS
from utils import load_centerline_points, filter_connected_component, farthest_point_sample


def visualize_window_3d(positions, frame_indices, lung_path, graph_path, seq_name, max_points=DEFAULT_MAX_MAP_POINTS):
    """
    Visualizes window frame positions in 3D context.
    Uses ball query centered at first frame (like deep_lung_dataset.py).
    
    Args:
        positions: (N, 3) array of all trajectory positions
        frame_indices: list of indices for the window frames
        lung_path: path to lungs.obj
        graph_path: path to centerline graph .npz
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
    centerline_pts = load_centerline_points(graph_path)
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
        
        # Apply FPS downsampling (same as dataset)
        if len(connected_points) > max_points:
            dists = np.linalg.norm(connected_points - p0, axis=1)
            start_idx = np.argmin(dists)
            fps_points, _ = farthest_point_sample(connected_points, max_points, start_idx=start_idx)
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

def analyze_dataset(data_root, frame_skip=1, window_size=None, lung_path=None, graph_path=None, visualize=False):
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
        
        # --- Save Window Logic ---
        if window_size is not None and not window_saved and os.path.exists(vid_path):
            effective_len = (window_size - 1) * frame_skip + 1
            if len(positions) >= effective_len:
                print(f"[INFO] Found valid sequence for window extraction: {os.path.basename(seq_dir)}")
                import cv2
                # Load video
                vid_data = np.load(vid_path, mmap_mode='r')
                # Select frames: 0, frame_skip, 2*frame_skip, ... for window_size frames
                frame_indices = [i * frame_skip for i in range(window_size)]
                window_frames = vid_data[frame_indices]
                
                print(f"[INFO] Saving {len(window_frames)} frames...")
                
                # Setup Video Writer
                h, w = window_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_vid_path = os.path.join(save_dir, "window_clip.mp4")
                out_vid = cv2.VideoWriter(out_vid_path, fourcc, 5.0, (w, h))
                
                for idx, frame in enumerate(window_frames):
                    # Frame is likely RGB or BGR. If RGB, convert to BGR for cv2
                    if frame.shape[-1] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    cv2.imwrite(os.path.join(save_dir, f"frame_{idx:04d}.png"), frame)
                    out_vid.write(frame)
                
                out_vid.release()
                print(f"[INFO] Saved frames and video to {save_dir}")
                window_saved = True
                
                # 3D Visualization
                if visualize and lung_path and graph_path:
                    visualize_window_3d(positions, frame_indices, lung_path, graph_path, os.path.basename(seq_dir))

        if i < 5:
            print(f"Seq: {os.path.basename(seq_dir)}")
            print(f"  Frames: {len(positions)}")
            if len(dists) > 0:
                print(f"  Mean Move (skip={frame_skip}): {np.mean(dists):.4f} mm")
                print(f"  Max Move  (skip={frame_skip}): {np.max(dists):.4f} mm")
            
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
    parser.add_argument('--data_root', type=str, default='./dataset/sequences')
    parser.add_argument('--frame_skip', type=int, default=10, help="Frame skipping interval")
    parser.add_argument('--window_size', type=int, default=16, help="Number of frames in one sample window")
    parser.add_argument('--lung_path', type=str, default='./patient/lungs.obj', help="Path to lungs mesh")
    parser.add_argument('--graph_path', type=str, default='./dataset/static/deep_lung_graph.npz', help="Path to centerline graph")
    parser.add_argument('--visualize', action='store_true', default=True, help="Show 3D visualization of window")
    args = parser.parse_args()
    
    # Resolve paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    lung_path = os.path.join(BASE_DIR, args.lung_path) if not os.path.isabs(args.lung_path) else args.lung_path
    graph_path = os.path.join(BASE_DIR, args.graph_path) if not os.path.isabs(args.graph_path) else args.graph_path
    
    analyze_dataset(args.data_root, args.frame_skip, args.window_size, lung_path, graph_path, args.visualize)
    
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
