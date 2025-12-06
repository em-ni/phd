import os
import glob
import argparse
import torch
import numpy as np
import cv2
import pyvista as pv
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

# Force Headless Mode for WSL/Server
# PyVista requires this when running without a display.
pv.OFF_SCREEN = True
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1" 

from deep_lung_st import ActionPredictor
from constants import MAP_QUERY_RADIUS, NORM_MAP_SCALE

# Heuristic threshold for connectivity (matches visualize_ball.py)
# Used for DBSCAN clustering to filter connected components.
CONNECTIVITY_THRESHOLD = 2.0 

def filter_connected_component(center_point, neighbors):
    """
    Filters neighbors to keep only those in the same cluster as the center_point.
    Uses DBSCAN. Matches the logic in dataset generation to ensure consistency.
    """
    if len(neighbors) == 0:
        return np.array([])
        
    # 1. Run DBSCAN on all points (neighbors)
    clustering = DBSCAN(eps=CONNECTIVITY_THRESHOLD, min_samples=1).fit(neighbors)
    labels = clustering.labels_
    
    # 2. Find label of the center point (or closest point to it)
    dists = np.linalg.norm(neighbors - center_point, axis=1)
    center_idx = np.argmin(dists)
    center_label = labels[center_idx]
    
    # 3. Select points with the same label
    mask = (labels == center_label)
    connected_indices = np.where(mask)[0]
    
    return neighbors[connected_indices]

def get_local_map(map_tree, map_points, current_pos, current_quat, K=32):
    """
    Finds nearest map points and transforms them to local camera frame during INFERENCE.
    Matches visualize_ball.py logic: Radius Search -> Filter Connected -> Top K for Model.
    
    Args:
        current_pos: Global position (3,).
        current_quat: Global orientation (4,) [x, y, z, w].
        K: Number of points to sample.
        
    Returns: 
        local_tensor: (K, 3) normalized tensor ready for model input.
        candidates_global: (M, 3) all connected points in radius (for visualization).
    """
    # 1. Query All Neighbors in Radius (Global)
    if map_tree is None:
        return torch.zeros(K, 3), np.zeros((0, 3))

    # visualize_ball.py uses query_ball_point which returns all indices within radius.
    indices = map_tree.query_ball_point(current_pos, r=MAP_QUERY_RADIUS)
    
    if len(indices) == 0:
        return torch.zeros(K, 3), np.zeros((0, 3))
        
    raw_neighbors = map_points[indices] # (M, 3)
    
    # 2. Filter Connected Component (DBSCAN)
    # This ensures we only look at the relevant branch and ignore adjacent disjoint airways.
    connected_neighbors = filter_connected_component(current_pos, raw_neighbors)
    
    if len(connected_neighbors) == 0:
        return torch.zeros(K, 3), np.zeros((0, 3))

    # 3. Transform to Local Frame
    rot_cam = R.from_quat(current_quat)
    inv_rot_cam = rot_cam.inv()
    
    # Vector from camera to point
    rel_vecs = connected_neighbors - current_pos
    
    # Rotate
    local_points_all = inv_rot_cam.apply(rel_vecs) # (M, 3)
    
    # 4. Select Top K for Model (Sorted by distance)
    # We want the K closest points to the camera to feed into the model.
    dists = np.linalg.norm(local_points_all, axis=1)
    sorted_idx = np.argsort(dists)
    
    local_points_sorted = local_points_all[sorted_idx]
    
    # Pad or Truncate to fixed size K
    out_points = np.zeros((K, 3), dtype=np.float32)
    M = min(len(local_points_sorted), K)
    out_points[:M] = local_points_sorted[:M]
    
    # Normalize to [-1, 1]
    out_points_norm = out_points / NORM_MAP_SCALE
    
    # Return normalized tensor for model, AND full connected set for viz
    return torch.from_numpy(out_points_norm).float(), connected_neighbors

def run_inference_and_viz(args):
    """
    Runs inference on test sequences and generates visualization videos.
    This simulates a closed-loop control scenario where the model's predictions
    update the camera's position estimate frame-by-frame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Testing on {device}")
    
    # --- 1. Load Resources ---
    static_dir = os.path.join(args.data_root, "static")
    
    # Lungs Mesh (for visualization context)
    lung_obj_path = os.path.join("patient", "lungs.obj")
    if not os.path.exists(lung_obj_path):
        lung_obj_path = os.path.join(static_dir, "lungs.obj")
    
    lung_mesh = None
    if os.path.exists(lung_obj_path):
        print(f"[INFO] Loading 3D Lung Mesh from {lung_obj_path}")
        lung_mesh = pv.read(lung_obj_path)

    # Graph (Centerlines)
    # This is the "Map" we are localizing against.
    graph_path = os.path.join(static_dir, "deep_lung_graph.npz")
    map_tree = None
    map_points = None
    
    if os.path.exists(graph_path):
        print(f"[INFO] Loading Graph from {graph_path}")
        gdata = np.load(graph_path)
        if 'centerline_points' in gdata:
            map_points = gdata['centerline_points']
        else:
            map_points = gdata['node_pos']
        map_tree = cKDTree(map_points)
    else:
        print("[ERROR] Graph file not found!")
        return

    # Initialize Model
    model = ActionPredictor(
        t_frames=args.t_frames, 
        mode=args.model_mode,
        img_size=args.img_size
    ).to(device)
    
    # Load Checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --- 2. Select Sequences ---
    if args.overfit:
        print("[INFO] Overfitting mode: Using 'seq_test' from TRAINING set.")
        sequences_dir = os.path.join(args.data_root, "sequences")
        seq_test_path = os.path.join(sequences_dir, "seq_test")
        if not os.path.exists(seq_test_path):
             print(f"[ERROR] 'seq_test' not found at {seq_test_path}")
             return
        seq_folders = [seq_test_path]
    else:
        if args.split == 'test':
             sequences_dir = os.path.join(args.data_root, "test")
             if not os.path.exists(sequences_dir):
                 sequences_dir = os.path.join(args.data_root, "sequences")
        else:
             sequences_dir = os.path.join(args.data_root, "sequences")
        seq_folders = sorted(glob.glob(os.path.join(sequences_dir, "seq_*")))

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 3. Processing Loop ---
    for seq_idx, seq_dir in enumerate(seq_folders):
        if seq_idx >= args.num_viz: break 
        print(f"[INFO] Processing: {os.path.basename(seq_dir)}")
        
        try:
            # Load Data (Video + Trajectory)
            # Use mmap_mode to save memory
            full_video_np = np.load(os.path.join(seq_dir, "video.npy"), mmap_mode='r')
            full_traj_np = np.load(os.path.join(seq_dir, "trajectory.npy"))
            
            # Apply Stride (Downsample)
            if args.stride > 1:
                full_video_np = full_video_np[::args.stride]
                full_traj_np = full_traj_np[::args.stride]
        except:
            print(f"[WARN] Could not load data for {seq_dir}")
            continue

        # Pre-process video frames for the model
        print("  > Pre-processing video for model...")
        resized_frames = []
        
        # Limit frames if requested for quick debugging
        limit_frames = len(full_video_np)
        if args.max_frames:
            limit_frames = min(limit_frames, args.max_frames + args.t_frames)

        for i in range(limit_frames):
            f = full_video_np[i]
            if f.shape[0] == 3: # Convert CHW -> HWC if needed
                f = np.transpose(f, (1, 2, 0))
            
            # Resize image to model input size (e.g., 128x128)
            f_model = cv2.resize(f, (args.img_size, args.img_size))
            f_model = np.transpose(f_model, (2, 0, 1)) # Back to CHW
            resized_frames.append(f_model)
            
        video_tensor = torch.from_numpy(np.array(resized_frames)).float()
        # Normalize to [-1, 1]
        video_tensor = (video_tensor / 127.5) - 1.0
        
        gt_pos_raw = full_traj_np[:, :3]
        gt_quat_raw = full_traj_np[:, 3:]
        
        N_frames = video_tensor.shape[0]
        T = args.t_frames
        
        # --- CLOSED LOOP INFERENCE ---
        # We start at the known Ground Truth position at Frame 0.
        # From then on, we update position using Model Predictions.
        current_pos_est = gt_pos_raw[0].copy()
        
        # Store data for visualization later
        viz_data = []
        
        print("  > Running Inference...")
        
        # Iterate through the sequence in windows of size T.
        # Stride = T (Non-overlapping windows for inference update, though video is continuous)
        for t in range(0, N_frames - T + 1, T):
            if args.max_frames and t >= args.max_frames:
                print(f"  > Reached max_frames ({args.max_frames}). Stopping.")
                break
            
            # 1. Map Query (at current estimated position)
            # Use the actual rotation from GT? Usually in SLAM we estimate this too, 
            # but here we might assume IMU gives orientation. For simplicity, use GT quat.
            q_curr = gt_quat_raw[t] 
            local_map_tensor, candidates_global = get_local_map(map_tree, map_points, current_pos_est, q_curr, K=32)
            
            # 2. Prepare Batch
            # Video input for this window
            batch_video = video_tensor[t : t+T].unsqueeze(0).to(device) # (1, T, C, H, W)
            
            # Replicate the static local map for all T frames in the window
            # (Model expects map points per frame, but they are constant in this window reference frame)
            batch_map = local_map_tensor.unsqueeze(0).unsqueeze(0).repeat(1, T, 1, 1).to(device)
            
            # 3. Model Prediction
            with torch.no_grad():
                pred_trans = model(batch_video, map_points=batch_map)
            
            pred_trans_np = pred_trans[0].cpu().numpy() # (T, 3)
            
            # 4. Process Predictions
            # Convert model outputs (local deltas) back to global positions
            window_preds_global = []
            
            for i in range(T):
                frame_idx = t + i
                if frame_idx >= N_frames: break
                
                # We un-rotate using the START OF WINDOW rotation (q_curr)
                # Because predictions are made relative to the Local Frame at T=0
                rot_mat_start = R.from_quat(q_curr)
                
                # Denormalize map scale
                delta_local_norm = pred_trans_np[i]
                delta_local_mm = delta_local_norm * NORM_MAP_SCALE
                
                # Local -> Global
                delta_global = rot_mat_start.apply(delta_local_mm)
                
                # New global position
                pred_pos_global = current_pos_est + delta_global
                window_preds_global.append(pred_pos_global)
                
                viz_data.append({
                    'frame_idx': frame_idx,
                    'gt_pos': gt_pos_raw[frame_idx],
                    'pred_pos': pred_pos_global,
                    'ball_center': current_pos_est,
                    'candidates': candidates_global
                })
            
            # 5. Update State
            # The last predicted position becomes the start for the next window
            current_pos_est = window_preds_global[-1]
            
        # --- VISUALIZATION GENERATION ---
        save_path = os.path.join(args.output_dir, f"{os.path.basename(seq_dir)}_viz.mp4")
        print(f"  > Generating Video: {save_path}")
        
        # Setup PyVista Plotter (Headless)
        pv.set_plot_theme("document")
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        
        # Configure Video Writer
        out_h = 600
        
        # Get aspect ratio from first frame
        f_sample = full_video_np[0]
        if f_sample.shape[0] == 3: f_sample = np.transpose(f_sample, (1, 2, 0))
        in_h, in_w = f_sample.shape[:2]
        aspect_ratio = in_w / in_h
        out_w_video = int(out_h * aspect_ratio)
        
        out_w_3d = 800
        total_w = out_w_video + out_w_3d
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 10.0, (total_w, out_h))
        
        for i, data in enumerate(tqdm(viz_data, desc="Rendering")):
            # 1. Input Video Frame (Read from mmap)
            frame_idx = data['frame_idx']
            frame_img = full_video_np[frame_idx]
            
            if frame_img.shape[0] == 3:
                frame_img = np.transpose(frame_img, (1, 2, 0))
                
            # Ensure uint8
            if frame_img.dtype != np.uint8:
                if frame_img.max() <= 1.0:
                    frame_img = (frame_img * 255).astype(np.uint8)
                else:
                    frame_img = frame_img.astype(np.uint8)
            
            # Resize for output video
            frame_img = cv2.resize(frame_img, (out_w_video, out_h), interpolation=cv2.INTER_NEAREST)
            
            # Check for black frame warnings
            if frame_img.mean() < 1.0:
                print(f"[WARN] Frame {frame_idx} is black! Mean: {frame_img.mean()}")

            # 2. 3D Render
            plotter.clear()
            
            # Add Lungs (Context)
            if lung_mesh:
                plotter.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
            
            # Add Full Centerline (Faint context)
            if map_points is not None:
                plotter.add_mesh(pv.PolyData(map_points), color='black', opacity=0.1, point_size=2, render_points_as_spheres=True)
            
            # Visualize the Query Ball (Wireframe)
            ball_center = data['ball_center']
            sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=ball_center, theta_resolution=20, phi_resolution=20)
            plotter.add_mesh(sphere, style='wireframe', color='gray', opacity=0.3, line_width=1)
            
            # Visualize Candidates (Red points used by model)
            candidates = data['candidates']
            if len(candidates) > 0:
                plotter.add_mesh(pv.PolyData(candidates), color='red', point_size=10, render_points_as_spheres=True)
            else:
                if i % args.t_frames == 0:
                    print(f"[WARN] No candidates for frame {data['frame_idx']}")

            # Visualize Current Prediction (Green Sphere)
            pred_pos = data['pred_pos']
            plotter.add_mesh(pv.Sphere(radius=2.5, center=pred_pos), color='green', label='Prediction')
            
            # Visualize Ground Truth (Blue Sphere - for comparison)
            gt_pos = data['gt_pos']
            plotter.add_mesh(pv.Sphere(radius=1.5, center=gt_pos), color='blue', opacity=0.5, label='GT')

            # Camera Follow Mode
            # Position the 3D camera to follow the PREDICTED position, simulating a 3rd person view of the scope.
            cam_target = pred_pos
            plotter.camera.position = (cam_target[0], cam_target[1] - 120, cam_target[2] + 30)
            plotter.camera.focal_point = cam_target
            plotter.camera.up = (0, 0, 1)
            plotter.camera.zoom(1.0)
            
            # Render to image
            img_3d = plotter.screenshot(return_img=True, transparent_background=False)
            img_3d = cv2.resize(img_3d, (out_w_3d, out_h))
            img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
            
            # Combine Side-by-Side
            combined = np.hstack([frame_img, img_3d])
            out.write(combined)
            
        plotter.close()
        out.release()
        
        # Save Snapshot of Trajectory (Global View)
        # Collect all preds to calculate error metrics
        all_preds = np.array([d['pred_pos'] for d in viz_data])
        all_gt = np.array([d['gt_pos'] for d in viz_data])
        
        # Calculate Average Distance Error (ADE)
        ade = np.linalg.norm(all_preds - all_gt, axis=1).mean()
        print(f"  > Sequence ADE: {ade:.2f} mm")
        
        # Create a static snapshot image
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 1000))
        if lung_mesh:
            plotter.add_mesh(lung_mesh, color='wheat', opacity=0.1)
        
        plotter.add_mesh(pv.PolyData(all_gt), color='blue', point_size=3, render_points_as_spheres=True, label='GT')
        plotter.add_mesh(pv.PolyData(all_preds), color='green', point_size=4, render_points_as_spheres=True, label='Pred')
        
        # Draw trajectory lines
        if len(all_preds) > 1:
            lines = pv.lines_from_points(all_preds)
            plotter.add_mesh(lines, color='green', line_width=2)
            
        center = np.mean(all_gt, axis=0)
        plotter.camera.position = (center[0], center[1] - 200, center[2] + 50)
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        
        snap_path = os.path.join(args.output_dir, f"{os.path.basename(seq_dir)}_ADE_{ade:.1f}mm.png")
        plotter.screenshot(snap_path)
        plotter.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./dataset/test/results')
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'b', 'm', 'l'])
    parser.add_argument('--t_frames', type=int, default=16)
    parser.add_argument('--stride', type=int, default=1, help="Stride for data sampling (match training)")
    parser.add_argument('--img_size', type=int, default=128, help="Image resolution (default: 128)")
    parser.add_argument('--num_viz', type=int, default=1)
    parser.add_argument('--overfit', action='store_true', help="Overfit on a small subset")
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train'])
    parser.add_argument('--max_frames', type=int, default=None, help="Limit number of frames for debugging")
    
    args = parser.parse_args()
    run_inference_and_viz(args)
