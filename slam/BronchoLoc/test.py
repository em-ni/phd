import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

from deep_lung_st import ActionPredictor
from deep_lung_dataset import DeepLungDataset
from constants import NORM_MAP_SCALE, MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS
from utils import load_centerline_points, filter_connected_component, farthest_point_sample


def render_3d_view(plotter, lung_mesh, centerline_pts, centerline_tree,
                   gt_traj_current, pred_traj_current,
                   current_gt, current_pred, frame_idx, total_frames):
    """
    Renders the 3D trajectory view to a numpy array.
    Style matches check_ball.py and check_traj.py.
    Shows FPS-downsampled points that the model sees.
    
    Args:
        plotter: PyVista Plotter (off-screen)
        lung_mesh: Loaded lung mesh (pyvista object) or None
        centerline_pts: Centerline points array (N, 3) or None
        centerline_tree: KDTree for centerline points or None
        gt_traj_current: GT trajectory up to current frame (M, 3)
        pred_traj_current: Pred trajectory up to current frame (M, 3)
        current_gt: Current frame GT position (3,)
        current_pred: Current frame Pred position (3,)
        frame_idx: Current frame index in window
        total_frames: Total frames in window
        
    Returns:
        np.ndarray: Rendered image (H, W, 3) as uint8
    """
    plotter.clear()
    
    # 1. Draw Lung Mesh (Ghostly - like check_ball.py)
    if lung_mesh is not None:
        plotter.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
    
    # 2. Draw Centerline (Faint black points - like check_ball.py)
    if centerline_pts is not None:
        plotter.add_mesh(pv.PolyData(centerline_pts), color='black', opacity=0.2, 
                        point_size=3, render_points_as_spheres=True, label='Centerline')
    
    # 3. Compute FPS points (what model sees) centered at first GT position
    fps_points = None
    if centerline_tree is not None and len(gt_traj_current) > 0:
        p0 = gt_traj_current[0]  # Ball is centered at START of window
        ball_indices = centerline_tree.query_ball_point(p0, r=MAP_QUERY_RADIUS)
        if len(ball_indices) > 0:
            ball_points = centerline_pts[ball_indices]
            connected_points, _ = filter_connected_component(p0, ball_points)
            if len(connected_points) > DEFAULT_MAX_MAP_POINTS:
                dists = np.linalg.norm(connected_points - p0, axis=1)
                start_idx = np.argmin(dists)
                fps_points, _ = farthest_point_sample(connected_points, DEFAULT_MAX_MAP_POINTS, start_idx=start_idx)
            else:
                fps_points = connected_points
    
    # 4. Draw Ball Query Sphere (Wireframe - centered at START, like dataset)
    if len(gt_traj_current) > 0:
        p0 = gt_traj_current[0]
        ball_sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=p0, 
                                theta_resolution=20, phi_resolution=20)
        plotter.add_mesh(ball_sphere, style='wireframe', color='gray', opacity=0.5, 
                        label=f'Ball R={MAP_QUERY_RADIUS}mm')
    
    # 5. Draw FPS Points (magenta - what model sees)
    if fps_points is not None and len(fps_points) > 0:
        plotter.add_mesh(pv.PolyData(fps_points), color='magenta', opacity=0.8, 
                        point_size=6, render_points_as_spheres=True, 
                        label=f'FPS Input ({len(fps_points)})')
    
    # 6. Draw GT Trajectory (Blue - building up as frames proceed)
    if len(gt_traj_current) > 1:
        gt_line = pv.lines_from_points(gt_traj_current)
        plotter.add_mesh(gt_line, color='blue', line_width=4, label='GT Trajectory')
    
    # 7. Draw Predicted Trajectory (Red - building up as frames proceed)  
    if len(pred_traj_current) > 1:
        pred_line = pv.lines_from_points(pred_traj_current)
        plotter.add_mesh(pred_line, color='red', line_width=4, label='Pred Trajectory')
    
    # 8. Draw current position markers (spheres)
    plotter.add_mesh(pv.Sphere(radius=1.5, center=current_gt), color='blue')
    plotter.add_mesh(pv.Sphere(radius=1.5, center=current_pred), color='red')
    
    # 9. Add Start label at first GT point
    if len(gt_traj_current) > 0:
        plotter.add_point_labels([gt_traj_current[0]], ["Start"], point_size=8, 
                                 text_color='green', always_visible=True, font_size=12)
    
    # 10. Add text overlay (black text for white background)
    text = f"Frame: {frame_idx+1}/{total_frames}\n"
    text += f"GT (mm):   X={current_gt[0]:+.2f} Y={current_gt[1]:+.2f} Z={current_gt[2]:+.2f}\n"
    text += f"Pred (mm): X={current_pred[0]:+.2f} Y={current_pred[1]:+.2f} Z={current_pred[2]:+.2f}"
    plotter.add_text(text, position='upper_left', font_size=10, color='black')
    
    # 11. Set camera to good viewpoint (like check_ball.py)
    # Focus on the current GT position
    plotter.camera.position = (current_gt[0], current_gt[1] - 150, current_gt[2] + 30)
    plotter.camera.focal_point = current_gt
    plotter.camera.up = (0, 0, 1)
    
    # Add legend and axes
    plotter.add_legend()
    plotter.add_axes()
    
    # Render to numpy array
    plotter.render()
    img = plotter.screenshot(return_img=True)
    
    return img


def test(args):
    """
    Main test function.
    Runs inference on the dataset and generates a side-by-side video visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Test on {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset (window_size/frame_skip loaded from config inside dataset)
    full_dataset = DeepLungDataset(
        data_root=os.path.join(args.data_root, "sequences"),
        mode='test',
        img_size=args.img_size
    )
    
    # Get window_size from dataset (loaded from config)
    window_size = full_dataset.window_size
    print(f"[INFO] Using window_size={window_size}, frame_skip={full_dataset.frame_skip}")
    
    # --- DEBUGGING / OVERFITTING MODES (same as train.py) ---
    if args.overfit:
        print("[INFO] Overfitting mode: Testing on 'seq_test' only.")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if "seq_test" in vp]
        
        if not indices:
            print("[ERROR] 'seq_test' not found in dataset!")
            return
            
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        test_ds = full_dataset
    else:
        test_ds = full_dataset
        
    # For debug_one, use same first batch as training (shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    if args.debug_one:
        print("[INFO] DEBUG ONE mode: Testing on a SINGLE batch (first 16 samples).")
        first_batch = next(iter(test_loader))
        
        class SingleBatchLoader:
            def __init__(self, batch): 
                self.batch = batch
            def __iter__(self): 
                yield self.batch
            def __len__(self): 
                return 1
                
        test_loader = SingleBatchLoader(first_batch)
    
    # Load Model
    model = ActionPredictor(
        window_size=window_size,
        mode=args.model_mode,
        img_size=args.img_size
    ).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Default fallback names (for backwards compatibility)
        if args.debug_one:
            ckpt_path = os.path.join(args.checkpoint_dir, "debug_one_model.pth")
        elif args.overfit:
            ckpt_path = os.path.join(args.checkpoint_dir, "overfit_model.pth")
        else:
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return
        
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle both old (state_dict only) and new (full checkpoint) formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        # Old format: checkpoint IS the state dict
        model.load_state_dict(checkpoint)
    model.eval()
    
    # --- LOAD 3D ASSETS (like check_traj.py) ---
    # Load lung mesh
    lung_mesh = None
    lung_path = os.path.join(args.data_root, "..", "patient", "lungs.obj")
    if os.path.exists(lung_path):
        print(f"[INFO] Loading lung mesh: {lung_path}")
        lung_mesh = pv.read(lung_path)
    else:
        print(f"[WARNING] Lung mesh not found at {lung_path}")
    
    # Load centerline
    graph_path = os.path.join(args.data_root, "static", "deep_lung_graph.npz")
    centerline_pts = load_centerline_points(graph_path)
    centerline_tree = None
    if centerline_pts is not None:
        print(f"[INFO] Loaded {len(centerline_pts)} centerline points")
        centerline_tree = cKDTree(centerline_pts)
    
    # --- INFERENCE AND VIDEO GENERATION ---
    print("[INFO] Running inference and generating video...")
    
    # Initialize PyVista off-screen plotter for 3D rendering
    pv.start_xvfb()  # For headless environments
    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    plotter.set_background('white')  # White background like check_ball.py
    
    # Collect all frames
    all_frames = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            video = batch['video'].to(device)  # (B, T, C, H, W)
            gt_deltas = batch['actions'].to(device)  # (B, T, 6) - LOCAL frame
            map_points = batch['map_points'].to(device)  # (B, T, K, 3)
            map_mask = batch['map_mask'].to(device)  # (B, T, K)
            
            # Get first frame poses for transforming back to global
            first_pos = batch['first_frame_pos'].numpy()  # (B, 3)
            first_quat = batch['first_frame_quat'].numpy()  # (B, 4)
            
            # Forward pass
            pred_trans = model(video, map_points=map_points, map_mask=map_mask)  # (B, T, 3) - normalized output
            
            # Get GT translation (first 3 dims of actions) - LOCAL frame
            gt_trans_local = gt_deltas[:, :, :3].cpu().numpy()  # (B, T, 3)
            
            # Denormalize prediction to mm (model outputs normalized values)
            pred_trans_local = pred_trans.cpu().numpy() * NORM_MAP_SCALE  # (B, T, 3)
            
            # Convert video back to displayable format
            # Dataset stores RGB, normalized to [-1, 1]
            video_np = video.cpu().numpy()  # (B, T, C, H, W)
            video_np = (video_np + 1.0) * 127.5  # Denormalize from [-1, 1] to [0, 255]
            video_np = video_np.clip(0, 255).astype(np.uint8)
            
            # Process each sample in batch
            B, T = video_np.shape[:2]
            for b in range(B):
                # Get first frame's global pose for this sample
                p0 = first_pos[b]  # (3,)
                q0 = first_quat[b]  # (4,)
                rot_0 = R.from_quat(q0)  # Rotation from frame 0's local to global
                
                # Transform LOCAL to GLOBAL coordinates
                # global = rot_0.apply(local) + p0
                gt_window_global = rot_0.apply(gt_trans_local[b]) + p0  # (T, 3)
                pred_window_global = rot_0.apply(pred_trans_local[b]) + p0  # (T, 3)
                
                # Also keep local values for printing
                gt_window_local = gt_trans_local[b]  # (T, 3)
                pred_window_local = pred_trans_local[b]  # (T, 3)
                
                # Process each frame in window
                for t in range(T):
                    # Get input frame (C, H, W) -> (H, W, C)
                    # Note: Different sequences may have different color orders
                    # Simulator sequences are BGR, phantom sequences are RGB
                    # We'll convert assuming BGR (most common for OpenCV-saved data)
                    frame_bgr = video_np[b, t].transpose(1, 2, 0)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Current positions (LOCAL for printing, GLOBAL for 3D viz)
                    current_gt_local = gt_window_local[t]
                    current_pred_local = pred_window_local[t]
                    current_gt_global = gt_window_global[t]
                    current_pred_global = pred_window_global[t]
                    
                    # Print frame info to console (local frame values)
                    print(f"[Batch {batch_idx+1} | Sample {b+1} | Frame {t+1}/{T}] "
                          f"GT: ({current_gt_local[0]:+.3f}, {current_gt_local[1]:+.3f}, {current_gt_local[2]:+.3f}) mm | "
                          f"Pred: ({current_pred_local[0]:+.3f}, {current_pred_local[1]:+.3f}, {current_pred_local[2]:+.3f}) mm")
                    
                    # Trajectory up to current frame in window (GLOBAL for 3D)
                    gt_traj_current = gt_window_global[:t+1]
                    pred_traj_current = pred_window_global[:t+1]
                    
                    # Render 3D view (with lung mesh and centerline + FPS points)
                    img_3d = render_3d_view(
                        plotter,
                        lung_mesh,
                        centerline_pts,
                        centerline_tree,
                        gt_traj_current,
                        pred_traj_current,
                        current_gt_global,
                        current_pred_global,
                        t,
                        T
                    )
                    
                    # Get dimensions
                    h_3d, w_3d = img_3d.shape[:2]
                    h_frame, w_frame = frame_rgb.shape[:2]
                    
                    # Resize video frame to match 3D view height while preserving aspect ratio
                    scale = h_3d / h_frame
                    new_w_frame = int(w_frame * scale)
                    frame_resized = cv2.resize(frame_rgb, (new_w_frame, h_3d))
                    
                    # Create side-by-side frame (RGB)
                    combined_rgb = np.hstack([frame_resized, img_3d])
                    
                    # Convert RGB to BGR for OpenCV video writing
                    combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Add frame info overlay on input side
                    cv2.putText(combined_bgr, f"Window: {batch_idx+1}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined_bgr, f"Frame: {t+1}/{T}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    all_frames.append(combined_bgr)
    
    plotter.close()
    
    # Save video
    if args.debug_one:
        output_path = os.path.join(args.output_dir, "test_debug_one.mp4")
    elif args.overfit:
        output_path = os.path.join(args.output_dir, "test_overfit.mp4")
    else:
        output_path = os.path.join(args.output_dir, "test_results.mp4")
    
    print(f"[INFO] Saving video to {output_path}")
    
    if len(all_frames) > 0:
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, args.fps, (w, h))
        
        for frame in tqdm(all_frames, desc="Writing video"):
            out.write(frame)
        
        out.release()
        print(f"[INFO] Video saved: {output_path}")
        print(f"[INFO] Total frames: {len(all_frames)}")
    else:
        print("[WARNING] No frames to save!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BronchoLoc model and generate visualization video")
    
    # Same arguments as train.py
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to specific checkpoint file")
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'b', 'm', 'l'])
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (1 recommended for video)")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overfit', action='store_true', help="Test on seq_test only (matches overfit training)")
    parser.add_argument('--debug_one', action='store_true', help="Test on SINGLE batch (matches debug_one training)")
    parser.add_argument('--img_size', type=int, default=128, help="Image resolution (default: 128)")
    
    # Test-specific arguments
    parser.add_argument('--output_dir', type=str, default='./dataset/test/results', help="Output directory for videos")
    parser.add_argument('--fps', type=int, default=10, help="Output video FPS")
    
    args = parser.parse_args()
    test(args)
