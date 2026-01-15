"""
Test script for BIRD: Bronchial Intraoperative Route Discriminator

Tests ANT + BIRD together:
1. ANT provides local predictions + visual features
2. BIRD refines predictions using global context (full centerline + neural memory)

Key difference from test_ant.py:
- Uses both ANT and BIRD models
- BIRD maintains memory state across windows for sequential processing
- Shows both ANT-only and BIRD-refined predictions for comparison
"""
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

from ant import ActionPredictor
from bird import create_bird
from ant_dataset import AntDataset
from constants import NORM_MAP_SCALE, MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS, MAP_POINT_SPACING
from utils.utils import load_centerline_points, filter_connected_component, density_based_sample, find_centerline_path, interpolate_trajectory


def downsample_centerline(centerline_pts, min_distance=MAP_POINT_SPACING):
    """Downsample centerline for BIRD cross-attention (same method as training)."""
    sampled, _ = density_based_sample(centerline_pts, min_distance=min_distance, start_idx=0)
    return sampled


def get_map_points_at_position(anchor_pos, centerline_pts, centerline_tree, 
                                max_points=DEFAULT_MAX_MAP_POINTS, 
                                query_radius=MAP_QUERY_RADIUS):
    """
    Query local centerline candidates around an anchor position.
    Used for closed-loop inference where BIRD's output feeds back to ANT.
    
    Args:
        anchor_pos: (3,) position to query around (in mm, NOT normalized)
        centerline_pts: (N, 3) full centerline points
        centerline_tree: cKDTree for efficient queries
        max_points: maximum number of candidates
        query_radius: search radius in mm
        
    Returns:
        map_points: (max_points, 3) normalized candidates
        map_mask: (max_points,) boolean mask for valid points
    """
    # Query points within radius
    indices = centerline_tree.query_ball_point(anchor_pos, query_radius)
    
    if len(indices) == 0:
        # Fallback: find nearest K points
        _, indices = centerline_tree.query(anchor_pos, k=min(10, len(centerline_pts)))
        if isinstance(indices, int):
            indices = [indices]
    
    # Get connected component around anchor
    nearby_pts = centerline_pts[indices]
    filtered_pts = filter_connected_component(nearby_pts, anchor_pos, max_gap=5.0)
    
    # Downsample for uniform spacing
    if len(filtered_pts) > 0:
        candidates, _ = density_based_sample(filtered_pts, min_distance=MAP_POINT_SPACING, start_idx=0)
    else:
        candidates = nearby_pts
    
    # Pad or truncate to max_points
    n_pts = len(candidates)
    map_points = np.zeros((max_points, 3), dtype=np.float32)
    map_mask = np.zeros(max_points, dtype=bool)
    
    if n_pts > max_points:
        candidates = candidates[:max_points]
        n_pts = max_points
    
    map_points[:n_pts] = candidates / NORM_MAP_SCALE  # Normalize
    map_mask[:n_pts] = True
    
    return map_points, map_mask


def render_3d_view_bird(plotter, lung_mesh, centerline_pts, centerline_tree,
                        gt_traj_current, ant_traj_current, bird_traj_current,
                        raw_gt_traj_current, current_gt, current_ant, current_bird,
                        current_raw_gt, frame_idx, total_frames):
    """
    Renders the 3D trajectory view showing ANT vs BIRD predictions.
    
    Colors:
    - Green: Raw GT (actual camera positions)
    - Blue: GT trajectory (centerline-projected)
    - Orange/Yellow: ANT prediction (local only)
    - Red: BIRD prediction (globally refined)
    """
    plotter.clear()
    
    # 1. Draw Lung Mesh
    if lung_mesh is not None:
        plotter.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
    
    # 2. Draw Centerline
    if centerline_pts is not None:
        plotter.add_mesh(pv.PolyData(centerline_pts), color='black', opacity=0.2, 
                        point_size=3, render_points_as_spheres=True, label='Centerline')
    
    # 3. Draw Ball Query Sphere
    if len(gt_traj_current) > 0:
        p0 = gt_traj_current[0]
        ball_sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=p0, 
                                theta_resolution=20, phi_resolution=20)
        plotter.add_mesh(ball_sphere, style='wireframe', color='gray', opacity=0.5,
                        label=f'Ball R={MAP_QUERY_RADIUS}mm')
    
    # 4. Draw Raw GT Trajectory (Green - actual camera positions)
    if len(raw_gt_traj_current) > 1:
        raw_gt_line = pv.lines_from_points(raw_gt_traj_current)
        plotter.add_mesh(raw_gt_line, color='green', line_width=3, label='Raw GT (actual)')
    
    # 5. Draw GT Trajectory (Blue)
    if len(gt_traj_current) > 1:
        gt_line = pv.lines_from_points(gt_traj_current)
        plotter.add_mesh(gt_line, color='blue', line_width=4, label='GT Trajectory')
    
    # 6. Draw ANT Trajectory (Orange - local prediction)
    if len(ant_traj_current) > 1:
        ant_line = pv.lines_from_points(ant_traj_current)
        plotter.add_mesh(ant_line, color='orange', line_width=3, label='ANT (local)')
    
    # 7. Draw BIRD Trajectory (Red - globally refined)
    if len(bird_traj_current) > 1:
        bird_line = pv.lines_from_points(bird_traj_current)
        plotter.add_mesh(bird_line, color='red', line_width=4, label='BIRD (global)')
    
    # 8. Draw current position markers
    plotter.add_mesh(pv.Sphere(radius=1.0, center=current_raw_gt), color='green')  # Raw GT
    plotter.add_mesh(pv.Sphere(radius=1.5, center=current_gt), color='blue')  # GT
    plotter.add_mesh(pv.Sphere(radius=1.5, center=current_ant), color='orange')  # ANT
    plotter.add_mesh(pv.Sphere(radius=2.0, center=current_bird), color='red')  # BIRD
    
    # 9. Add Start label
    if len(gt_traj_current) > 0:
        plotter.add_point_labels([gt_traj_current[0]], ["Start"], point_size=8, 
                                 text_color='green', always_visible=True, font_size=12)
    
    # 10. Add text overlay with error info
    ant_error = np.linalg.norm(current_ant - current_gt)
    bird_error = np.linalg.norm(current_bird - current_gt)
    text = f"Frame: {frame_idx+1}/{total_frames}\n"
    text += f"ANT Error:  {ant_error:.2f}mm\n"
    text += f"BIRD Error: {bird_error:.2f}mm"
    plotter.add_text(text, position='upper_left', font_size=10, color='black')
    
    # 11. Set camera
    plotter.camera.position = (current_gt[0], current_gt[1] - 150, current_gt[2] + 30)
    plotter.camera.focal_point = current_gt
    plotter.camera.up = (0, 0, 1)
    
    plotter.add_legend()
    plotter.add_axes()
    
    plotter.render()
    img = plotter.screenshot(return_img=True)
    
    return img


def load_full_window_frames(vid_path, start_idx, window_size, frame_skip, img_size):
    """Loads ALL frames within a window span for smooth visualization."""
    vid_mmap = np.load(vid_path, mmap_mode='r')
    
    total_frames = (window_size - 1) * frame_skip + 1
    end_idx = start_idx + total_frames
    actual_end = min(end_idx, len(vid_mmap))
    
    all_frames_raw = vid_mmap[start_idx:actual_end]
    
    display_frames = []
    for frame in all_frames_raw:
        frame = np.array(frame)
        
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        elif len(frame.shape) == 3 and frame.shape[-1] == 3:
            pass
        else:
            if len(frame.shape) == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
        
        frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        display_frames.append(frame_rgb)
    
    display_frames = np.array(display_frames)
    keyframe_indices = [i * frame_skip for i in range(window_size) if i * frame_skip < len(display_frames)]
    
    return display_frames, keyframe_indices


def interpolate_positions_for_frames(pred_positions, keyframe_indices, total_frames, 
                                     centerline_pts, centerline_tree):
    """Interpolate positions for ALL frames based on predictions at keyframes."""
    frame_positions = np.zeros((total_frames, 3))
    
    for i, kf_idx in enumerate(keyframe_indices):
        if kf_idx < total_frames:
            frame_positions[kf_idx] = pred_positions[i]
    
    for i in range(len(keyframe_indices) - 1):
        start_kf = keyframe_indices[i]
        end_kf = keyframe_indices[i + 1]
        
        if end_kf >= total_frames:
            end_kf = total_frames - 1
            
        start_pos = pred_positions[i]
        end_pos = pred_positions[i + 1] if i + 1 < len(pred_positions) else pred_positions[i]
        
        if centerline_tree is not None:
            path = find_centerline_path(start_pos, end_pos, centerline_pts, centerline_tree)
        else:
            path = np.array([start_pos, end_pos])
        
        path_len = len(path)
        
        for f_idx in range(start_kf, end_kf + 1):
            t = (f_idx - start_kf) / max(1, end_kf - start_kf)
            path_idx = int(t * (path_len - 1))
            path_idx = min(path_idx, path_len - 1)
            frame_positions[f_idx] = path[path_idx]
    
    return frame_positions


def test(args):
    """Main test function for ANT + BIRD."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting ANT+BIRD Test on {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    full_dataset = AntDataset(
        data_root=os.path.join(args.data_root, "sequences"),
        mode='test',
        img_size=args.img_size,
        chain_mode=True  # Always use chain mode for BIRD (sequential processing)
    )
    
    window_size = full_dataset.window_size
    frame_skip = full_dataset.frame_skip
    print(f"[INFO] Using window_size={window_size}, frame_skip={frame_skip}")
    
    # Sequence filtering
    if args.seq_filter:
        print(f"[INFO] Filtering sequences matching: '{args.seq_filter}'")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if args.seq_filter in vp]
        
        if not indices:
            print(f"[ERROR] No sequences matching '{args.seq_filter}' found!")
            return
        
        print(f"[INFO] Found {len(indices)} windows matching filter.")
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        test_ds = full_dataset
    else:
        test_ds = full_dataset
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # ===========================================================================
    # Load ANT Model
    # ===========================================================================
    if not args.ant_checkpoint:
        print("[ERROR] --ant_checkpoint is required!")
        return
    
    print(f"[INFO] Loading ANT from: {args.ant_checkpoint}")
    ant_checkpoint = torch.load(args.ant_checkpoint, map_location=device)
    
    if 'model_state_dict' in ant_checkpoint:
        ant_state_dict = ant_checkpoint['model_state_dict']
    else:
        ant_state_dict = ant_checkpoint
    
    ant_model = ActionPredictor(
        window_size=window_size,
        mode=args.model_mode,
        img_size=args.img_size
    ).to(device)
    ant_model.load_state_dict(ant_state_dict)
    ant_model.eval()
    
    # ===========================================================================
    # Load BIRD Model
    # ===========================================================================
    if not args.bird_checkpoint:
        print("[ERROR] --bird_checkpoint is required!")
        return
    
    print(f"[INFO] Loading BIRD from: {args.bird_checkpoint}")
    bird_checkpoint = torch.load(args.bird_checkpoint, map_location=device)
    
    if 'bird_state_dict' in bird_checkpoint:
        bird_state_dict = bird_checkpoint['bird_state_dict']
    else:
        bird_state_dict = bird_checkpoint
    
    # Load and downsample centerline for BIRD
    centerline_path = os.path.join(args.data_root, "static", "centerline.npz")
    centerline_pts = load_centerline_points(centerline_path)
    if centerline_pts is None:
        print("[ERROR] Centerline not found!")
        return
    print(f"[INFO] Loaded {len(centerline_pts)} centerline points")
    
    centerline_ds = downsample_centerline(centerline_pts)
    print(f"[INFO] Downsampled to {len(centerline_ds)} points for BIRD")
    
    bird_model = create_bird(
        ant_mode=args.model_mode,
        num_centerline_pts=len(centerline_ds)
    ).to(device)
    bird_model.load_state_dict(bird_state_dict)
    bird_model.eval()
    
    # Pre-encode centerline
    centerline_normalized = torch.tensor(centerline_ds / NORM_MAP_SCALE, dtype=torch.float32).to(device)
    with torch.no_grad():
        centerline_encoded = bird_model.encode_centerline(centerline_normalized)
    
    # Build KDTree for visualization
    centerline_tree = cKDTree(centerline_pts)
    
    # Load lung mesh
    lung_mesh = None
    lung_path = os.path.join(args.data_root, "..", "patient", "lungs.obj")
    if os.path.exists(lung_path):
        print(f"[INFO] Loading lung mesh: {lung_path}")
        lung_mesh = pv.read(lung_path)
    
    # ===========================================================================
    # Inference and Video Generation
    # ===========================================================================
    print("[INFO] Running inference and generating video...")
    
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    plotter.set_background('white')
    
    all_frames = []
    
    # Memory state for BIRD (persists across windows)
    mem_state = None
    prev_seq_path = None
    
    # Chain anchors for open-loop testing
    ant_chain_anchor = None
    bird_chain_anchor = None  # Used for closed-loop feedback
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            video = batch['video'].to(device)
            gt_deltas = batch['actions'].to(device)
            
            first_pos = batch['first_frame_pos'].numpy()
            first_quat = batch['first_frame_quat'].numpy()
            raw_positions = batch['raw_positions'].numpy()
            
            # === SEQUENCE CHANGE DETECTION (before BIRD runs) ===
            # Detect sequence change early so we reset memory before using it
            if hasattr(test_ds, 'dataset'):
                sample_idx = test_ds.indices[batch_idx]
                curr_seq_path = test_ds.dataset.samples[sample_idx][0]
            else:
                curr_seq_path = test_ds.samples[batch_idx][0]
            
            if prev_seq_path is not None and curr_seq_path != prev_seq_path:
                print(f"[INFO] New sequence detected, resetting memory and chain anchors.")
                mem_state = None
                ant_chain_anchor = None
                bird_chain_anchor = None
            prev_seq_path = curr_seq_path
            
            # === CLOSED-LOOP MAP POINT QUERY ===
            # First window uses dataset map_points, subsequent windows query around BIRD's prediction
            if bird_chain_anchor is not None:
                # Query map points around BIRD's last prediction (closed-loop)
                map_pts_np, map_mask_np = get_map_points_at_position(
                    bird_chain_anchor, centerline_pts, centerline_tree
                )
                map_points = torch.tensor(map_pts_np, dtype=torch.float32).unsqueeze(0).to(device)
                map_mask = torch.tensor(map_mask_np, dtype=torch.bool).unsqueeze(0).to(device)
            else:
                # First window: use dataset map_points (anchored at GT start)
                map_points = batch['map_points'].to(device)
                map_mask = batch['map_mask'].to(device)
            
            # Get ANT predictions + features (5-tuple return)
            ant_pos, delta_pos, delta_quat, visual_tokens, _ = ant_model(
                video, map_points, map_mask, return_features=True
            )
            
            # Get BIRD refinement (4-tuple: pred, mem_state, attn, surprise)
            bird_pred, mem_state, _, _ = bird_model(
                ant_pos, delta_pos, delta_quat, visual_tokens,
                centerline_encoded, centerline_normalized,
                mem_state=mem_state
            )
            
            # Denormalize
            gt_trans_local = gt_deltas[:, :, :3].cpu().numpy() * NORM_MAP_SCALE
            ant_trans_local = ant_pos.cpu().numpy() * NORM_MAP_SCALE
            bird_trans_local = bird_pred.cpu().numpy() * NORM_MAP_SCALE
            
            B, T = video.shape[:2]
            for b in range(B):
                p0_gt = first_pos[b]
                q0 = first_quat[b]
                rot_0 = R.from_quat(q0)
                
                # Sequence change already handled before BIRD runs (see above)
                
                # GT always uses true anchor
                gt_window_global = rot_0.apply(gt_trans_local[b]) + p0_gt
                
                # ANT: open-loop chaining
                if ant_chain_anchor is None:
                    ant_chain_anchor = p0_gt.copy()
                ant_window_global = rot_0.apply(ant_trans_local[b]) + ant_chain_anchor
                ant_chain_anchor = ant_window_global[-1].copy()
                
                # BIRD: open-loop chaining
                if bird_chain_anchor is None:
                    bird_chain_anchor = p0_gt.copy()
                bird_window_global = rot_0.apply(bird_trans_local[b]) + bird_chain_anchor
                bird_chain_anchor = bird_window_global[-1].copy()
                
                # Calculate errors
                ant_end_error = np.linalg.norm(ant_window_global[-1] - gt_window_global[-1])
                bird_end_error = np.linalg.norm(bird_window_global[-1] - gt_window_global[-1])
                print(f"[Window {batch_idx+1}] ANT error: {ant_end_error:.2f}mm, BIRD error: {bird_end_error:.2f}mm")
                
                # Interpolation mode
                if args.interpolate:
                    if hasattr(test_ds, 'dataset'):
                        vid_path, _, start_idx = test_ds.dataset.samples[test_ds.indices[batch_idx]]
                        fs = test_ds.dataset.frame_skip
                    else:
                        vid_path, _, start_idx = test_ds.samples[batch_idx]
                        fs = test_ds.frame_skip
                    
                    display_frames, keyframe_indices = load_full_window_frames(
                        vid_path, start_idx, window_size, fs, args.img_size
                    )
                    total_full_frames = len(display_frames)
                    
                    # Load raw trajectory
                    traj_path = vid_path.replace('video.npy', 'trajectory.npy')
                    traj_mmap = np.load(traj_path, mmap_mode='r')
                    raw_traj_full = traj_mmap[start_idx:start_idx + total_full_frames, :3].copy()
                    
                    # Interpolate all trajectories
                    gt_all = interpolate_positions_for_frames(gt_window_global, keyframe_indices, total_full_frames, centerline_pts, centerline_tree)
                    ant_all = interpolate_positions_for_frames(ant_window_global, keyframe_indices, total_full_frames, centerline_pts, centerline_tree)
                    bird_all = interpolate_positions_for_frames(bird_window_global, keyframe_indices, total_full_frames, centerline_pts, centerline_tree)
                    
                    for f_idx in range(total_full_frames):
                        frame_rgb = display_frames[f_idx]
                        
                        img_3d = render_3d_view_bird(
                            plotter, lung_mesh, centerline_pts, centerline_tree,
                            gt_all[:f_idx+1], ant_all[:f_idx+1], bird_all[:f_idx+1],
                            raw_traj_full[:f_idx+1], gt_all[f_idx], ant_all[f_idx], bird_all[f_idx],
                            raw_traj_full[f_idx], f_idx, total_full_frames
                        )
                        
                        h_3d, w_3d = img_3d.shape[:2]
                        h_frame, w_frame = frame_rgb.shape[:2]
                        scale = h_3d / h_frame
                        new_w_frame = int(w_frame * scale)
                        frame_resized = cv2.resize(frame_rgb, (new_w_frame, h_3d))
                        combined_rgb = np.hstack([frame_resized, img_3d])
                        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                        
                        cv2.putText(combined_bgr, f"Window: {batch_idx+1}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(combined_bgr, f"Frame: {f_idx+1}/{total_full_frames}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        all_frames.append(combined_bgr)
                else:
                    # Keyframe only mode - simplified for now
                    print(f"[INFO] Use --interpolate for smooth visualization")
    
    plotter.close()
    
    # Save video
    output_path = os.path.join(args.output_dir, "test_bird_results.mp4")
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
    parser = argparse.ArgumentParser(description="Test ANT + BIRD models together")
    
    # Model checkpoints
    parser.add_argument('--ant_checkpoint', type=str, required=True,
                        help="Path to trained ANT checkpoint")
    parser.add_argument('--bird_checkpoint', type=str, required=True,
                        help="Path to trained BIRD checkpoint")
    parser.add_argument('--model_mode', type=str, default='s', choices=['xs', 's', 'b', 'm', 'l'],
                        help="Model size (must match checkpoints)")
    
    # Data
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seq_filter', type=str, default=None,
                        help="Filter sequences by name (substring match)")
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./dataset/test/results')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--interpolate', action='store_true',
                        help="Interpolate trajectory for smooth visualization")
    
    args = parser.parse_args()
    test(args)
