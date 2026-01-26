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
from bird import BIRD, BIRD_CONFIGS
from ant_dataset import AntDataset
from constants import NORM_MAP_SCALE, MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS, MAP_POINT_SPACING
from utils.utils import load_centerline_points, filter_connected_component, density_based_sample, find_centerline_path, interpolate_trajectory


def render_3d_view(plotter, lung_mesh, centerline_pts, centerline_tree,
                   gt_traj_current, pred_traj_current, raw_gt_traj_current,
                   current_gt, current_pred, current_raw_gt, frame_idx, total_frames,
                   bird_traj_current=None, current_bird=None):
    """
    Renders the 3D trajectory view to a numpy array.
    Style matches check_ball.py and check_traj.py.
    Shows FPS-downsampled points that the model sees.
    
    Args:
        plotter: PyVista Plotter (off-screen)
        lung_mesh: Loaded lung mesh (pyvista object) or None
        centerline_pts: Centerline points array (N, 3) or None
        centerline_tree: KDTree for centerline points or None
        gt_traj_current: GT trajectory (centerline-projected) up to current frame (M, 3)
        pred_traj_current: ANT Pred trajectory up to current frame (M, 3)
        raw_gt_traj_current: Raw GT trajectory (actual camera positions) up to current frame (M, 3)
        current_gt: Current frame GT position - centerline projected (3,)
        current_pred: Current frame ANT Pred position (3,)
        current_raw_gt: Current frame actual GT position (3,)
        frame_idx: Current frame index in window
        total_frames: Total frames in window
        bird_traj_current: BIRD trajectory up to current frame (M, 3) or None
        current_bird: Current frame BIRD position (3,) or None
        
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
            if len(connected_points) > 0:
                dists = np.linalg.norm(connected_points - p0, axis=1)
                start_idx = np.argmin(dists)
                fps_points, _ = density_based_sample(
                    connected_points, 
                    min_distance=MAP_POINT_SPACING, 
                    start_idx=start_idx,
                    max_points=DEFAULT_MAX_MAP_POINTS
                )
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
    
    # 6. Draw Raw GT Trajectory (Green - actual camera positions)
    if len(raw_gt_traj_current) > 1:
        raw_gt_line = pv.lines_from_points(raw_gt_traj_current)
        plotter.add_mesh(raw_gt_line, color='green', line_width=3, label='Raw GT (actual)')
    
    # 7. Draw Centerline-Projected GT Trajectory (Blue - what model predicts)
    if len(gt_traj_current) > 1:
        gt_line = pv.lines_from_points(gt_traj_current)
        plotter.add_mesh(gt_line, color='blue', line_width=4, label='GT Trajectory')
    
    # 8. Draw Predicted Trajectory (Red - building up as frames proceed)  
    if len(pred_traj_current) > 1:
        pred_line = pv.lines_from_points(pred_traj_current)
        plotter.add_mesh(pred_line, color='red', line_width=4, label='ANT Trajectory')
    
    # 9. Draw BIRD Trajectory (Cyan - if available)
    if bird_traj_current is not None and len(bird_traj_current) > 1:
        bird_line = pv.lines_from_points(bird_traj_current)
        plotter.add_mesh(bird_line, color='cyan', line_width=4, label='BIRD Trajectory')
    
    # 10. Draw current position markers (spheres)
    plotter.add_mesh(pv.Sphere(radius=1.0, center=current_raw_gt), color='green')  # Raw GT
    plotter.add_mesh(pv.Sphere(radius=1.5, center=current_gt), color='blue')  # Centerline GT
    plotter.add_mesh(pv.Sphere(radius=1.5, center=current_pred), color='red')  # ANT Prediction
    if current_bird is not None:
        plotter.add_mesh(pv.Sphere(radius=1.5, center=current_bird), color='cyan')  # BIRD Prediction
    
    # 11. Add Start label at first GT point
    if len(gt_traj_current) > 0:
        plotter.add_point_labels([gt_traj_current[0]], ["Start"], point_size=8, 
                                 text_color='green', always_visible=True, font_size=12)
    
    # 12. Add text overlay (black text for white background)
    text = f"Frame: {frame_idx+1}/{total_frames}\n"
    text += f"GT (mm):   X={current_gt[0]:+.2f} Y={current_gt[1]:+.2f} Z={current_gt[2]:+.2f}\n"
    text += f"ANT (mm):  X={current_pred[0]:+.2f} Y={current_pred[1]:+.2f} Z={current_pred[2]:+.2f}"
    if current_bird is not None:
        text += f"\nBIRD (mm): X={current_bird[0]:+.2f} Y={current_bird[1]:+.2f} Z={current_bird[2]:+.2f}"
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


def load_full_window_frames(vid_path, start_idx, window_size, frame_skip, img_size):
    """
    Loads ALL frames within a window span (not just the skipped ones).
    
    Args:
        vid_path: Path to the video.npy file
        start_idx: Starting frame index
        window_size: Number of prediction points
        frame_skip: Frames between each prediction
        img_size: Size to resize frames to (for model input, not display)
        
    Returns:
        display_frames: np.array of original resolution frames for display (N, H, W, 3) RGB uint8
        keyframe_indices: List of indices in all_frames that correspond to prediction points
    """
    vid_mmap = np.load(vid_path, mmap_mode='r')
    
    # Total frames in window: from start to start + (window_size-1)*frame_skip
    total_frames = (window_size - 1) * frame_skip + 1
    end_idx = start_idx + total_frames
    
    # Make sure we don't go past the end of the video
    actual_end = min(end_idx, len(vid_mmap))
    
    # Load all frames in range
    all_frames_raw = vid_mmap[start_idx:actual_end]
    
    # Keep original quality frames for display (convert to HWC RGB format)
    display_frames = []
    for frame in all_frames_raw:
        frame = np.array(frame)  # Copy from mmap
        
        # Convert to HWC format for display
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            # Channel-first (C, H, W) -> (H, W, C)
            frame = np.transpose(frame, (1, 2, 0))
        elif len(frame.shape) == 3 and frame.shape[-1] == 3:
            # Already HWC format
            pass
        else:
            # Grayscale - stack to 3 channels
            if len(frame.shape) == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
        
        # Convert BGR to RGB if needed (assume video is BGR from OpenCV)
        frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        display_frames.append(frame_rgb)
    
    display_frames = np.array(display_frames)
    
    # Keyframe indices (the frames where we have predictions)
    keyframe_indices = [i * frame_skip for i in range(window_size) if i * frame_skip < len(display_frames)]
    
    return display_frames, keyframe_indices


def interpolate_positions_for_frames(pred_positions, keyframe_indices, total_frames, 
                                     centerline_pts, centerline_tree):
    """
    Interpolate positions for ALL frames based on predictions at keyframes.
    Uses centerline path for smooth interpolation.
    
    Args:
        pred_positions: (K, 3) array of predicted positions at keyframes
        keyframe_indices: List of frame indices where we have predictions
        total_frames: Total number of frames
        centerline_pts: Centerline points for path interpolation
        centerline_tree: KDTree for centerline queries
        
    Returns:
        frame_positions: (N, 3) array of positions for each frame
    """
    frame_positions = np.zeros((total_frames, 3))
    
    # Set positions at keyframes
    for i, kf_idx in enumerate(keyframe_indices):
        if kf_idx < total_frames:
            frame_positions[kf_idx] = pred_positions[i]
    
    # Interpolate between keyframes
    for i in range(len(keyframe_indices) - 1):
        start_kf = keyframe_indices[i]
        end_kf = keyframe_indices[i + 1]
        
        if end_kf >= total_frames:
            end_kf = total_frames - 1
            
        start_pos = pred_positions[i]
        end_pos = pred_positions[i + 1] if i + 1 < len(pred_positions) else pred_positions[i]
        
        # Get centerline path between the two positions
        if centerline_tree is not None:
            path = find_centerline_path(start_pos, end_pos, centerline_pts, centerline_tree)
        else:
            # Linear interpolation fallback
            path = np.array([start_pos, end_pos])
        
        # Distribute path points across the frame range
        num_frames_between = end_kf - start_kf + 1
        path_len = len(path)
        
        for f_idx in range(start_kf, end_kf + 1):
            # Map frame index to path index
            t = (f_idx - start_kf) / max(1, end_kf - start_kf)  # 0 to 1
            path_idx = int(t * (path_len - 1))
            path_idx = min(path_idx, path_len - 1)
            frame_positions[f_idx] = path[path_idx]
    
    return frame_positions


def test(args):
    """
    Main test function.
    Runs inference on the dataset and generates a side-by-side video visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Test on {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset (window_size/frame_skip loaded from config inside dataset)
    # Override frame_skip if specified, or auto-detect for phantom sequences
    dataset_kwargs = dict(
        data_root=os.path.join(args.data_root, "sequences"),
        mode='test',
        img_size=args.img_size,
        chain_mode=args.chain_ant  # Enable overlapping windows for chain mode
    )
    
    # Check if testing phantom sequences - use phantom_frame_skip
    if args.frame_skip is not None:
        dataset_kwargs['frame_skip'] = args.frame_skip
        print(f"[INFO] Overriding frame_skip to {args.frame_skip}")
    elif args.seq_filter and 'phantom' in args.seq_filter.lower():
        # Auto-detect phantom sequence and use phantom_frame_skip
        import json
        config_path = os.path.join(args.data_root, "..", "window_config.json")
        if not os.path.exists(config_path):
            config_path = "window_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        phantom_frame_skip = config.get('phantom_frame_skip', config.get('frame_skip', 60))
        dataset_kwargs['frame_skip'] = phantom_frame_skip
        print(f"[INFO] Detected phantom sequence, using phantom_frame_skip={phantom_frame_skip}")
    
    full_dataset = AntDataset(**dataset_kwargs)
    
    if args.chain_ant:
        print("[INFO] Chain mode enabled for ANT: windows overlap by 1 frame, predictions are chained.")
        if args.batch_size != 1:
            print("[WARNING] Chain mode requires batch_size=1, setting to 1.")
            args.batch_size = 1
    
    if args.interpolate:
        print("[INFO] Interpolation enabled: trajectories will follow centerline for smooth visualization.")
    
    # Get window_size from dataset (loaded from config)
    window_size = full_dataset.window_size
    print(f"[INFO] Using window_size={window_size}, frame_skip={full_dataset.frame_skip}")
    
    # --- SEQUENCE FILTERING ---
    if args.seq_filter:
        print(f"[INFO] Filtering sequences matching: '{args.seq_filter}'")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if args.seq_filter in vp]
        
        if not indices:
            # Try as a relative path to a sequence directory
            alt_seq_path = os.path.join(args.data_root, args.seq_filter)
            if os.path.isdir(alt_seq_path):
                print(f"[INFO] Sequence not in dataset/sequences, trying as path: {alt_seq_path}")
                # Create a new dataset rooted at the parent of the sequence
                parent_dir = os.path.dirname(alt_seq_path)
                alt_dataset_kwargs = dict(
                    data_root=parent_dir,
                    mode='test',
                    img_size=args.img_size,
                    chain_mode=args.chain_ant
                )
                if args.frame_skip is not None:
                    alt_dataset_kwargs['frame_skip'] = args.frame_skip
                
                full_dataset = AntDataset(**alt_dataset_kwargs)
                # Filter to only the specified sequence
                seq_name = os.path.basename(alt_seq_path)
                indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if seq_name in vp]
                
                if not indices:
                    print(f"[ERROR] No sequences found at '{alt_seq_path}'!")
                    return
                    
                print(f"[INFO] Found {len(indices)} windows at custom path.")
                full_dataset = torch.utils.data.Subset(full_dataset, indices)
                test_ds = full_dataset
            else:
                print(f"[ERROR] No sequences matching '{args.seq_filter}' found in dataset!")
                print(f"[ERROR] Also tried as path: {alt_seq_path} (not found)")
                return
        else:
            print(f"[INFO] Found {len(indices)} windows matching filter.")
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            test_ds = full_dataset
    # --- DEBUGGING / OVERFITTING MODES (same as train.py) ---
    elif args.overfit:
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
    
    # Load ANT Model
    ant_model = ActionPredictor(
        window_size=window_size,
        mode=args.model_mode,
        img_size=args.img_size
    ).to(device)
    
    # Load ANT checkpoint
    if args.ant_checkpoint:
        ckpt_path = args.ant_checkpoint
    else:
        print("[ERROR] No ANT checkpoint specified (--ant_checkpoint)!")
        return
    
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return
        
    print(f"[INFO] Loading ANT checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle both old (state_dict only) and new (full checkpoint) formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ant_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] ANT loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        # Old format: checkpoint IS the state dict
        ant_model.load_state_dict(checkpoint)
    ant_model.eval()
    
    # --- Load BIRD Model (optional) ---
    bird_model = None
    centerline_normalized = None
    centerline_encoded = None
    if args.bird_checkpoint:
        print(f"[INFO] Loading BIRD checkpoint: {args.bird_checkpoint}")
        bird_config = BIRD_CONFIGS[args.model_mode]
        
        # Load centerline for BIRD (needed for encoding)
        centerline_path = os.path.join(args.data_root, "static", "centerline.npz")
        centerline_pts_full = load_centerline_points(centerline_path)
        
        if centerline_pts_full is not None:
            # Downsample centerline (like train_bird.py)
            from utils.utils import density_based_sample
            centerline_ds, _ = density_based_sample(centerline_pts_full, min_distance=MAP_POINT_SPACING)
            print(f"[INFO] Downsampled centerline: {len(centerline_ds)} points")
            
            # Create BIRD model (pass ant_mode, BIRD gets visual_dim internally)
            bird_model = BIRD(
                ant_mode=args.model_mode,
                **bird_config
            ).to(device)
            
            # Load BIRD weights
            bird_ckpt = torch.load(args.bird_checkpoint, map_location=device)
            if isinstance(bird_ckpt, dict) and 'bird_state_dict' in bird_ckpt:
                bird_model.load_state_dict(bird_ckpt['bird_state_dict'])
                print(f"[INFO] BIRD loaded from epoch {bird_ckpt.get('epoch', '?')}")
            elif isinstance(bird_ckpt, dict) and 'model_state_dict' in bird_ckpt:
                bird_model.load_state_dict(bird_ckpt['model_state_dict'])
                print(f"[INFO] BIRD loaded from epoch {bird_ckpt.get('epoch', '?')}")
            else:
                bird_model.load_state_dict(bird_ckpt)
            bird_model.eval()
            
            # Encode centerline (normalized, like train_bird.py)
            centerline_normalized = torch.tensor(centerline_ds / NORM_MAP_SCALE, dtype=torch.float32).to(device)
            with torch.no_grad():
                centerline_encoded = bird_model.encode_centerline(centerline_normalized)
            print(f"[INFO] BIRD ready with {len(centerline_ds)} centerline points")
        else:
            print("[ERROR] Cannot load BIRD without centerline!")
            args.bird_checkpoint = None
    
    # --- LOAD 3D ASSETS (like check_traj.py) ---
    # Load lung mesh
    lung_mesh = None
    lung_path = os.path.join(args.data_root, "..", "patient", "lungs.obj")
    if os.path.exists(lung_path):
        print(f"[INFO] Loading lung mesh: {lung_path}")
        lung_mesh = pv.read(lung_path)
    else:
        print(f"[WARNING] Lung mesh not found at {lung_path}")
    
    # Load centerline (full, for visualization)
    centerline_path = os.path.join(args.data_root, "static", "centerline.npz")
    centerline_pts = load_centerline_points(centerline_path)
    centerline_tree = None
    if centerline_pts is not None:
        print(f"[INFO] Loaded {len(centerline_pts)} centerline points for visualization")
        centerline_tree = cKDTree(centerline_pts)
    
    # --- INFERENCE AND VIDEO GENERATION ---
    print("[INFO] Running inference and generating video...")
    
    # Initialize PyVista off-screen plotter for 3D rendering
    pv.start_xvfb()  # For headless environments
    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    plotter.set_background('white')  # White background like check_ball.py
    
    # Collect all frames
    all_frames = []
    
    # --- State for chained predictions ---
    # In chain mode, we carry forward the last predicted position as the anchor for the next window
    chain_pred_anchor = None  # Will be set after first window
    prev_seq_path = None  # Track sequence changes to reset chain
    bird_mem_state = None  # BIRD memory state (chains across windows)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            video = batch['video'].to(device)  # (B, T, C, H, W)
            gt_deltas = batch['actions'].to(device)  # (B, T, 6) - LOCAL frame
            map_points = batch['map_points'].to(device)  # (B, T, K, 3)
            map_mask = batch['map_mask'].to(device)  # (B, T, K)
            
            # Get first frame poses for transforming back to global
            first_pos = batch['first_frame_pos'].numpy()  # (B, 3)
            first_quat = batch['first_frame_quat'].numpy()  # (B, 4)
            
            # Get raw GT positions (global frame) for visualization
            raw_positions = batch['raw_positions'].numpy()  # (B, T, 3)
            
            # Forward pass - ANT model (get features if BIRD is enabled)
            if bird_model is not None:
                pred_pos, delta_pos, delta_quat, visual_tokens, attn_probs = ant_model(
                    video, map_points=map_points, map_mask=map_mask, return_features=True
                )
            else:
                pred_pos, pred_quat = ant_model(video, map_points=map_points, map_mask=map_mask)
            
            # Get GT translation - LOCAL frame
            # Dataset stores actions NORMALIZED (divided by NORM_MAP_SCALE in ant_dataset.py)
            # So we need to denormalize to get mm
            gt_trans_local = gt_deltas.cpu().numpy() * NORM_MAP_SCALE  # (B, T, 3)
            
            # Denormalize prediction to mm (model outputs normalized values to match training targets)
            pred_pos_local = pred_pos.cpu().numpy() * NORM_MAP_SCALE  # (B, T, 3)
            # Convert video back to displayable format
            # Dataset stores RGB, normalized to [-1, 1]
            video_np = video.cpu().numpy()  # (B, T, C, H, W)
            video_np = (video_np + 1.0) * 127.5  # Denormalize from [-1, 1] to [0, 255]
            video_np = video_np.clip(0, 255).astype(np.uint8)
            
            # Process each sample in batch
            B, T = video_np.shape[:2]
            for b in range(B):
                # Get first frame's global pose for this sample (from GT)
                p0_gt = first_pos[b]  # (3,)
                q0 = first_quat[b]  # (4,)
                rot_0 = R.from_quat(q0)  # Rotation from frame 0's local to global
                
                # Check if we're still in the same sequence (for chaining)
                # Get current sequence path from dataset
                if hasattr(test_ds, 'dataset'):  # Subset
                    sample_idx = test_ds.indices[batch_idx * args.batch_size + b]
                    curr_seq_path = test_ds.dataset.samples[sample_idx][0]
                else:
                    curr_seq_path = test_ds.samples[batch_idx * args.batch_size + b][0]
                
                # Reset chain if sequence changed
                if prev_seq_path is not None and curr_seq_path != prev_seq_path:
                    if args.chain_ant:
                        print(f"[CHAIN] New sequence detected, resetting chain anchor.")
                        chain_pred_anchor = None
                    if bird_model is not None:
                        print(f"[BIRD] New sequence detected, resetting memory.")
                        bird_mem_state = None
                prev_seq_path = curr_seq_path
                
                # Transform LOCAL to GLOBAL coordinates
                # GT always uses the actual GT anchor (for reference/comparison)
                gt_window_global = rot_0.apply(gt_trans_local[b]) + p0_gt  # (T, 3)
                
                # For predictions: in chain mode, use the chained anchor instead of GT anchor
                if args.chain_ant:
                    if chain_pred_anchor is None:
                        # First window: initialize with GT first frame
                        chain_pred_anchor = p0_gt.copy()
                        print(f"[Window 1] Starting from GT anchor at {p0_gt}")
                    
                    # Use the chain anchor (previous prediction end point) instead of GT
                    pred_window_global = rot_0.apply(pred_pos_local[b]) + chain_pred_anchor  # (T, 3)
                    ant_anchor_offset = np.linalg.norm(chain_pred_anchor - p0_gt)
                    
                    # Update chain anchor to the END of this window's prediction
                    chain_pred_anchor = pred_window_global[-1].copy()
                else:
                    # Normal mode: predictions use GT anchor (no cascading)
                    pred_window_global = rot_0.apply(pred_pos_local[b]) + p0_gt  # (T, 3)
                    ant_anchor_offset = 0.0
                
                # --- BIRD Inference (if enabled) ---
                bird_window_global = None
                bird_anchor_offset = 0.0
                if bird_model is not None:
                    # Transform ANT predictions to global frame (normalized)
                    ant_pos_global = torch.tensor(pred_window_global / NORM_MAP_SCALE, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Get visual features for this sample
                    vis_tokens_b = visual_tokens[b:b+1]  # (1, T, D)
                    delta_pos_b = delta_pos[b:b+1]  # (1, T, 3)
                    delta_quat_b = delta_quat[b:b+1]  # (1, T, 4)
                    
                    # Run BIRD (BIRD chains memory internally across windows of same sequence)
                    p_refined, bird_mem_state, _, _ = bird_model(
                        ant_pos_global, delta_pos_b, delta_quat_b, vis_tokens_b,
                        centerline_encoded, centerline_normalized,
                        mem_state=bird_mem_state
                    )
                    
                    # Store BIRD predictions in global mm
                    bird_window_global = p_refined.cpu().numpy()[0] * NORM_MAP_SCALE  # (T, 3)
                    
                    # Compute BIRD anchor offset (BIRD's first frame vs GT first frame)
                    bird_anchor_offset = np.linalg.norm(bird_window_global[0] - gt_window_global[0])
                
                # --- Print Drift (BEFORE frame prints) ---
                print_line = f"[Window {batch_idx+1}] ANT: drift={ant_anchor_offset:.2f}mm"
                if bird_window_global is not None:
                    drift_improvement = ant_anchor_offset - bird_anchor_offset
                    drift_label = " (BIRD improved)" if drift_improvement > 0 else ""
                    print_line += f" | BIRD: drift={bird_anchor_offset:.2f}mm | Δ={drift_improvement:+.2f}mm{drift_label}"
                print(print_line)
                
                # Also keep local values for printing
                gt_window_local = gt_trans_local[b]  # (T, 3)
                pred_window_local = pred_pos_local[b]  # (T, 3)
                
                # === FRAME PROCESSING ===
                # When interpolate mode is enabled, we load ALL frames and compute positions for each.
                # Otherwise, we just use the keyframes from the dataset.
                
                if args.interpolate:
                    # --- FULL FRAME INTERPOLATION MODE ---
                    # Get sample info to load full video
                    if hasattr(test_ds, 'dataset'):  # Subset
                        sample_idx_for_path = test_ds.indices[batch_idx * args.batch_size + b]
                        vid_path, _, start_idx = test_ds.dataset.samples[sample_idx_for_path]
                        frame_skip = test_ds.dataset.frame_skip
                    else:
                        vid_path, _, start_idx = test_ds.samples[batch_idx * args.batch_size + b]
                        frame_skip = test_ds.frame_skip
                    
                    # Load ALL frames in this window (not just keyframes) - original quality for display
                    display_frames, keyframe_indices = load_full_window_frames(
                        vid_path, start_idx, window_size, frame_skip, args.img_size
                    )
                    total_full_frames = len(display_frames)
                    
                    # Load raw trajectory (actual camera positions) for all frames in this window
                    traj_path = vid_path.replace('video.npy', 'trajectory.npy')
                    traj_mmap = np.load(traj_path, mmap_mode='r')
                    raw_traj_full = traj_mmap[start_idx:start_idx + total_full_frames, :3].copy()  # (N, 3) positions only
                    
                    # Interpolate GT positions for all frames
                    gt_all_positions = interpolate_positions_for_frames(
                        gt_window_global, keyframe_indices, total_full_frames,
                        centerline_pts, centerline_tree
                    )
                    
                    # Interpolate Predicted positions for all frames
                    pred_all_positions = interpolate_positions_for_frames(
                        pred_window_global, keyframe_indices, total_full_frames,
                        centerline_pts, centerline_tree
                    )
                    
                    # Interpolate BIRD positions if available
                    bird_all_positions = None
                    if bird_window_global is not None:
                        bird_all_positions = interpolate_positions_for_frames(
                            bird_window_global, keyframe_indices, total_full_frames,
                            centerline_pts, centerline_tree
                        )
                    
                    # Process each frame (display_frames is already RGB uint8, HWC format)
                    for f_idx in range(total_full_frames):
                        # Use original quality frame directly (already RGB)
                        frame_rgb = display_frames[f_idx]
                        
                        current_gt_global = gt_all_positions[f_idx]
                        current_pred_global = pred_all_positions[f_idx]
                        
                        # Trajectory up to current frame
                        gt_traj_current = gt_all_positions[:f_idx+1]
                        pred_traj_current = pred_all_positions[:f_idx+1]
                        # Use actual raw trajectory positions
                        raw_gt_traj_current = raw_traj_full[:f_idx+1]
                        current_raw_gt = raw_traj_full[f_idx]
                        
                        # BIRD trajectory up to current frame
                        bird_traj_current = None
                        current_bird = None
                        if bird_all_positions is not None:
                            bird_traj_current = bird_all_positions[:f_idx+1]
                            current_bird = bird_all_positions[f_idx]
                        
                        # Render 3D view (with BIRD trajectory)
                        img_3d = render_3d_view(
                            plotter, lung_mesh, centerline_pts, centerline_tree,
                            gt_traj_current, pred_traj_current, raw_gt_traj_current,
                            current_gt_global, current_pred_global, current_raw_gt,
                            f_idx, total_full_frames,
                            bird_traj_current=bird_traj_current, current_bird=current_bird
                        )
                        
                        # Combine video frame and 3D view
                        h_3d, w_3d = img_3d.shape[:2]
                        h_frame, w_frame = frame_rgb.shape[:2]
                        scale = h_3d / h_frame
                        new_w_frame = int(w_frame * scale)
                        frame_resized = cv2.resize(frame_rgb, (new_w_frame, h_3d))
                        combined_rgb = np.hstack([frame_resized, img_3d])
                        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Add overlay
                        cv2.putText(combined_bgr, f"Window: {batch_idx+1}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(combined_bgr, f"Frame: {f_idx+1}/{total_full_frames} (interpolated)", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        all_frames.append(combined_bgr)
                    
                    print(f"[Batch {batch_idx+1} | Sample {b+1}] Processed {total_full_frames} frames (full interpolation)")
                    
                else:
                    # --- KEYFRAME ONLY MODE (original behavior) ---
                    for t in range(T):
                        frame_bgr = video_np[b, t].transpose(1, 2, 0)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        
                        current_gt_local = gt_window_local[t]
                        current_pred_local = pred_window_local[t]
                        current_gt_global = gt_window_global[t]
                        current_pred_global = pred_window_global[t]
                        
                        # Print GT, ANT Pred, and BIRD Pred for each frame
                        print_line = (f"  [Frame {t+1}/{T}] "
                              f"GT: ({current_gt_global[0]:+.3f}, {current_gt_global[1]:+.3f}, {current_gt_global[2]:+.3f}) mm | "
                              f"ANT: ({current_pred_global[0]:+.3f}, {current_pred_global[1]:+.3f}, {current_pred_global[2]:+.3f}) mm")
                        if bird_window_global is not None:
                            current_bird_global = bird_window_global[t]
                            print_line += f" | BIRD: ({current_bird_global[0]:+.3f}, {current_bird_global[1]:+.3f}, {current_bird_global[2]:+.3f}) mm"
                        print(print_line)
                        
                        raw_gt_traj_current = raw_positions[b, :t+1]
                        current_raw_gt = raw_positions[b, t]
                        gt_traj_current = gt_window_global[:t+1]
                        pred_traj_current = pred_window_global[:t+1]
                        # Get BIRD trajectory if available
                        bird_traj_current = None
                        current_bird_global = None
                        if bird_window_global is not None:
                            bird_traj_current = bird_window_global[:t+1]
                            current_bird_global = bird_window_global[t]
                        
                        img_3d = render_3d_view(
                            plotter, lung_mesh, centerline_pts, centerline_tree,
                            gt_traj_current, pred_traj_current, raw_gt_traj_current,
                            current_gt_global, current_pred_global, current_raw_gt,
                            t, T,
                            bird_traj_current=bird_traj_current, current_bird=current_bird_global
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
                        cv2.putText(combined_bgr, f"Frame: {t+1}/{T}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        all_frames.append(combined_bgr)
                    
                    # --- Print Error Summary (AFTER all frame prints) ---
                    ant_error = np.linalg.norm(pred_window_global - gt_window_global, axis=1).mean()
                    summary_line = f"[Window {batch_idx+1} Summary] ANT: error={ant_error:.2f}mm"
                    if bird_window_global is not None:
                        bird_error = np.linalg.norm(bird_window_global - gt_window_global, axis=1).mean()
                        improvement = ant_error - bird_error
                        improved_label = " (BIRD improved)" if improvement > 0 else ""
                        summary_line += f" | BIRD: error={bird_error:.2f}mm | Δ={improvement:+.2f}mm{improved_label}"
                    print(summary_line)

    
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
        # Check for inconsistent frame shapes
        shapes = [f.shape for f in all_frames]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            print(f"[WARNING] Inconsistent frame shapes detected: {unique_shapes}")
            # Find the most common shape and filter to only those frames
            from collections import Counter
            shape_counts = Counter(shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]
            print(f"[WARNING] Keeping only frames with shape {most_common_shape}")
            all_frames = [f for f in all_frames if f.shape == most_common_shape]
        
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for better compatibility
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
    parser.add_argument('--ant_checkpoint', type=str, default=None, help="Path to ANT checkpoint file (required)")
    parser.add_argument('--bird_checkpoint', type=str, default=None, help="Path to BIRD checkpoint file (optional)")
    parser.add_argument('--model_mode', type=str, default='s', choices=['xs', 's', 'b', 'm', 'l'])
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (1 recommended for video)")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overfit', action='store_true', help="Test on seq_test only (matches overfit training)")
    parser.add_argument('--debug_one', action='store_true', help="Test on SINGLE batch (matches debug_one training)")
    parser.add_argument('--img_size', type=int, default=128, help="Image resolution (default: 128)")
    parser.add_argument('--seq_filter', type=str, default=None, help="Filter sequences by name (substring match, e.g. 'seq_001')")
    
    # Test-specific arguments
    parser.add_argument('--output_dir', type=str, default='./dataset/test/results', help="Output directory for videos")
    parser.add_argument('--fps', type=int, default=10, help="Output video FPS")
    parser.add_argument('--chain_ant', action='store_true', help="Chain ANT predictions: last pred of window N becomes anchor for window N+1")
    parser.add_argument('--interpolate', action='store_true', help="Interpolate trajectory along centerline for smooth visualization")
    parser.add_argument('--frame_skip', type=int, default=None, help="Override frame_skip (default: use config). Use 20 for phantom sequences.")
    
    args = parser.parse_args()
    test(args)

