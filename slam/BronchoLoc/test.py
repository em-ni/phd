import os
import glob
import argparse
import torch
import numpy as np
import cv2
import pyvista as pv
from tqdm import tqdm

# Force Headless Mode for WSL/Server
pv.OFF_SCREEN = True
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1" 

from deep_lung_st import DeepLungST
from constants import get_norm_center_tensor, NORM_SCALE, NORM_CENTER

# ==============================================================================
# VISUALIZATION LOGIC (De-Normalize for Plotting)
# ==============================================================================
def save_long_video(video_np, pred_traj_particles, gt_traj, lung_mesh, save_path):
    print(f"[VIZ] Generating video: {save_path}")
    
    # De-normalize trajectories for visualization (Back to mm)
    # pred: (N, K, 3) -> (N, K, 3)
    pred_np = pred_traj_particles.numpy() * NORM_SCALE + NORM_CENTER
    # gt: (N, 3) -> (N, 3)
    gt_np = gt_traj.numpy() * NORM_SCALE + NORM_CENTER
    
    # Handle Video Format (Assume N, C, H, W or N, H, W, C)
    # We want N, H, W, C for cv2
    if video_np.ndim == 4:
        if video_np.shape[1] == 3: # N, C, H, W -> N, H, W, C
            video_np = np.transpose(video_np, (0, 2, 3, 1))
            
    # Ensure uint8
    if video_np.dtype != np.uint8:
        # If float 0..1 or -1..1, handle it. 
        # But raw_video_np from npy is likely uint8. 
        # Just in case it's float:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    N, H, W, C = video_np.shape
    
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 600))
    
    if lung_mesh:
        plotter.add_mesh(lung_mesh, color='wheat', opacity=0.15, label='Lungs CAD')
        # Center camera on the trajectory
        center = np.mean(gt_np, axis=0)
        plotter.camera.position = (center[0], center[1] - 150, center[2] + 30)
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        plotter.camera.zoom(1.2)
    else:
        center = np.mean(gt_np, axis=0)
        plotter.camera.position = (center[0], center[1] - 200, center[2] + 50)
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)

    out_h = 600
    scale_factor = out_h / H
    out_w_video = int(W * scale_factor)
    out_w_3d = 1000
    total_w = out_w_video + out_w_3d
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (total_w, out_h))
    
    step = 2 
    for t in tqdm(range(0, N, step), desc="Rendering Frames"):
        frame_img = video_np[t]
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        frame_img = cv2.resize(frame_img, (out_w_video, out_h), interpolation=cv2.INTER_NEAREST)
        
        # 1. Draw Ground Truth
        if t > 1:
            points_gt = gt_np[:t+1]
            if len(points_gt) > 1:
                # Approximate tube using lines for speed or actual tube
                # For long sequences, creating tubes every frame is slow in PyVista
                # We draw just the tip and a trail
                
                # Draw full trail as a line (lighter)
                # plotter.add_mesh(pv.lines_from_points(points_gt), color='green', opacity=0.5, name='gt_trail')
                
                # Draw recent tail as tube
                tail_len = 20
                start_tail = max(0, t - tail_len)
                if t - start_tail > 1:
                    tube_gt = pv.lines_from_points(points_gt[start_tail:t+1]).tube(radius=1.0)
                    plotter.add_mesh(tube_gt, color='green', name='gt_trace_recent')

        # 2. Draw Multi-Hypothesis Particles
        K = pred_np.shape[1]
        if t > 1:
            for k in range(K):
                # Draw only recent history to keep view clean
                tail_len = 15
                start_tail = max(0, t - tail_len)
                points_p = pred_np[start_tail:t+1, k]
                
                if len(points_p) > 1:
                    tube_p = pv.lines_from_points(points_p).tube(radius=0.3)
                    plotter.add_mesh(tube_p, color='cyan', opacity=0.6, name=f'p_trace_{k}')
        
        # Draw Tips
        plotter.add_mesh(pv.Sphere(radius=2.0, center=gt_np[t]), color='green', name='gt_tip')
        
        # Draw particle tips
        for k in range(K):
             plotter.add_mesh(pv.Sphere(radius=0.8, center=pred_np[t, k]), color='cyan', name=f'p_tip_{k}')

        # 4. Composite Image
        img_3d = plotter.screenshot(return_img=True, transparent_background=False)
        img_3d = cv2.resize(img_3d, (out_w_3d, out_h))
        img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
        
        combined = np.hstack([frame_img, img_3d])
        out.write(combined)
    
    plotter.close()
    out.release()

# ==============================================================================
# MAIN TEST LOGIC
# ==============================================================================
def run_long_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Testing on {device}")
    
    static_dir = os.path.join(args.data_root, "static")
    
    lung_obj_path = os.path.join("patient", "lungs.obj")
    if not os.path.exists(lung_obj_path):
        lung_obj_path = os.path.join(static_dir, "lungs.obj")
    
    lung_mesh = None
    if os.path.exists(lung_obj_path):
        print(f"[INFO] Loading 3D Lung Mesh from {lung_obj_path}")
        lung_mesh = pv.read(lung_obj_path)

    # Load Static Tensors
    try:
        sdf_vol = torch.load(os.path.join(static_dir, "lung_sdf.pt"), map_location=device)
        grid_trans = torch.from_numpy(np.load(os.path.join(static_dir, "grid_transform.npy"))).float().to(device)
        graph_data = np.load(os.path.join(static_dir, "deep_lung_graph.npz"))
    except FileNotFoundError:
        print(f"[ERROR] Static data missing")
        return

    node_pos = torch.from_numpy(graph_data['node_pos']).float().to(device)
    edge_index = torch.from_numpy(graph_data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(graph_data['edge_attr']).float().to(device)

    # Stats for Model Init
    norm_center_t = get_norm_center_tensor(device)
    
    # Load Model
    model = DeepLungST(
        args.t_frames, sdf_vol, grid_trans, mode=args.model_mode,
        norm_center=norm_center_t, norm_scale=NORM_SCALE
    ).to(device)
    
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    sequences_dir = os.path.join(args.data_root, "test")
    seq_folders = sorted(glob.glob(os.path.join(sequences_dir, "seq_*")))
    if not seq_folders:
        # Fallback to train folder if test is empty just to see result
        sequences_dir = os.path.join(args.data_root, "sequences")
        seq_folders = sorted(glob.glob(os.path.join(sequences_dir, "seq_*")))

    os.makedirs(args.output_dir, exist_ok=True)

    for seq_idx, seq_dir in enumerate(seq_folders):
        if seq_idx >= args.num_viz: break 
        print(f"\n[INFO] Processing: {os.path.basename(seq_dir)}")
        
        try:
            full_video_np = np.load(os.path.join(seq_dir, "video.npy"))
            full_traj_np = np.load(os.path.join(seq_dir, "trajectory.npy"))
        except:
            continue

        # Keep raw video for visualization
        raw_video_np = full_video_np.copy()

        # Resize video to 128x128 manually here since we aren't using Dataset
        # (Assuming dataset does resizing, we must match it)
        resized_frames = []
        for f in full_video_np:
            if f.shape[0] == 3: # CHW -> HWC
                f = np.transpose(f, (1, 2, 0))
                f = cv2.resize(f, (128, 128))
                f = np.transpose(f, (2, 0, 1))
            else:
                f = cv2.resize(f, (128, 128))
            resized_frames.append(f)
        full_video_np = np.array(resized_frames)

        video_tensor = torch.from_numpy(full_video_np).float()
        if video_tensor.shape[-1] == 3: 
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = (video_tensor / 127.5) - 1.0
        
        # Raw GT (mm)
        gt_pos_raw = torch.from_numpy(full_traj_np[:, :3]).float().to(device)
        
        # Normalize GT for Model Input
        gt_pos_norm = (gt_pos_raw - norm_center_t) / NORM_SCALE
        
        N_frames = video_tensor.shape[0]
        T = args.t_frames
        
        full_pred_traj_norm = []
        
        # Start with ground truth (Normalized)
        current_tracker_pos = gt_pos_norm[0].unsqueeze(0) 
        
        print("  > Inference...")
        for t in range(0, N_frames - T + 1, T):
            batch_video = video_tensor[t : t+T].unsqueeze(0).to(device) 
            
            with torch.no_grad():
                # Physics ON for testing!
                pred_block, _, _, _ = model(
                    batch_video, node_pos, edge_index, edge_attr, current_tracker_pos, 
                    physics_on=True, resample=True
                )
            
            pred_block_cpu = pred_block[0].cpu().permute(1, 0, 2)
            full_pred_traj_norm.append(pred_block_cpu)
            
            # Carry over state
            current_tracker_pos = pred_block[:, :, -1, :]
        
        if len(full_pred_traj_norm) > 0:
            # Combined Preds (Normalized)
            full_pred_traj_norm = torch.cat(full_pred_traj_norm, dim=0)
            
            valid_len = full_pred_traj_norm.shape[0]
            valid_gt_norm = gt_pos_norm[:valid_len].cpu()
            
            # Use raw video for viz (sliced to valid length)
            valid_raw_video = raw_video_np[:valid_len]
            
            # Calc ADE in Normalized Space
            mean_traj_norm = full_pred_traj_norm.mean(dim=1)
            ade_norm = torch.norm(mean_traj_norm - valid_gt_norm, dim=1).mean().item()
            
            # Convert ADE to mm for readable log
            ade_mm = ade_norm * NORM_SCALE
            print(f"  > ADE: {ade_mm:.2f} mm")
            
            save_name = os.path.join(args.output_dir, f"{os.path.basename(seq_dir)}_ADE_{ade_mm:.1f}mm.mp4")
            save_long_video(valid_raw_video, full_pred_traj_norm, valid_gt_norm, lung_mesh, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./dataset/test/results')
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'm', 'l'])
    parser.add_argument('--t_frames', type=int, default=16)
    parser.add_argument('--num_viz', type=int, default=1)
    
    args = parser.parse_args()
    run_long_inference(args)