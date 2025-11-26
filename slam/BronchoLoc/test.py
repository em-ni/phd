import os
import glob
import argparse
import torch
import numpy as np
import cv2
import pyvista as pv
from tqdm import tqdm

# --- FIX 1: Force Headless Mode for WSL ---
# This prevents PyVista from looking for a monitor and crashing
pv.OFF_SCREEN = True
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1" 

from deep_lung_st import DeepLungST

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================
def save_long_video(video_tensor, pred_traj, gt_traj, lung_mesh, save_path):
    print(f"[VIZ] Generating video: {save_path}")
    
    # Data is already on CPU from the main loop
    video_np = video_tensor.numpy()
    pred_np = pred_traj.numpy()
    gt_np = gt_traj.numpy()
    
    N, C, H, W = video_np.shape
    
    # Un-normalize video
    video_np = (video_np + 1.0) * 127.5
    video_np = np.clip(video_np, 0, 255).astype(np.uint8)
    video_np = np.transpose(video_np, (0, 2, 3, 1))
    
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 600))
    
    if lung_mesh:
        plotter.add_mesh(lung_mesh, color='wheat', opacity=0.25, label='Lungs CAD')
        center = np.array(lung_mesh.center)
        bounds = lung_mesh.bounds
        max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        cam_pos = (center[0], center[1] - max_dim * 1.5, center[2] + max_dim * 0.2)
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        plotter.camera.zoom(1.0)
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
        
        if t > 1:
            points_gt = gt_np[:t+1]
            if len(points_gt) > 1:
                tube_gt = pv.lines_from_points(points_gt).tube(radius=1.0)
                plotter.add_mesh(tube_gt, color='green', name='gt_trace')

        if t > 1:
            points_pred = pred_np[:t+1]
            if len(points_pred) > 1:
                tube_pred = pv.lines_from_points(points_pred).tube(radius=1.5)
                plotter.add_mesh(tube_pred, color='blue', name='pred_trace')
        
        plotter.add_mesh(pv.Sphere(radius=1.5, center=gt_np[t]), color='green', name='gt_tip')
        plotter.add_mesh(pv.Sphere(radius=1.5, center=pred_np[t]), color='blue', name='pred_tip')

        img_3d = plotter.screenshot(return_img=True, transparent_background=False)
        img_3d = cv2.resize(img_3d, (out_w_3d, out_h))
        img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
        
        combined = np.hstack([frame_img, img_3d])
        
        overlay = combined.copy()
        cv2.rectangle(overlay, (out_w_video, 0), (out_w_video + 400, 50), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, combined, 0.4, 0, combined)
        cv2.putText(combined, f"Frame: {t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Blue: Prediction | Green: GT", (out_w_video + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
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
    
    # Resolve Lung Mesh Path
    lung_obj_path = os.path.join("patient", "lungs.obj")
    if not os.path.exists(lung_obj_path):
        lung_obj_path = os.path.join(static_dir, "lungs.obj")
    
    lung_mesh = None
    if os.path.exists(lung_obj_path):
        print(f"[INFO] Loading 3D Lung Mesh from {lung_obj_path}")
        lung_mesh = pv.read(lung_obj_path)
    else:
        print("[WARNING] No lungs.obj found. Visualization will be trajectory only.")

    # Load Static Tensors
    try:
        sdf_vol = torch.load(os.path.join(static_dir, "lung_sdf.pt"), map_location=device)
        grid_trans = torch.from_numpy(np.load(os.path.join(static_dir, "grid_transform.npy"))).float().to(device)
        graph_data = np.load(os.path.join(static_dir, "deep_lung_graph.npz"))
    except FileNotFoundError:
        print(f"[ERROR] Static data missing in {static_dir}")
        return

    node_pos = torch.from_numpy(graph_data['node_pos']).float().to(device)
    edge_index = torch.from_numpy(graph_data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(graph_data['edge_attr']).float().to(device)

    # Load Model
    model = DeepLungST(args.t_frames, sdf_vol, grid_trans, mode=args.model_mode).to(device)
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Find Sequences
    sequences_dir = os.path.join(args.data_root, "test")
    seq_folders = sorted(glob.glob(os.path.join(sequences_dir, "seq_*")))
    
    if not seq_folders:
        print(f"[ERROR] No sequences found in {sequences_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for seq_idx, seq_dir in enumerate(seq_folders):
        if seq_idx >= args.num_viz: break 
        
        print(f"\n[INFO] Processing Sequence: {os.path.basename(seq_dir)}")
        
        try:
            print("  > Loading Numpy files...")
            full_video_np = np.load(os.path.join(seq_dir, "video.npy"))
            full_traj_np = np.load(os.path.join(seq_dir, "trajectory.npy"))
            print(f"  > Loaded {len(full_video_np)} frames.")
        except Exception as e:
            print(f"Skipping {seq_dir}: {e}")
            continue

        # --- FIX 2: VRAM Optimization ---
        # Keep the main video tensor on CPU to prevent OOM
        video_tensor = torch.from_numpy(full_video_np).float()
        if video_tensor.shape[-1] == 3: 
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = (video_tensor / 127.5) - 1.0
        
        # Ground truth can go to GPU (it's small, just coordinates)
        gt_pos_tensor = torch.from_numpy(full_traj_np[:, :3]).float().to(device)
        
        N_frames = video_tensor.shape[0]
        T = args.t_frames
        
        full_pred_traj = []
        current_tracker_pos = gt_pos_tensor[0].unsqueeze(0) 
        
        print("  > Starting Inference Loop...")
        for t in range(0, N_frames - T + 1, T):
            
            # --- FIX 2 (Cont.): Stream Batch to GPU ---
            # Only move the small chunk we need right now
            batch_video = video_tensor[t : t+T].unsqueeze(0).to(device) 
            
            with torch.no_grad():
                pred_block, _ = model(batch_video, node_pos, edge_index, edge_attr, current_tracker_pos)
            
            # Move result back to CPU immediately to free VRAM
            pred_block_cpu = pred_block[0].cpu()
            full_pred_traj.append(pred_block_cpu)
            
            # Update tracker state (keep on device for next iteration)
            current_tracker_pos = pred_block[:, -1, :] 
        
        if len(full_pred_traj) > 0:
            full_pred_traj = torch.cat(full_pred_traj, dim=0) # (N_processed, 3) on CPU
            
            valid_len = full_pred_traj.shape[0]
            valid_gt = gt_pos_tensor[:valid_len].cpu() # Move GT to CPU for metric calc
            valid_video = video_tensor[:valid_len]     # Already on CPU
            
            ade = torch.norm(full_pred_traj - valid_gt, dim=1).mean().item()
            print(f"  > Full Sequence ADE: {ade:.2f} mm")
            
            save_name = os.path.join(args.output_dir, f"{os.path.basename(seq_dir)}_ADE_{ade:.1f}.mp4")
            save_long_video(valid_video, full_pred_traj, valid_gt, lung_mesh, save_name)
        else:
            print("  > Sequence too short.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./dataset/test/results')
    parser.add_argument('--model_mode', type=str, default='tiny', choices=['tiny', 'big'])
    parser.add_argument('--t_frames', type=int, default=16)
    parser.add_argument('--num_viz', type=int, default=1)
    
    args = parser.parse_args()
    run_long_inference(args)