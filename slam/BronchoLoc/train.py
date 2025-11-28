import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from deep_lung_st import DeepLungST
from deep_lung_dataset import DeepLungDataset 
from constants import get_norm_center_tensor, NORM_SCALE

def multi_hypothesis_loss(
    pred_traj_particles, gt_traj, violations, 
    visual_proj, map_proj, map_node_positions, initial_gt_pos, 
    sdf_lambda=10.0, norm_center=None, norm_scale=1.0
):
    B, K, T, _ = pred_traj_particles.shape
    
    # 1. Best-of-K Trajectory Loss
    gt_expanded = gt_traj.unsqueeze(1)
    mse_per_particle = torch.sum((pred_traj_particles - gt_expanded)**2, dim=(2, 3))
    best_loss_vals, best_indices = torch.min(mse_per_particle, dim=1)
    loss_pose = best_loss_vals.mean()
    
    # 2. Geometry Violation Loss
    loss_geo = violations.mean()
    
    # 3. Smoothness
    batch_indices = torch.arange(B, device=pred_traj_particles.device)
    best_trajs = pred_traj_particles[batch_indices, best_indices] 
    vel = best_trajs[:, 1:] - best_trajs[:, :-1]
    accel = vel[:, 1:] - vel[:, :-1]
    loss_smooth = torch.mean(accel**2)
    
    # 4. Map Retrieval (InfoNCE) - Dense Frame-wise
    # gt_traj: (B, T, 3) -> World Space
    gt_world = (gt_traj * norm_scale) + norm_center
    
    # Calculate target node for EACH frame
    # gt_world: (B, T, 3), map_node_positions: (N, 3)
    # dists: (B, T, N)
    dists = torch.cdist(gt_world, map_node_positions)
    target_node_indices = torch.argmin(dists, dim=2) # (B, T)
    
    # visual_proj: (B, T, D)
    # map_proj: (N, D)
    v_norm = F.normalize(visual_proj, dim=-1)
    m_all_norm = F.normalize(map_proj, dim=-1)
    
    # logits: (B, T, N)
    logits = torch.matmul(v_norm, m_all_norm.T) / 0.07 
    
    loss_retrieval = F.cross_entropy(logits.view(-1, logits.size(-1)), target_node_indices.view(-1))
    
    total_loss = loss_pose + (sdf_lambda * loss_geo) + (0.1 * loss_smooth) + (1.0 * loss_retrieval)
    
    return total_loss, {
        "pose": loss_pose, "geo": loss_geo, 
        "smooth": loss_smooth, "retrieval": loss_retrieval
    }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Training on {device}")
    
    static_dir = os.path.join(args.data_root, "static")
    sequences_dir = os.path.join(args.data_root, "sequences")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    try:
        sdf_vol = torch.load(os.path.join(static_dir, "lung_sdf.pt"), map_location=device)
        grid_trans = torch.from_numpy(np.load(os.path.join(static_dir, "grid_transform.npy"))).float().to(device)
        graph_data = np.load(os.path.join(static_dir, "deep_lung_graph.npz"))
    except FileNotFoundError:
        print(f"[ERROR] Static data missing.")
        return

    node_pos = torch.from_numpy(graph_data['node_pos']).float().to(device)
    edge_index = torch.from_numpy(graph_data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(graph_data['edge_attr']).float().to(device)

    # Stats
    norm_center = get_norm_center_tensor(device)
    norm_scale = NORM_SCALE

    full_dataset = DeepLungDataset(data_root=sequences_dir, t_frames=args.t_frames, mode='train')
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = DeepLungST(
        t_frames=args.t_frames, 
        sdf_volume_tensor=sdf_vol, 
        grid_transform_matrix=grid_trans,
        mode=args.model_mode,
        norm_center=norm_center,
        norm_scale=norm_scale
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Curriculum: Physics OFF for first 5 epochs, then ramp up
        use_physics = (epoch >= 5)
        current_sdf_lambda = min(args.sdf_lambda, args.sdf_lambda * ((epoch - 5) / 5.0)) if use_physics else 0.0
        
        model.train()
        run_loss = 0.0
        
        pbar = tqdm(train_loader)
        for batch in pbar:
            video = batch['video'].to(device)
            gt_pos = batch['gt_pos'].to(device)
            initial_pose = gt_pos[:, 0, :] 
            
            optimizer.zero_grad()
            
            pred_particles, violations, vis_proj, map_proj = model(
                video, node_pos, edge_index, edge_attr, initial_pose,
                physics_on=use_physics
            )
            
            loss, comps = multi_hypothesis_loss(
                pred_particles, gt_pos, violations, 
                vis_proj, map_proj, node_pos, initial_pose, 
                sdf_lambda=current_sdf_lambda,
                norm_center=norm_center, norm_scale=norm_scale
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            run_loss += loss.item()
            pbar.set_postfix({'Pos': f"{comps['pose']:.3f}", 'Ret': f"{comps['retrieval']:.3f}"})

        avg_train = run_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                gt_pos = batch['gt_pos'].to(device)
                initial_pose = gt_pos[:, 0, :]
                
                # Val always uses Physics ON for true metric
                pred, viol, vis, map_p = model(video, node_pos, edge_index, edge_attr, initial_pose, physics_on=True)
                l, _ = multi_hypothesis_loss(
                    pred, gt_pos, viol, vis, map_p, node_pos, initial_pose,
                    sdf_lambda=args.sdf_lambda, norm_center=norm_center, norm_scale=norm_scale
                )
                val_loss += l.item()
        
        avg_val = val_loss / len(val_loader)
        print(f"  >> Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        scheduler.step(avg_val)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'm', 'l'])
    parser.add_argument('--t_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sdf_lambda', type=float, default=10.0) 
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    train(args)