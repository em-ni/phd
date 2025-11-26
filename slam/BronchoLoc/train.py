import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the architecture and dataset class
from deep_lung_st import DeepLungST
from deep_lung_dataset import DeepLungDataset 

# ==============================================================================
# NEW LOSS FUNCTION (Best-of-K + Contrastive)
# ==============================================================================
def multi_hypothesis_loss(
    pred_traj_particles,    # (B, K, T, 3)
    gt_traj,                # (B, T, 3)
    violations,             # (B, K, T, 1)
    visual_proj,            # (B, D)
    map_proj,               # (N_nodes, D)
    map_node_positions,     # (N_nodes, 3)
    initial_gt_pos,         # (B, 3) - to find correct map node
    sdf_lambda=10.0
):
    """
    Computes loss for multi-hypothesis tracking and map retrieval.
    """
    B, K, T, _ = pred_traj_particles.shape
    
    # 1. Best-of-K Trajectory Loss (Min-of-N)
    # Allows particles to explore; only penalize the best one against GT.
    # gt_traj expanded: (B, 1, T, 3)
    gt_expanded = gt_traj.unsqueeze(1)
    
    # MSE for each particle: (B, K)
    # Sum over Time and Coord dimensions
    mse_per_particle = torch.sum((pred_traj_particles - gt_expanded)**2, dim=(2, 3))
    
    # Find best particle (min error)
    best_loss_vals, best_indices = torch.min(mse_per_particle, dim=1) # (B,)
    loss_pose = best_loss_vals.mean()
    
    # 2. Geometry Violation Loss (Penalize ALL particles to stay inside)
    # We want all hypotheses to be physically valid, even if wrong branch.
    loss_geo = violations.mean()
    
    # 3. Smoothness Loss (on best particle only to save compute)
    # Gather best trajectories: (B, T, 3)
    batch_indices = torch.arange(B, device=pred_traj_particles.device)
    best_trajs = pred_traj_particles[batch_indices, best_indices] 
    
    vel = best_trajs[:, 1:] - best_trajs[:, :-1]
    accel = vel[:, 1:] - vel[:, :-1]
    loss_smooth = torch.mean(accel**2)
    
    # 4. Improvement 3: Map Retrieval (Contrastive/InfoNCE Loss)
    # Which Map Node is the "Ground Truth" for this sequence?
    # We approximate this by finding the closest graph node to the GT start position.
    
    # Distances: (B, 1, 3) - (1, N, 3) -> (B, N)
    dists = torch.norm(initial_gt_pos.unsqueeze(1) - map_node_positions.unsqueeze(0), dim=2)
    target_node_indices = torch.argmin(dists, dim=1) # (B,) indices of closest nodes
    
    # Get the embedding of the target nodes: (B, D)
    target_map_embs = map_proj[target_node_indices]
    
    # Normalize embeddings
    v_norm = F.normalize(visual_proj, dim=1)
    m_norm = F.normalize(target_map_embs, dim=1)
    
    # Cosine Similarity -> maximize this
    similarity = torch.sum(v_norm * m_norm, dim=1)
    loss_retrieval = 1.0 - similarity.mean() # Simple Cosine Loss
    
    # Total Loss
    total_loss = loss_pose + (sdf_lambda * loss_geo) + (0.1 * loss_smooth) + (0.5 * loss_retrieval)
    
    return total_loss, {
        "pose": loss_pose, 
        "geo": loss_geo, 
        "smooth": loss_smooth,
        "retrieval": loss_retrieval
    }

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Multi-Hypothesis Training on {device}")
    
    static_dir = os.path.join(args.data_root, "static")
    sequences_dir = os.path.join(args.data_root, "sequences")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load Static Data
    print("[INFO] Loading Static Map Data...")
    try:
        sdf_vol = torch.load(os.path.join(static_dir, "lung_sdf.pt"), map_location=device)
        grid_trans = torch.from_numpy(np.load(os.path.join(static_dir, "grid_transform.npy"))).float().to(device)
        graph_data = np.load(os.path.join(static_dir, "deep_lung_graph.npz"))
    except FileNotFoundError:
        print(f"[ERROR] Static data not found in {static_dir}.")
        return

    node_pos = torch.from_numpy(graph_data['node_pos']).float().to(device)
    edge_index = torch.from_numpy(graph_data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(graph_data['edge_attr']).float().to(device)

    # Dataset
    full_dataset = DeepLungDataset(data_root=sequences_dir, t_frames=args.t_frames, mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = DeepLungST(
        t_frames=args.t_frames, 
        sdf_volume_tensor=sdf_vol, 
        grid_transform_matrix=grid_trans,
        mode=args.model_mode
    ).to(device)

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model has {total_params / 1e6:.2f} million trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print(f"[INFO] Model initialized with {model.K} particles per sample.")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            video = batch['video'].to(device)
            gt_pos = batch['gt_pos'].to(device)
            initial_pose = gt_pos[:, 0, :] 
            
            optimizer.zero_grad()
            
            # Forward (Multi-Hypothesis)
            # Returns: (B, K, T, 3), (B, K, T, 1), (B, D), (N, D)
            pred_particles, violations, vis_proj, map_proj = model(
                video, node_pos, edge_index, edge_attr, initial_pose
            )
            
            # Loss Calculation (Best-of-K + Retrieval)
            loss, components = multi_hypothesis_loss(
                pred_particles, gt_pos, violations, 
                vis_proj, map_proj, node_pos, initial_pose, 
                sdf_lambda=args.sdf_lambda
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'L_Pose': f"{components['pose']:.2f}", 'L_Ret': f"{components['retrieval']:.2f}"})

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                gt_pos = batch['gt_pos'].to(device)
                initial_pose = gt_pos[:, 0, :]
                
                pred_particles, violations, vis_proj, map_proj = model(
                    video, node_pos, edge_index, edge_attr, initial_pose
                )
                
                loss, _ = multi_hypothesis_loss(
                    pred_particles, gt_pos, violations, 
                    vis_proj, map_proj, node_pos, initial_pose, 
                    sdf_lambda=args.sdf_lambda
                )
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"  >> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))

    print("[INFO] Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'm', 'l'])
    parser.add_argument('--t_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sdf_lambda', type=float, default=10.0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    train(args)