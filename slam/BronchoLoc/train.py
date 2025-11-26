import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import the architecture and dataset class
from deep_lung_st import DeepLungST, deep_lung_loss
# Ensure BronchoSim.py or a separate utils file contains DeepLungDataset
from deep_lung_dataset import DeepLungDataset 

def train(args):
    # --- 1. SETUP & INITIALIZATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Training on {device}")
    print(f"[INFO] Model Mode: {args.model_mode} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    
    static_dir = os.path.join(args.data_root, "static")
    sequences_dir = os.path.join(args.data_root, "sequences")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- 2. LOAD STATIC GEOMETRIC DATA ---
    # This data is constant for the patient and shared across all batches
    print("[INFO] Loading Static Map Data...")
    try:
        # Load SDF Volume
        sdf_vol = torch.load(os.path.join(static_dir, "lung_sdf.pt"), map_location=device)
        # Load Transform Matrix
        grid_trans = torch.from_numpy(np.load(os.path.join(static_dir, "grid_transform.npy"))).float().to(device)
        # Load Graph Topology
        graph_data = np.load(os.path.join(static_dir, "deep_lung_graph.npz"))
    except FileNotFoundError:
        print(f"[ERROR] Static data not found in {static_dir}. Run vtk_cad_to_graph.py first.")
        return

    # Convert Graph Data to GPU Tensors
    node_pos = torch.from_numpy(graph_data['node_pos']).float().to(device)
    edge_index = torch.from_numpy(graph_data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(graph_data['edge_attr']).float().to(device)

    # --- 3. DATASET SETUP ---
    print("[INFO] Indexing Sequences...")
    full_dataset = DeepLungDataset(data_root=sequences_dir, t_frames=args.t_frames, mode='train')
    
    if len(full_dataset) == 0:
        print("[ERROR] Dataset is empty. Run BronchoSim data collection first.")
        return

    # Split Dataset (80% Train, 20% Val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    print(f"[INFO] Training Samples: {len(train_ds)} | Validation Samples: {len(val_ds)}")

    # --- 4. MODEL INITIALIZATION ---
    model = DeepLungST(
        t_frames=args.t_frames, 
        sdf_volume_tensor=sdf_vol, 
        grid_transform_matrix=grid_trans,
        mode=args.model_mode
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # --- 5. MAIN TRAINING LOOP ---
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # --- A. TRAIN STEP ---
        model.train()
        running_loss = 0.0
        geo_violations = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move data to GPU
            video = batch['video'].to(device)   # (B, T, 3, H, W)
            gt_pos = batch['gt_pos'].to(device) # (B, T, 3)
            
            # Initial Pose for LSTM (Teacher Forcing / Anchor)
            # In a real live scenario, we'd use the last known position.
            # Here we use the GT start position to help it learn.
            initial_pose = gt_pos[:, 0, :] 
            
            optimizer.zero_grad()
            
            # Forward Pass
            # Pass static graph data alongside batch video
            pred_traj, violations = model(video, node_pos, edge_index, edge_attr, initial_pose)
            
            # Loss Calculation
            loss, components = deep_lung_loss(pred_traj, gt_pos, violations, sdf_lambda=args.sdf_lambda)
            
            # Backward Pass & Optimization
            loss.backward()
            # Clip gradients to prevent LSTM explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Logging
            running_loss += loss.item()
            geo_violations += components['geo'].item()
            pbar.set_postfix({'Loss': loss.item(), 'Viol': components['geo'].item()})

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- B. VALIDATION STEP ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(device)
                gt_pos = batch['gt_pos'].to(device)
                initial_pose = gt_pos[:, 0, :]
                
                pred_traj, violations = model(video, node_pos, edge_index, edge_attr, initial_pose)
                loss, _ = deep_lung_loss(pred_traj, gt_pos, violations, sdf_lambda=args.sdf_lambda)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"  >> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Update Learning Rate
        scheduler.step(avg_val_loss)

        # --- C. CHECKPOINTING ---
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [*] New Best Model saved to {save_path}")

        # Periodic Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"ckpt_ep{epoch+1}.pth"))

    # --- 6. VISUALIZE TRAINING ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'DeepLungST Training History ({args.model_mode})')
    plt.legend()
    plt.savefig(os.path.join(args.checkpoint_dir, "training_curve.png"))
    print("[INFO] Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLungST")
    
    # Path Arguments
    parser.add_argument('--data_root', type=str, default='./dataset', 
                        help='Root containing static/ and sequences/ folders')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='Directory to save model weights and plots')
    
    # Model Configuration
    parser.add_argument('--model_mode', type=str, default='tiny', choices=['tiny', 'big'], 
                        help='Select model capacity')
    parser.add_argument('--t_frames', type=int, default=16, 
                        help='Number of frames in input sequence')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sdf_lambda', type=float, default=10.0, 
                        help='Weight for the wall collision penalty')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of worker threads for data loading')
    
    args = parser.parse_args()
    train(args)