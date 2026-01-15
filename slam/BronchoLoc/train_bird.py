"""
Training script for BIRD: Bronchial Intraoperative Route Discriminator

Key differences from train_ant.py:
1. Loads and FREEZES pretrained ANT model
2. Processes sequences IN ORDER (no shuffle) to maintain memory state
3. Trains only BIRD parameters
4. Resets memory state at trajectory boundaries
"""
import os
import argparse
import torch
import shutil
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from ant import ActionPredictor, MODEL_CONFIGS
from bird import BIRD, BIRD_CONFIGS, create_bird
from titans_pytorch.neural_memory import mem_state_detach
from ant_dataset import AntDataset
from constants import NORM_MAP_SCALE, DEFAULT_MAX_MAP_POINTS, load_window_config, MAP_POINT_SPACING
from utils.utils import load_centerline_points, density_based_sample


# Global flag for atomic saving
_saving_in_progress = False


def get_checkpoint_name(args):
    """Generate checkpoint filename: bird_model_{mode}_{timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"bird_model_{args.model_mode}_{timestamp}"
    return name


def save_checkpoint(path, bird_model, optimizer, scheduler, epoch, name, best_val_loss=None):
    """Save checkpoint atomically."""
    global _saving_in_progress
    _saving_in_progress = True
    
    temp_path = path + ".tmp"
    try:
        ckpt = {
            'epoch': epoch,
            'bird_state_dict': bird_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'checkpoint_name': name
        }
        if best_val_loss is not None:
            ckpt['best_val_loss'] = best_val_loss
        torch.save(ckpt, temp_path)
        shutil.move(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
    finally:
        _saving_in_progress = False


def downsample_centerline(centerline_pts, min_distance=MAP_POINT_SPACING):
    """
    Downsample centerline for efficient cross-attention.
    Uses density-based sampling for uniform spacing, same as ANT model.
    The number of points is determined naturally by the spacing.
    """
    sampled, _ = density_based_sample(centerline_pts, min_distance=min_distance, start_idx=0)
    return sampled


def get_sequence_indices(dataset):
    """
    Group dataset indices by sequence for sequential processing.
    Returns dict: seq_name -> list of indices in order.
    """
    seq_to_indices = {}
    for idx in range(len(dataset)):
        seq_name = dataset.samples[idx][0].split(os.sep)[-2]  # Extract seq name from path
        if seq_name not in seq_to_indices:
            seq_to_indices[seq_name] = []
        seq_to_indices[seq_name].append((idx, dataset.samples[idx][2]))  # (idx, start_frame)
    
    # Sort each sequence by start frame
    for seq_name in seq_to_indices:
        seq_to_indices[seq_name].sort(key=lambda x: x[1])
        seq_to_indices[seq_name] = [x[0] for x in seq_to_indices[seq_name]]
    
    return seq_to_indices


def train(args):
    """Main training function for BIRD."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting BIRD Training on {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load window config
    window_size, frame_skip = load_window_config()
    print(f"[INFO] Using window_size={window_size}, frame_skip={frame_skip}")
    
    # ===========================================================================
    # Load and Freeze ANT Model
    # ===========================================================================
    if not args.ant_checkpoint:
        raise ValueError("--ant_checkpoint is required. Provide path to trained ANT model.")
    
    print(f"[INFO] Loading ANT from: {args.ant_checkpoint}")
    ant_checkpoint = torch.load(args.ant_checkpoint, map_location=device)
    
    # Handle both old and new checkpoint formats
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
    
    # FREEZE ANT
    ant_model.eval()
    for param in ant_model.parameters():
        param.requires_grad = False
    
    ant_config = MODEL_CONFIGS[args.model_mode]
    print(f"[INFO] ANT frozen. Visual dim: {ant_config['embed_dim']}")
    
    # ===========================================================================
    # Load and Downsample Centerline (BEFORE BIRD init so we know the size)
    # ===========================================================================
    centerline_path = os.path.join(args.data_root, "static", "centerline.npz")
    centerline_pts = load_centerline_points(centerline_path)
    if centerline_pts is None:
        raise FileNotFoundError(f"Centerline not found at {centerline_path}")
    
    print(f"[INFO] Full centerline: {len(centerline_pts)} points")
    
    # Downsample using density-based sampling (same as ANT model)
    centerline_ds = downsample_centerline(centerline_pts)
    print(f"[INFO] Downsampled centerline: {len(centerline_ds)} points (spacing={MAP_POINT_SPACING}mm)")
    
    # ===========================================================================
    # Initialize BIRD
    # ===========================================================================
    bird_model = create_bird(
        ant_mode=args.model_mode,
        num_centerline_pts=len(centerline_ds)
        # window_size auto-loaded from window_config
    ).to(device)
    
    bird_config = BIRD_CONFIGS[args.model_mode]
    trainable_params = sum(p.numel() for p in bird_model.parameters() if p.requires_grad)
    print(f"[INFO] BIRD initialized. Memory dim: {bird_config['memory_dim']}")
    print(f"[INFO] BIRD trainable parameters: {trainable_params:,}")
    
    # Normalize and encode centerline (detach to prevent graph issues)
    centerline_normalized = torch.tensor(centerline_ds / NORM_MAP_SCALE, dtype=torch.float32).to(device)
    with torch.no_grad():
        centerline_encoded = bird_model.encode_centerline(centerline_normalized)
    
    # ===========================================================================
    # Load Dataset
    # ===========================================================================
    dataset = AntDataset(
        data_root=os.path.join(args.data_root, "sequences"),
        mode='train',
        max_map_points=DEFAULT_MAX_MAP_POINTS,
        img_size=args.img_size
    )
    
    # Get sequence structure for sequential processing
    seq_to_indices = get_sequence_indices(dataset)
    all_seq_names = list(seq_to_indices.keys())
    
    # Filter to simulation-only if requested
    if args.sim_only:
        all_seq_names = [s for s in all_seq_names if s.startswith('seq_b')]
        print(f"[INFO] --sim_only: Filtering to {len(all_seq_names)} simulation sequences")
    
    # =========================================================================
    # Train/Val Split (80/20 at sequence level)
    # =========================================================================
    import random
    random.shuffle(all_seq_names)
    n_train = max(1, int(0.8 * len(all_seq_names)))
    train_seq_names = all_seq_names[:n_train]
    val_seq_names = all_seq_names[n_train:]
    
    print(f"[INFO] Sequence split: {len(train_seq_names)} train, {len(val_seq_names)} val")
    print(f"[INFO] Train seqs: {train_seq_names}")
    print(f"[INFO] Val seqs: {val_seq_names}")
    
    # =========================================================================
    # Optimizer and Scheduler (ANT-style with warmup + cosine decay)
    # =========================================================================
    optimizer = optim.AdamW(bird_model.parameters(), lr=args.lr, weight_decay=5e-3)  # Increased regularization
    
    # Warmup + Cosine Decay: linear warmup then smooth cosine decay
    # Same strategy as train_ant.py for consistency
    warmup_epochs = max(1, int(args.epochs * 0.05))  # 5% warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    print(f"[INFO] Using Warmup ({warmup_epochs} epochs) + Cosine Decay scheduler")
    
    # =========================================================================
    # Resume from checkpoint if specified
    # =========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        bird_model.load_state_dict(ckpt['bird_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
        print(f"[INFO] Resuming from epoch {start_epoch}")
    
    # =========================================================================
    # TensorBoard
    # =========================================================================
    if args.resume:
        # Use existing checkpoint name when resuming
        checkpoint_name = os.path.splitext(os.path.basename(args.resume))[0]
    else:
        checkpoint_name = get_checkpoint_name(args)
    log_dir = os.path.join(args.checkpoint_dir, "logs", checkpoint_name)
    writer = SummaryWriter(log_dir)
    print(f"[INFO] TensorBoard logs at: {log_dir}")
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{checkpoint_name}.pth")
    print(f"[INFO] Checkpoint will be saved as: {checkpoint_path}")
    
    # ===========================================================================
    # Training Loop
    # ===========================================================================
    criterion = torch.nn.MSELoss()
    global_step = 0
    
    try:
        for epoch in range(start_epoch, args.epochs):
            # =========================================================
            # TRAINING PHASE
            # =========================================================
            bird_model.train()
            train_loss = 0.0
            train_windows = 0
            
            # Shuffle sequence order each epoch (but keep windows in order within seq)
            train_seq_shuffled = train_seq_names.copy()
            np.random.shuffle(train_seq_shuffled)
            
            pbar = tqdm(train_seq_shuffled, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for seq_name in pbar:
                indices = seq_to_indices[seq_name]
                
                if len(indices) == 0:
                    continue
                
                # =========================================================
                # PHASE 1: Forward pass on all windows to collect surprises
                # =========================================================
                mem_state = bird_model.reset_memory()
                window_data = []  # Store (idx, loss_tensor, surprise, batch_data) tuples
                
                for window_idx, idx in enumerate(indices):
                    batch = dataset[idx]
                    
                    # Move to device
                    video = batch['video'].unsqueeze(0).to(device)
                    map_points = batch['map_points'].unsqueeze(0).to(device)
                    map_mask = batch['map_mask'].unsqueeze(0).to(device)
                    target = batch['actions'][:, :3].unsqueeze(0).to(device)
                    
                    # Get frozen ANT predictions
                    with torch.no_grad():
                        ant_pos, delta_pos, delta_quat, visual_tokens, _ = ant_model(
                            video, map_points, map_mask, return_features=True
                        )
                    
                    # Detach mem_state for proper streaming
                    if mem_state is not None:
                        mem_state = mem_state_detach(mem_state)
                    
                    # BIRD forward (get surprise)
                    p_refined, mem_state, _, avg_surprise = bird_model(
                        ant_pos, delta_pos, delta_quat, visual_tokens,
                        centerline_encoded, centerline_normalized,
                        mem_state=mem_state
                    )
                    
                    # Compute loss (but don't backprop yet)
                    loss = criterion(p_refined, target)
                    
                    # Store window data with surprise
                    window_data.append({
                        'loss': loss,
                        'surprise': avg_surprise.item(),
                        'window_idx': window_idx
                    })
                
                # =========================================================
                # PHASE 2: Select top-N most surprising and backprop
                # =========================================================
                if len(window_data) == 0:
                    continue
                
                # Sort by surprise (descending) and take top-N
                window_data.sort(key=lambda x: x['surprise'], reverse=True)
                top_k = min(args.top_k_surprise, len(window_data))
                selected_windows = window_data[:top_k]
                
                # Sum losses of selected windows and backprop
                selected_loss = sum(w['loss'] for w in selected_windows) / top_k
                
                optimizer.zero_grad()
                selected_loss.backward()
                torch.nn.utils.clip_grad_norm_(bird_model.parameters(), 1.0)
                optimizer.step()
                
                # Logging
                train_loss += selected_loss.item() * top_k
                train_windows += top_k
                global_step += 1
                
                avg_surprise_selected = sum(w['surprise'] for w in selected_windows) / top_k
                pbar.set_postfix(MSE=f"{selected_loss.item():.4f}", Sur=f"{avg_surprise_selected:.2f}")
                
                # Log periodically
                if global_step % 50 == 0:
                    writer.add_scalar('train/loss', selected_loss.item(), global_step)
                    writer.add_scalar('train/avg_surprise', avg_surprise_selected, global_step)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            
            avg_train_loss = train_loss / train_windows if train_windows > 0 else 0
            scheduler.step()
            
            # =========================================================
            # VALIDATION PHASE
            # =========================================================
            bird_model.eval()
            val_loss = 0.0
            val_windows = 0
            
            with torch.no_grad():
                for seq_name in val_seq_names:
                    indices = seq_to_indices[seq_name]
                    if len(indices) == 0:
                        continue
                    
                    mem_state = bird_model.reset_memory()
                    
                    for idx in indices:
                        batch = dataset[idx]
                        video = batch['video'].unsqueeze(0).to(device)
                        map_points = batch['map_points'].unsqueeze(0).to(device)
                        map_mask = batch['map_mask'].unsqueeze(0).to(device)
                        target = batch['actions'][:, :3].unsqueeze(0).to(device)
                        
                        ant_pos, delta_pos, delta_quat, visual_tokens, _ = ant_model(
                            video, map_points, map_mask, return_features=True
                        )
                        
                        if mem_state is not None:
                            mem_state = mem_state_detach(mem_state)
                        
                        p_refined, mem_state, _, _ = bird_model(
                            ant_pos, delta_pos, delta_quat, visual_tokens,
                            centerline_encoded, centerline_normalized,
                            mem_state=mem_state
                        )
                        
                        loss = criterion(p_refined, target)
                        val_loss += loss.item()
                        val_windows += 1
            
            avg_val_loss = val_loss / val_windows if val_windows > 0 else 0
            
            # Print both train and val MSE
            print(f"  >> Train MSE: {avg_train_loss:.6f}")
            print(f"  >> Val MSE:   {avg_val_loss:.6f}")
            writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
            writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
            
            # Save best model based on VAL loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(checkpoint_path, bird_model, optimizer, scheduler, epoch, checkpoint_name, best_val_loss)
                print(f"  >> Saved best model (Val MSE: {best_val_loss:.6f})")
    
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).")
        save_checkpoint(checkpoint_path, bird_model, optimizer, scheduler, epoch, checkpoint_name, best_val_loss)
        print(f"[INFO] Checkpoint saved to {checkpoint_path}")
    
    print(f"[INFO] Best Val MSE: {best_val_loss:.6f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BIRD global memory module")
    
    # Required
    parser.add_argument('--ant_checkpoint', type=str, required=True,
                        help="Path to trained ANT checkpoint")
    
    # Model
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'b', 'm', 'l'],
                        help="Model size (must match ANT checkpoint)")
    
    # Data
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help="Root directory containing sequences/ and static/")
    parser.add_argument('--img_size', type=int, default=128,
                        help="Input image size (must match ANT training)")
    parser.add_argument('--sim_only', action='store_true',
                        help="Train only on simulation sequences (seq_b*)")
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--top_k_surprise', type=int, default=4,
                        help="Train on top-K most surprising windows per sequence")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    
    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save checkpoints")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to BIRD checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)
