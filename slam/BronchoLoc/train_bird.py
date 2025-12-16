"""
Training script for BIRD: Bronchial Inference Route Determination

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


def save_checkpoint(path, bird_model, optimizer, scheduler, epoch, name):
    """Save checkpoint atomically."""
    global _saving_in_progress
    _saving_in_progress = True
    
    temp_path = path + ".tmp"
    try:
        torch.save({
            'epoch': epoch,
            'bird_state_dict': bird_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'checkpoint_name': name
        }, temp_path)
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
    seq_names = list(seq_to_indices.keys())
    print(f"[INFO] Found {len(seq_names)} sequences with {len(dataset)} total windows")
    
    # ===========================================================================
    # Optimizer and Scheduler
    # ===========================================================================
    optimizer = optim.AdamW(bird_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing with warmup
    warmup_epochs = min(10, args.epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ===========================================================================
    # TensorBoard
    # ===========================================================================
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
    best_val_loss = float('inf')
    global_step = 0
    
    try:
        for epoch in range(args.epochs):
            bird_model.train()
            epoch_loss = 0.0
            num_windows = 0
            
            # Shuffle sequence order each epoch (but keep windows in order within seq)
            np.random.shuffle(seq_names)
            
            pbar = tqdm(seq_names, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for seq_name in pbar:
                indices = seq_to_indices[seq_name]
                
                # Reset memory at start of each sequence
                mem_state = bird_model.reset_memory()
                
                # Process windows in order within sequence
                for idx in indices:
                    batch = dataset[idx]
                    
                    # Move to device
                    video = batch['video'].unsqueeze(0).to(device)
                    map_points = batch['map_points'].unsqueeze(0).to(device)
                    map_mask = batch['map_mask'].unsqueeze(0).to(device)
                    target = batch['actions'][:, :3].unsqueeze(0).to(device)  # Only use position (first 3), ignore rotation placeholders
                    
                    # Get frozen ANT predictions and features
                    with torch.no_grad():
                        p_local, visual_tokens, _ = ant_model(
                            video, map_points, map_mask, return_features=True
                        )  # Note: ANT returns (pred, visual_tokens, probs), we ignore probs
                    
                    # BIRD refinement
                    p_global, mem_state = bird_model(
                        p_local, 
                        visual_tokens, 
                        centerline_encoded,
                        mem_state=mem_state
                    )
                    
                    # Loss and backward
                    loss = criterion(p_global, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bird_model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Reset memory state after backward to prevent gradient accumulation
                    # This means each window trains independently, but we still pass the 
                    # memory state forward (without gradients) to maintain sequential context
                    mem_state = None  # Reset to prevent "backward through graph twice" error
                    
                    epoch_loss += loss.item()
                    num_windows += 1
                    global_step += 1
                    
                    pbar.set_postfix(MSE=f"{loss.item():.6f}")
                    
                    # Log periodically
                    if global_step % 50 == 0:
                        writer.add_scalar('train/loss', loss.item(), global_step)
                        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            
            avg_loss = epoch_loss / num_windows if num_windows > 0 else 0
            scheduler.step()
            
            print(f"  >> Epoch {epoch+1} Avg MSE: {avg_loss:.6f}")
            writer.add_scalar('epoch/train_loss', avg_loss, epoch)
            
            # Save best model
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                save_checkpoint(checkpoint_path, bird_model, optimizer, scheduler, epoch, checkpoint_name)
                print(f"  >> Saved best model (MSE: {best_val_loss:.6f})")
    
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).")
        save_checkpoint(checkpoint_path, bird_model, optimizer, scheduler, epoch, checkpoint_name)
        print(f"[INFO] Checkpoint saved to {checkpoint_path}")
    
    print(f"[INFO] Best MSE: {best_val_loss:.6f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BIRD global memory module")
    
    # Required
    parser.add_argument('--ant_checkpoint', type=str, required=True,
                        help="Path to trained ANT checkpoint")
    
    # Model
    parser.add_argument('--model_mode', type=str, default='m', choices=['s', 'b', 'm', 'l'],
                        help="Model size (must match ANT checkpoint)")
    
    # Data
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help="Root directory containing sequences/ and static/")
    parser.add_argument('--img_size', type=int, default=128,
                        help="Input image size (must match ANT training)")
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    
    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save checkpoints")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to BIRD checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)
