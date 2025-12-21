import os
import argparse
import torch
import shutil
import signal
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from ant import ActionPredictor, MODEL_CONFIGS
from ant_dataset import AntDataset 
from constants import NORM_MAP_SCALE, DEFAULT_MAX_MAP_POINTS

# Global flag to prevent saving during an ongoing save
_saving_in_progress = False

    
def get_checkpoint_name(args, is_debug=False):
    """Generate checkpoint filename: ant_model_{mode}_{timestamp}."""
    prefix = "ant"
    if is_debug:
        prefix = "ant_debug"
    elif args.overfit:
        prefix = "ant_overfit"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{prefix}_model_{args.model_mode}_{timestamp}"
    return name


def save_checkpoint(path, model, optimizer, scheduler, epoch, name):
    """Save checkpoint atomically - writes to temp file first, then renames.
    
    This ensures the checkpoint is either fully written or not at all,
    preventing corruption from Ctrl+C or crashes during save.
    """
    global _saving_in_progress
    _saving_in_progress = True
    
    temp_path = path + ".tmp"
    try:
        # Save to temporary file first
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'checkpoint_name': name
        }, temp_path)
        
        # Atomic rename (on same filesystem, this is atomic on most OSes)
        shutil.move(temp_path, path)
        
    except Exception as e:
        # Clean up temp file if save failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
    finally:
        _saving_in_progress = False


def train(args):
    """
    Main training function.
    Sets up the dataset, model, optimizer, and executes the training loop.
    Supports resuming from a checkpoint with --resume.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Training on {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize full dataset
    full_dataset = AntDataset(
        data_root=os.path.join(args.data_root, "sequences"), 
        mode='train', 
        img_size=args.img_size
    )
    
    window_size = full_dataset.window_size
    print(f"[INFO] Using window_size={window_size}, frame_skip={full_dataset.frame_skip}")
    
    # --- DEBUGGING / OVERFITTING / FINETUNING MODES ---
    if args.finetune_phantom:
        # Finetune on phantom sequences only
        print("[INFO] Finetune-phantom mode: Training on 'seq_phantom_*' sequences only.")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if "seq_phantom" in vp]
        
        if not indices:
            print("[ERROR] No 'seq_phantom_*' sequences found in dataset!")
            return
        
        # Use same sequence-level split logic for phantom data
        from collections import defaultdict
        seq_to_indices = defaultdict(list)
        for idx in indices:
            vid_path = full_dataset.samples[idx][0]
            seq_name = os.path.basename(os.path.dirname(vid_path))
            seq_to_indices[seq_name].append(idx)
        
        all_seqs = list(seq_to_indices.keys())
        n_train_seqs = max(1, int(0.8 * len(all_seqs)))
        
        import random
        random.shuffle(all_seqs)
        train_seqs = all_seqs[:n_train_seqs]
        val_seqs = all_seqs[n_train_seqs:]
        
        train_indices = []
        val_indices = []
        for seq in train_seqs:
            train_indices.extend(seq_to_indices[seq])
        for seq in val_seqs:
            val_indices.extend(seq_to_indices[seq])
        
        print(f"[INFO] Phantom sequences: {len(train_seqs)} train, {len(val_seqs)} val")
        print(f"[INFO] Train sequences: {train_seqs}")
        print(f"[INFO] Val sequences: {val_seqs}")
        print(f"[INFO] Windows: {len(train_indices)} train, {len(val_indices)} val")
        
        train_ds = torch.utils.data.Subset(full_dataset, train_indices)
        val_ds = torch.utils.data.Subset(full_dataset, val_indices)
        
    elif args.overfit:
        print("[INFO] Overfitting mode: Training on 'seq_test' only.")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if "seq_test" in vp]
        
        if not indices:
            print("[ERROR] 'seq_test' not found in dataset!")
            return
            
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        train_ds = full_dataset
        val_ds = full_dataset
    else:
        # --- SEQUENCE-LEVEL SPLIT (PREVENTS DATA LEAKAGE) ---
        # Group sample indices by sequence path
        from collections import defaultdict
        seq_to_indices = defaultdict(list)
        for i, (vid_path, _, _) in enumerate(full_dataset.samples):
            # Extract sequence name from video path (e.g., "seq_b1_1234")
            seq_name = os.path.basename(os.path.dirname(vid_path))
            seq_to_indices[seq_name].append(i)
        
        # Split sequences (not windows) into train/val
        all_seqs = list(seq_to_indices.keys())
        n_train_seqs = max(1, int(0.8 * len(all_seqs)))
        
        # Shuffle sequences for randomness
        import random
        random.shuffle(all_seqs)
        train_seqs = all_seqs[:n_train_seqs]
        val_seqs = all_seqs[n_train_seqs:]
        
        # Collect indices for each split
        train_indices = []
        val_indices = []
        for seq in train_seqs:
            train_indices.extend(seq_to_indices[seq])
        for seq in val_seqs:
            val_indices.extend(seq_to_indices[seq])
        
        print(f"[INFO] Sequence-level split: {len(train_seqs)} train seqs, {len(val_seqs)} val seqs")
        print(f"[INFO] Train sequences: {train_seqs}")
        print(f"[INFO] Val sequences: {val_seqs}")
        print(f"[INFO] Windows: {len(train_indices)} train, {len(val_indices)} val")
        
        train_ds = torch.utils.data.Subset(full_dataset, train_indices)
        val_ds = torch.utils.data.Subset(full_dataset, val_indices)
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    if args.debug_one:
        print("[INFO] DEBUG ONE mode: Training on a SINGLE batch.")
        debug_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        first_batch = next(iter(debug_loader))
        
        class InfiniteLoader:
            def __init__(self, batch): self.batch = batch
            def __iter__(self): return self
            def __next__(self): return self.batch
            def __len__(self): return 100
            
        train_loader = InfiniteLoader(first_batch)
        val_loader = InfiniteLoader(first_batch)
    else:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Initialize Model
    model = ActionPredictor(
        window_size=window_size, 
        mode=args.model_mode,
        img_size=args.img_size
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {total_params:,}")
    
    # Optimization Setup
    # Use model's suggested LR if user didn't specify one
    if args.lr is None:
        lr = MODEL_CONFIGS[args.model_mode]['suggested_lr']
        print(f"[INFO] Using model's suggested LR: {lr}")
    else:
        lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)  # Increased regularization
    
    # Scheduler Setup
    if args.scheduler == 'cosine':
        """
                LR
        │    ___
        │   /   \___          ← 1e-4 (peak after warmup)
        │  /        \___
        │ /             \_    ← 1e-6 (end)
        └────────────────────→ epochs
        5%      95%
        warmup  cosine decay
        """
        # Warmup + Cosine Decay: linear warmup then smooth cosine decay
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
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, min_lr=1e-6)
    
    criterion = torch.nn.MSELoss(reduction='none')
    
    # Resume state
    start_epoch = 0
    checkpoint_name = None
    
    # --- RESUME FROM CHECKPOINT ---
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[INFO] Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if args.reset_lr is not None:
                # Reset optimizer LR to specified value
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.reset_lr
                print(f"[INFO] Learning rate reset to {args.reset_lr}")
            else:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            start_epoch = checkpoint['epoch'] + 1
            checkpoint_name = checkpoint.get('checkpoint_name', None)
            print(f"[INFO] Resuming from epoch {start_epoch}")
        else:
            print(f"[WARNING] Checkpoint not found: {args.resume}, starting fresh")
    
    # Generate checkpoint name if not resuming
    if checkpoint_name is None:
        checkpoint_name = get_checkpoint_name(args, is_debug=args.debug_one)
    
    # For finetune_phantom, append _finetuned to original checkpoint name
    if args.finetune_phantom and args.resume:
        # Use original checkpoint name with _finetuned suffix
        checkpoint_name = checkpoint_name + "_finetuned"
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{checkpoint_name}.pth")
    print(f"[INFO] Checkpoint will be saved as: {checkpoint_path}")
    
    # --- TENSORBOARD SETUP ---
    log_dir = os.path.join(args.checkpoint_dir, "logs", checkpoint_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard logs at: {log_dir}")
    print(f"[INFO] Run: tensorboard --logdir={log_dir}")
    
    # --- EARLY STOPPING SETUP ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f"{checkpoint_name}_best.pth")

    # --- TRAINING LOOP ---
    epoch = start_epoch  # Initialize for KeyboardInterrupt handler
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            model.train()
            run_loss = 0.0
            
            pbar = tqdm(train_loader)
            for batch in pbar:
                video = batch['video'].to(device)
                gt_deltas = batch['actions'].to(device)
                map_points = batch['map_points'].to(device)
                map_mask = batch['map_mask'].to(device)
                target_indices = batch['target_indices'].to(device)  # (B, T)
                
                optimizer.zero_grad()
                
                # Get predictions and attention probs for CE loss
                pred_trans, _, attn_probs = model(video, map_points=map_points, map_mask=map_mask, return_features=True)
                # NOTE: gt_deltas are already normalized in the dataset (see ant_dataset.py line 293)
                gt_trans = gt_deltas[:, :, :3]  # Already normalized
                
                # MSE Loss on position predictions
                mse_loss = criterion(pred_trans, gt_trans).mean()
                
                # Cross-Entropy Loss on attention weights
                # This directly supervises attention to select the correct point
                # attn_probs: (B, T, K), target_indices: (B, T)
                B, T, K = attn_probs.shape
                attn_flat = attn_probs.view(B * T, K)  # (B*T, K)
                target_flat = target_indices.view(B * T)  # (B*T,)
                # Use NLL loss with log probs (cross_entropy already applies log, so use nll_loss)
                log_probs = torch.log(attn_flat.clamp(min=1e-10))  # Clamp to prevent log(0)
                ce_loss = torch.nn.functional.nll_loss(log_probs, target_flat)
                
                # Combined loss: MSE + CE
                loss = mse_loss + args.ce_weight * ce_loss
                
                if args.motion_weight > 0:
                    pred_variance = pred_trans.var(dim=1).mean()
                    motion_term = -args.motion_weight * pred_variance
                    loss = loss + motion_term
                
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                run_loss += loss.item()
                pbar.set_postfix({"MSE": f"{mse_loss.item():.6f}", "CE": f"{ce_loss.item():.4f}"})
            
            avg_train_loss = run_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    video = batch['video'].to(device)
                    gt_deltas = batch['actions'].to(device)
                    map_points = batch['map_points'].to(device)
                    map_mask = batch['map_mask'].to(device)
                    
                    pred_trans = model(video, map_points=map_points, map_mask=map_mask)
                    # NOTE: gt_deltas are already normalized in the dataset
                    gt_trans = gt_deltas[:, :, :3]  # Already normalized
                    
                    loss = criterion(pred_trans, gt_trans).mean()
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  >> Train MSE: {avg_train_loss:.6f}")
            print(f"  >> Val MSE:   {avg_val_loss:.6f}")
            
            # Step scheduler (different schedulers have different APIs)
            if args.scheduler == 'cosine':
                scheduler.step()
            else:
                scheduler.step(avg_val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  >> Current LR: {current_lr:.2e}")
            
            # --- TENSORBOARD LOGGING ---
            writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
            writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # --- EARLY STOPPING CHECK ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, epoch, checkpoint_name)
                print(f"  >> New best model saved! (Val MSE: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  >> No improvement ({patience_counter}/{args.early_stop_patience})")
            
            # Save regular checkpoint
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, checkpoint_name)
            
            # Check early stopping
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"\n[INFO] Early stopping triggered after {patience_counter} epochs without improvement.")
                print(f"[INFO] Best model saved at: {best_checkpoint_path}")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, checkpoint_name)
        print(f"[INFO] Checkpoint saved to {checkpoint_path}")
    finally:
        writer.close()
        print(f"[INFO] Best validation MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_mode', type=str, default='s', choices=['xs', 's', 'b', 'm', 'l'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=None,
                        help="Learning rate (default: use model's suggested_lr)")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overfit', action='store_true', help="Overfit on seq_test only")
    parser.add_argument('--finetune_phantom', action='store_true', 
                        help="Finetune on phantom sequences (seq_phantom_*) only")
    parser.add_argument('--debug_one', action='store_true', help="Overfit on a SINGLE batch")
    parser.add_argument('--motion_weight', type=float, default=0.0, help="Weight for motion incentive")
    parser.add_argument('--img_size', type=int, default=128, help="Image resolution")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--reset_lr', type=float, default=1e-6, help="Reset learning rate to this value when resuming")
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'],
                        help="LR scheduler: 'plateau' (default) or 'cosine' for warmup + cosine decay")
    parser.add_argument('--early_stop_patience', type=int, default=100,
                        help="Stop if val loss doesn't improve for N epochs (0=disabled)")
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help="Weight for cross-entropy loss on attention (0=disabled, 1.0=equal to MSE)")
    args = parser.parse_args()
    train(args)