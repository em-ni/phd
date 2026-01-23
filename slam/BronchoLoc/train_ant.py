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
    
    # Determine frame_skip override for phantom finetuning
    dataset_kwargs = dict(
        data_root=os.path.join(args.data_root, "sequences"),
        mode='train',
        img_size=args.img_size
    )
    
    if args.finetune_phantom:
        # Load phantom_frame_skip from config
        import json
        config_path = os.path.join(args.data_root, "..", "window_config.json")
        if not os.path.exists(config_path):
            config_path = "window_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        phantom_frame_skip = config.get('phantom_frame_skip', config.get('frame_skip', 60))
        dataset_kwargs['frame_skip'] = phantom_frame_skip
        print(f"[INFO] Using phantom_frame_skip={phantom_frame_skip}")
    
    # Initialize full dataset
    full_dataset = AntDataset(**dataset_kwargs)
    
    window_size = full_dataset.window_size
    print(f"[INFO] Using window_size={window_size}, frame_skip={full_dataset.frame_skip}")
    
    # --- DEBUGGING / OVERFITTING / FINETUNING MODES ---
    if args.finetune_phantom:
        # Finetune on phantom sequences only with k-fold CV
        print("[INFO] Finetune-phantom mode: Training on 'seq_phantom_*' sequences only.")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if "seq_phantom" in vp]
        
        if not indices:
            print("[ERROR] No 'seq_phantom_*' sequences found in dataset!")
            return
        
        # Group windows by sequence
        from collections import defaultdict
        seq_to_indices = defaultdict(list)
        for idx in indices:
            vid_path = full_dataset.samples[idx][0]
            seq_name = os.path.basename(os.path.dirname(vid_path))
            seq_to_indices[seq_name].append(idx)
        
        all_seqs = sorted(list(seq_to_indices.keys()))
        n_seqs = len(all_seqs)
        
        if args.k_folds > 1 and n_seqs >= args.k_folds:
            # K-fold CV: store folds for rotation during training
            print(f"[INFO] Using {args.k_folds}-fold CV across {n_seqs} phantom sequences")
            
            # Create fold assignments
            fold_size = n_seqs // args.k_folds
            folds = []
            for i in range(args.k_folds):
                start_idx = i * fold_size
                if i == args.k_folds - 1:
                    # Last fold gets remaining sequences
                    fold_seqs = all_seqs[start_idx:]
                else:
                    fold_seqs = all_seqs[start_idx:start_idx + fold_size]
                fold_indices = []
                for seq in fold_seqs:
                    fold_indices.extend(seq_to_indices[seq])
                folds.append((fold_seqs, fold_indices))
                print(f"[INFO] Fold {i+1}: {fold_seqs} ({len(fold_indices)} windows)")
            
            # Store folds for use in training loop
            args._phantom_folds = folds
            args._phantom_all_indices = indices
            args._phantom_seq_to_indices = seq_to_indices
            
            # Initial fold 0 as val, rest as train
            val_fold_idx = 0
            val_seqs, val_indices = folds[val_fold_idx]
            train_indices = []
            train_seqs = []
            for i, (seqs, idxs) in enumerate(folds):
                if i != val_fold_idx:
                    train_indices.extend(idxs)
                    train_seqs.extend(seqs)
            
            print(f"[INFO] Starting with Fold 0 as validation")
            print(f"[INFO] Train sequences: {train_seqs}")
            print(f"[INFO] Val sequences: {val_seqs}")
            print(f"[INFO] Windows: {len(train_indices)} train, {len(val_indices)} val")
            
            train_ds = torch.utils.data.Subset(full_dataset, train_indices)
            val_ds = torch.utils.data.Subset(full_dataset, val_indices)
        else:
            # Fallback to 80/20 split if not enough sequences for k-fold
            print(f"[INFO] Using 80/20 split (not enough sequences for {args.k_folds}-fold CV)")
            n_train_seqs = max(1, int(0.8 * n_seqs))
            
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
    
    elif args.sim_only:
        # Train on simulation sequences only (exclude phantom)
        print("[INFO] Sim-only mode: Training on simulation sequences (excluding 'seq_phantom_*').")
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if "seq_phantom" not in vp]
        
        if not indices:
            print("[ERROR] No simulation sequences found in dataset!")
            return
        
        # Use same sequence-level split logic
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
        
        print(f"[INFO] Simulation sequences: {len(train_seqs)} train, {len(val_seqs)} val")
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
    # Disable weight decay in debug/overfit modes to allow true overfitting
    weight_decay = 0.0 if (args.debug_one or args.overfit) else 5e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
            
            # K-fold rotation for phantom finetuning
            if hasattr(args, '_phantom_folds') and args.k_folds > 1:
                folds = args._phantom_folds
                val_fold_idx = epoch % args.k_folds
                val_seqs, val_indices = folds[val_fold_idx]
                train_indices = []
                for i, (seqs, idxs) in enumerate(folds):
                    if i != val_fold_idx:
                        train_indices.extend(idxs)
                
                # Recreate dataloaders with new fold split
                train_ds = torch.utils.data.Subset(full_dataset, train_indices)
                val_ds = torch.utils.data.Subset(full_dataset, val_indices)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                
                if epoch == start_epoch or (epoch % args.k_folds == 0):
                    print(f"[FOLD] Epoch {epoch+1}: Val fold {val_fold_idx+1}/{args.k_folds} = {val_seqs}")
            
            
            model.train()
            run_loss = 0.0
            
            pbar = tqdm(train_loader)
            for batch in pbar:
                video = batch['video'].to(device)
                gt_pos = batch['actions'].to(device)  # (B, T, 3) - already normalized
                map_points = batch['map_points'].to(device)
                map_mask = batch['map_mask'].to(device)
                target_indices = batch['target_indices'].to(device)  # (B, T)
                
                # Get delta targets (local frame deltas for VO training)
                gt_delta_quat = batch['delta_quats'].to(device)  # (B, T, 4) - relative to q0
                gt_delta_pos = batch['delta_positions'].to(device)  # (B, T, 3) - frame-to-frame
                
                optimizer.zero_grad()
                
                # Get predictions with 5-tuple return (delta_pos added for BIRD)
                pred_pos, pred_delta_pos, pred_delta_quat, _, attn_probs = model(video, map_points=map_points, map_mask=map_mask, return_features=True)
                
                # MSE Loss on position predictions (candidate selection)
                mse_loss = criterion(pred_pos, gt_pos).mean()
                
                # VO Position Loss (train VO head to predict frame-to-frame deltas)
                vo_pos_loss = criterion(pred_delta_pos, gt_delta_pos).mean()
                
                # Orientation Loss (quaternion MSE on delta quaternions)
                quat_loss = criterion(pred_delta_quat, gt_delta_quat).mean()
                
                # Cross-Entropy Loss on attention weights
                B, T, K = attn_probs.shape
                attn_flat = attn_probs.view(B * T, K)
                target_flat = target_indices.view(B * T)
                log_probs = torch.log(attn_flat.clamp(min=1e-10))
                ce_loss = torch.nn.functional.nll_loss(log_probs, target_flat)
                
                # Combined loss: MSE + VO + Quat + CE
                loss = mse_loss + args.vo_weight * (vo_pos_loss + quat_loss) + args.ce_weight * ce_loss
                
                if args.motion_weight > 0:
                    pred_variance = pred_pos.var(dim=1).mean()
                    motion_term = -args.motion_weight * pred_variance
                    loss = loss + motion_term
                
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                run_loss += loss.item()
                pbar.set_postfix({"MSE": f"{mse_loss.item():.4f}", "VO": f"{(vo_pos_loss + quat_loss).item():.4f}", "CE": f"{ce_loss.item():.4f}"})
            
            avg_train_loss = run_loss / len(train_loader)
            
            # Validation
            # NOTE: Keep model in train mode so soft selection is used (not argmax)
            # This allows validation MSE to show gradual improvement
            # Hard selection (argmax) is only used for final inference/testing
            model.train()  # Use soft selection for validation too
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    video = batch['video'].to(device)
                    gt_pos = batch['actions'].to(device)
                    map_points = batch['map_points'].to(device)
                    map_mask = batch['map_mask'].to(device)
                    
                    # Model returns (pred_pos, delta_quat)
                    pred_pos, _ = model(video, map_points=map_points, map_mask=map_mask)
                    
                    loss = criterion(pred_pos, gt_pos).mean()
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
    parser.add_argument('--k_folds', type=int, default=5,
                        help="Number of folds for k-fold CV during phantom finetuning (default: 5)")
    parser.add_argument('--sim_only', action='store_true', 
                        help="Train on simulation sequences only (exclude seq_phantom_*)")
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
                        help="Weight for cross-entropy loss on attention (0=disabled)")
    parser.add_argument('--vo_weight', type=float, default=0.1,
                        help="Weight for VO loss (position + quaternion, 0=disabled)")
    args = parser.parse_args()
    train(args)