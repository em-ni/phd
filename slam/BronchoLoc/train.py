import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from deep_lung_st import ActionPredictor
from deep_lung_dataset import DeepLungDataset 
from constants import NORM_MAP_SCALE 

def train(args):
    """
    Main training function.
    Sets up the dataset, model, optimizer, and executes the training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Training on {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize full dataset (window_size/frame_skip loaded from config inside dataset)
    full_dataset = DeepLungDataset(
        data_root=os.path.join(args.data_root, "sequences"), 
        mode='train', 
        img_size=args.img_size
    )
    
    # Get window_size from dataset (loaded from config)
    window_size = full_dataset.window_size
    print(f"[INFO] Using window_size={window_size}, frame_skip={full_dataset.frame_skip}")
    
    # --- DEBUGGING / OVERFITTING MODES ---
    if args.overfit:
        print("[INFO] Overfitting mode: Training on 'seq_test' only.")
        # Filter dataset to only include samples from "seq_test".
        # This is useful to verify if the model can memorize a single sequence.
        indices = [i for i, (vp, _, _) in enumerate(full_dataset.samples) if "seq_test" in vp]
        
        if not indices:
            print("[ERROR] 'seq_test' not found in dataset! Please ensure 'dataset/sequences/seq_test' exists.")
            return
            
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        
        # Train on this subset, Val on the SAME subset to check overfitting capability
        train_ds = full_dataset
        val_ds = full_dataset
    else:
        # Standard Split: 80% Train, 20% Val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    if args.debug_one:
        print("[INFO] DEBUG ONE mode: Training on a SINGLE batch (first 16 samples).")
        # Extreme debug mode: Overfit on just one batch of data.
        # Use shuffle=False to ensure the same batch is used in test.py
        debug_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        first_batch = next(iter(debug_loader))
        
        # Create a dummy loader that yields this batch forever
        class InfiniteLoader:
            def __init__(self, batch): self.batch = batch
            def __iter__(self): return self
            def __next__(self): return self.batch
            def __len__(self): return 100 # arbitrary length for tqdm visualization
            
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Learning rate scheduler: Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Regression Loss Function
    # Use reduction='none' to apply per-sample weighting later
    criterion = torch.nn.MSELoss(reduction='none')

    # --- TRAINING LOOP ---
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            model.train()
            run_loss = 0.0
            
            pbar = tqdm(train_loader)
            for batch in pbar:
                video = batch['video'].to(device) # (B, T, C, H, W)
                gt_deltas = batch['actions'].to(device) # (B, T, 6)
                map_points = batch['map_points'].to(device) # (B, T, K, 3)
                map_mask = batch['map_mask'].to(device) # (B, T, K)
                
                optimizer.zero_grad()
                
                # Forward Pass
                # (B, T, 3) - Model now only predicts translation (via graph selection)
                pred_trans = model(video, map_points=map_points, map_mask=map_mask)
                
                # Get Ground Truth Translation
                gt_trans = gt_deltas[:, :, :3]
                
                # Normalize GT to match map point scale [-1, 1]
                # This is critical because the model's output is derived from normalized map points.
                gt_trans_norm = gt_trans / NORM_MAP_SCALE
                
                # Calculate Loss
                raw_loss = criterion(pred_trans, gt_trans_norm)
                
                # Weighted Loss (Motion Incentive)
                # Penalize errors more on frames with larger movement.
                # This combats the "stop-and-stare" problem where models predict zero motion.
                motion_mag = torch.norm(gt_trans_norm, dim=2)
                weights = 1.0 + args.motion_weight * motion_mag
                weights = weights.unsqueeze(2)
                loss = (raw_loss * weights).mean()
                
                # Backward Pass
                loss.backward()
                optimizer.step()
                
                run_loss += loss.item()
                
                pbar.set_postfix({'MSE': f"{loss.item():.6f}"})

            avg_train_loss = run_loss / len(train_loader)
            
            # --- VALIDATION LOOP ---
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    video = batch['video'].to(device)
                    gt_deltas = batch['actions'].to(device)
                    map_points = batch['map_points'].to(device)
                    map_mask = batch['map_mask'].to(device)
                    
                    pred_trans = model(video, map_points=map_points, map_mask=map_mask)
                    
                    gt_trans = gt_deltas[:, :, :3]
                    gt_trans_norm = gt_trans / NORM_MAP_SCALE
                    
                    raw_loss = criterion(pred_trans, gt_trans_norm)
                    motion_mag = torch.norm(gt_trans_norm, dim=2)
                    weights = 1.0 + args.motion_weight * motion_mag
                    weights = weights.unsqueeze(2)
                    loss = (raw_loss * weights).mean()
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  >> Train MSE: {avg_train_loss:.6f}")
            print(f"  >> Val MSE:   {avg_val_loss:.6f}")
            
            # Step LR Scheduler
            scheduler.step(avg_val_loss)
            
            # Save Checkpoint (Best so far)
            # Note: Currently saves every epoch, could be improved to save only on improvement.
            if args.debug_one:
                ckpt_name = "debug_one_model.pth"
            elif args.overfit:
                ckpt_name = "overfit_model.pth"
            else:
                ckpt_name = "best_model.pth"
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, ckpt_name))
                
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).")
        # Save model on interrupt
        if args.debug_one:
            save_path = os.path.join(args.checkpoint_dir, "debug_one_model.pth")
        elif args.overfit:
            save_path = os.path.join(args.checkpoint_dir, "overfit_model.pth")
        else:
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_mode', type=str, default='s', choices=['s', 'b', 'm', 'l'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overfit', action='store_true', help="Overfit on a small subset")
    parser.add_argument('--debug_one', action='store_true', help="Overfit on a SINGLE batch to verify learning capability")
    parser.add_argument('--motion_weight', type=float, default=0.0, help="Weight for motion incentive (0.0 = standard MSE)")
    parser.add_argument('--img_size', type=int, default=128, help="Image resolution (default: 128)")
    args = parser.parse_args()
    train(args)