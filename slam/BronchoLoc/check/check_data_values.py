"""
Quick diagnostic to check the actual values in the dataset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ant_dataset import AntDataset

if __name__ == "__main__":
    # Load dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(base_dir, 'dataset', 'sequences')
    dataset = AntDataset(data_root, mode='test')
    
    print(f"\nAnalyzing {len(dataset)} samples...")
    
    # Sample a few items
    for idx in [0, len(dataset)//2, len(dataset)-1]:
        sample = dataset[idx]
        
        map_points = sample['map_points'].numpy()  # (T, K, 3)
        map_mask = sample['map_mask'].numpy()      # (T, K)
        actions = sample['actions'].numpy()         # (T, 6)
        
        # Get first frame's data
        valid_pts = map_points[0][map_mask[0]]  # Valid map points at T=0
        target = actions[0, :3]                  # Target position at T=0
        
        print(f"\n{'='*60}")
        print(f"Sample {idx}:")
        print(f"  Map points shape: {map_points.shape}")
        print(f"  Valid points at T=0: {valid_pts.shape[0]}")
        print(f"  Map points range: [{valid_pts.min():.4f}, {valid_pts.max():.4f}]")
        print(f"  Map points mean: {valid_pts.mean(axis=0)}")
        print(f"  Target at T=0: {target}")
        print(f"  Target norm: {np.linalg.norm(target):.4f}")
        
        # Check if target is one of the map points
        dists = np.linalg.norm(valid_pts - target, axis=1)
        min_dist = dists.min()
        print(f"  Min dist from target to any map point: {min_dist:.6f}")
        
        # Expected: min_dist should be 0.0 since target IS one of the map points
        if min_dist > 0.001:
            print(f"  ⚠️  WARNING: Target is NOT matching any map point!")
        else:
            print(f"  ✓ Target matches a map point (as expected)")
    
    # Compute baseline MSE (if model outputs mean of valid map points)
    print(f"\n{'='*60}")
    print("Computing BASELINE MSE (uniform attention = mean of map points)...")
    total_mse = 0.0
    total_frames = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        map_points = sample['map_points'].numpy()  # (T, K, 3)
        map_mask = sample['map_mask'].numpy()      # (T, K)
        actions = sample['actions'].numpy()[:, :3] # (T, 3)
        
        for t in range(map_points.shape[0]):
            valid_pts = map_points[t][map_mask[t]]
            if len(valid_pts) > 0:
                mean_pred = valid_pts.mean(axis=0)
                target = actions[t]
                mse = np.sum((mean_pred - target) ** 2)
                total_mse += mse
                total_frames += 1
    
    avg_baseline_mse = total_mse / total_frames if total_frames > 0 else 0
    print(f"  Baseline MSE (uniform attention): {avg_baseline_mse:.6f}")
    print(f"  This is the MSE if model outputs equal weights on all K points")
