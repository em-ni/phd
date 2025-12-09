"""
Diagnostic script to check if ground truth targets lie within the convex hull
of the model's downsampled map points (K candidates).

The model predicts positions as a weighted sum of K map points:
    pred = sum(prob_i * map_point_i)
    
This means predictions are CONSTRAINED to the convex hull of the K candidates.
If GT is outside this hull, MSE cannot reach 0 no matter the model capacity.
"""
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import ConvexHull, Delaunay
from ant_dataset import AntDataset
from constants import NORM_MAP_SCALE


def point_in_hull(point, hull_points, tolerance=1e-6):
    """
    Check if a point lies inside the convex hull of hull_points.
    Uses Delaunay triangulation approach.
    
    Returns:
        (bool, float): (is_inside, distance_to_hull_surface)
    """
    if len(hull_points) < 4:
        # Not enough points for 3D convex hull
        # Check distance to nearest point instead
        dists = np.linalg.norm(hull_points - point, axis=1)
        return False, np.min(dists)
    
    try:
        hull = Delaunay(hull_points)
        simplex = hull.find_simplex(point)
        is_inside = simplex >= 0
        
        if is_inside:
            return True, 0.0
        else:
            # Calculate distance to nearest hull point as approximation
            dists = np.linalg.norm(hull_points - point, axis=1)
            return False, np.min(dists)
    except Exception as e:
        # Degenerate hull (e.g., coplanar points)
        dists = np.linalg.norm(hull_points - point, axis=1)
        return False, np.min(dists)


def main():
    print("Checking if GT targets lie within convex hull of K map candidates...")
    
    # Load dataset (use sequences folder where seq_test is located for overfitting analysis)
    data_root = './dataset/sequences'
    dataset = AntDataset(data_root, mode='test')
    
    if len(dataset) == 0:
        print("[ERROR] No samples found!")
        return
    
    print(f"\n[INFO] Analyzing {len(dataset)} samples...")
    
    # Statistics
    total_frames = 0
    inside_count = 0
    outside_count = 0
    distances_when_outside = []
    min_distances_to_candidates = []  # Distance from GT to nearest candidate
    
    for idx in tqdm(range(len(dataset)), desc="Checking samples"):
        sample = dataset[idx]
        
        # Get model map points (downsampled K candidates) - shape (T, K, 3)
        map_points = sample['map_points'].numpy()  # Already normalized
        map_mask = sample['map_mask'].numpy()      # (T, K)
        
        # Get ground truth targets - shape (T, 6), we only use first 3 (position)
        actions = sample['actions'].numpy()[:, :3]  # (T, 3) - normalized positions
        
        T, K, _ = map_points.shape
        
        for t in range(T):
            total_frames += 1
            
            # Get valid map points for this frame
            valid_mask = map_mask[t]
            valid_points = map_points[t][valid_mask]  # (M, 3) where M <= K
            
            # Get GT target for this frame
            gt_target = actions[t]  # (3,)
            
            # Skip if too few points
            if len(valid_points) < 3:
                outside_count += 1
                continue
            
            # Check minimum distance to any candidate
            dists = np.linalg.norm(valid_points - gt_target, axis=1)
            min_dist = np.min(dists)
            min_distances_to_candidates.append(min_dist)
            
            # Check if GT is in convex hull
            is_inside, hull_dist = point_in_hull(gt_target, valid_points)
            
            if is_inside:
                inside_count += 1
            else:
                outside_count += 1
                distances_when_outside.append(hull_dist)
    
    # Report Results
    pct_inside = 100 * inside_count / total_frames if total_frames > 0 else 0
    pct_outside = 100 * outside_count / total_frames if total_frames > 0 else 0
    
    print(f"Total frames analyzed: {total_frames}")
    print(f"GT INSIDE convex hull:  {inside_count:5d} ({pct_inside:.1f}%)")
    print(f"GT OUTSIDE convex hull: {outside_count:5d} ({pct_outside:.1f}%)")
    
    # Theoretical MSE lower bound
    if min_distances_to_candidates:
        # Best case: model outputs exact nearest candidate
        best_case_mse = np.mean(np.array(min_distances_to_candidates) ** 2)
        print(f"  Best theoretical MSE: {best_case_mse:.6f}")

if __name__ == "__main__":
    main()
