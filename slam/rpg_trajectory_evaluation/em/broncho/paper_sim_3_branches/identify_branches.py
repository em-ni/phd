
import os
import sys
import numpy as np
from scipy.spatial import KDTree
import glob
import itertools
from tqdm import tqdm

def load_tum(path):
    """
    Load TUM format file.
    Timestamp tx ty tz qx qy qz qw
    Returns Nx3 array of positions (tx, ty, tz).
    """
    try:
        data = np.loadtxt(path, delimiter=' ')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 1:4]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.array([])

def umeyama_alignment(source, target):
    """
    source: Nx3, target: Nx3
    Returns R, t, s such that target approx s * R * source + t
    """
    n = source.shape[0]
    if n < 3: return np.eye(3), np.zeros(3), 1.0
    
    mu_s = np.mean(source, axis=0)
    mu_t = np.mean(target, axis=0)
    
    s_prime = source - mu_s
    t_prime = target - mu_t
    
    sigma_st = np.dot(s_prime.T, t_prime) / n
    U, D, V_T = np.linalg.svd(sigma_st)
    V = V_T.T
    
    S = np.eye(3)
    if np.linalg.det(sigma_st) < 0:
        S[2, 2] = -1
    
    R = np.dot(V, np.dot(S, U.T))
    
    var_s = np.var(source, axis=0).sum()
    if var_s < 1e-6: c = 1.0
    else: c = (1.0 / var_s) * np.trace(np.dot(np.diag(D), S))
    
    t = mu_t - c * np.dot(R, mu_s)
    return R, t, c

def compute_rmse(source, target_tree):
    """
    Compute RMSE of source points against a KDTree of target points.
    """
    dists, _ = target_tree.query(source, k=1)
    rmse = np.sqrt(np.mean(dists**2))
    return rmse

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    centerlines_dir = os.path.abspath(os.path.join(base_dir, "../../../../BronchoSim/data/mesh/lungs/sim/centerlines"))
    
    # 1. Load all branches
    print(f"Loading branches from {centerlines_dir}...")
    branches = {}
    branch_files = glob.glob(os.path.join(centerlines_dir, "b*_tum.txt"))
    branch_keys = []
    
    for bf in branch_files:
        basename = os.path.basename(bf)
        branch_name = basename.replace("_tum.txt", "")
        points = load_tum(bf)
        if len(points) > 0:
            branches[branch_name] = points
            branch_keys.append(branch_name)
    
    branch_keys.sort()
    num_branches = len(branch_keys)
    print(f"Loaded {num_branches} branches.")

    # Pre-compute combinations to verify count
    combinations = list(itertools.combinations(branch_keys, 3))
    print(f"Total combinations to test per trajectory: {len(combinations)}")

    # 2. Iterate through results folders
    search_pattern = os.path.join(base_dir, "**", "stamped_groundtruth.txt")
    gt_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(gt_files)} groundtruth files.")
    
    results = []

    for gt_file in gt_files:
        folder_name = os.path.basename(os.path.dirname(gt_file))
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(gt_file)))
        display_name = os.path.join(parent_folder, folder_name)
        
        gt_points = load_tum(gt_file)
        if len(gt_points) < 10:
            print(f"Skipping {display_name}: Not enough GT points.")
            continue
            
        print(f"\nProcessing {display_name}...")
        
        best_rmse = float('inf')
        best_combo = None
        best_aligned_gt = None
        
        # Optimize: Pre-build KDTrees for combinations? Too memory heavy potentially (1330 trees).
        # Better: Build points for combination on fly.
        # Even better: The alignment loop is the bottleneck.
        # We can optimize the ICP.
        
        # HEURISTIC OPTIMIZATION:
        # Instead of full ICP on ALL 1330 combos from scratch (which might take long),
        # we could filtering?
        # BUT user asked for "take all possible combinations... and for each... test it".
        # So we must be rigorous.
        
        # We can drastically speed this up by building KDTrees once if we group?
        # No, triplet structures are unique.
        
        # Let's just run it. 1330 combos * 10 iterations ICP is ~13k alignments per file.
        # If each alignment takes 0.01s -> 130s per file. 20 files -> 45 mins.
        # That's a bit slow but acceptable for a "brute force" script if running in background.
        # To speed up: reduce ICP iterations or use fewer points for alignment (downsample GT).
        
        gt_points_downsampled = gt_points
        if len(gt_points) > 200:
            # Downsample for speed during search
            indices = np.linspace(0, len(gt_points)-1, 200, dtype=int)
            gt_points_downsampled = gt_points[indices]
            
        
        for combo in tqdm(combinations, desc=f"Searching {folder_name}", leave=False):
            # 1. Construct triplet cloud
            triplet_points = np.vstack([branches[b] for b in combo])
            triplet_tree = KDTree(triplet_points)
            
            # 2. ICP Alignment
            aligned = gt_points_downsampled.copy()
            prev_loss = float('inf')
            
            for _ in range(10): # Max 10 iters for search
                dists, idxs = triplet_tree.query(aligned, k=1)
                targets = triplet_points[idxs]
                R, t, s = umeyama_alignment(aligned, targets)
                aligned = s * np.dot(aligned, R.T) + t
                loss = np.mean(dists)
                if abs(prev_loss - loss) < 1e-4: break
                prev_loss = loss
            
            # 3. Compute Final RMSE on this combo
            rmse = prev_loss # Approx RMSE (actually MAD or mean dist)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_combo = combo
                best_aligned_gt = aligned # Save for vis (approx)
        
        # --- Final Refinement of Best Combo ---
        # Run high-quality ICP on the full GT points with the best combo
        print(f"  Best candidate: {best_combo} (RMSE approx: {best_rmse:.4f})")
        print("  Refining final alignment...")
        
        final_triplet_points = np.vstack([branches[b] for b in best_combo])
        final_tree = KDTree(final_triplet_points)
        
        final_aligned_gt = gt_points.copy()
        for _ in range(50): # High precision
            dists, idxs = final_tree.query(final_aligned_gt, k=1)
            targets = final_triplet_points[idxs]
            R, t, s = umeyama_alignment(final_aligned_gt, targets)
            final_aligned_gt = s * np.dot(final_aligned_gt, R.T) + t
            loss = np.mean(dists) # Mean dist
            if abs(prev_loss - loss) < 1e-7: break
            prev_loss = loss
            
        # Final RMSE calculation
        final_dists, _ = final_tree.query(final_aligned_gt)
        final_rmse = np.sqrt(np.mean(final_dists**2))
        
        print(f"[{display_name}] Final Selection: {', '.join(best_combo)} | RMSE: {final_rmse:.5f}")
        results.append((display_name, best_combo, final_rmse))
        
        # Visualization (PyVista HTML)
        try:
            import pyvista as pv
            pl = pv.Plotter(off_screen=True)
            
            # Context: All branches
            for b_name, pts in branches.items():
                if len(pts) > 1:
                    line = pv.lines_from_points(pts)
                    pl.add_mesh(line, color='lightgray', opacity=0.1, line_width=1)
            
            # Highlight Best Combo
            colors = ['red', 'green', 'blue']
            for i, b_name in enumerate(best_combo):
                pts = branches[b_name]
                if len(pts) > 1:
                    line = pv.lines_from_points(pts)
                    tube = line.tube(radius=0.5)
                    pl.add_mesh(tube, color=colors[i], label=f"{b_name}")
            
            # Aligned GT
            if len(final_aligned_gt) > 1:
                gt_line = pv.lines_from_points(final_aligned_gt)
                gt_tube = gt_line.tube(radius=0.7)
                pl.add_mesh(gt_tube, color='black', label='Ground Truth')
            
            pl.add_legend()
            pl.show_grid()
            pl.camera_position = 'iso'
            
            html_path = os.path.join(os.path.dirname(gt_file), "vis_branches.html")
            pl.export_html(html_path)
            print(f"  Saved visualization to {html_path}")
            pl.close()
            
        except Exception as e:
            print(f"  Visualization Error: {e}")

    # Save summary
    with open("branch_identification_results.txt", "w") as f:
        for name, combo, rmse in results:
            f.write(f"[{name}] {', '.join(combo)} (RMSE: {rmse:.5f})\n")

if __name__ == "__main__":
    main()
