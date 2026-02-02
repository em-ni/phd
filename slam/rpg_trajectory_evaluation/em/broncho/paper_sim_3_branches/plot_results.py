
import os
import sys
import numpy as np
import glob
import re
import pyvista as pv

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

def align_to_branches(gt_points, branch_points, iterations=50):
    from scipy.spatial import KDTree
    tree = KDTree(branch_points)
    aligned = gt_points.copy()
    prev_loss = float('inf')
    
    for _ in range(iterations):
        dists, idxs = tree.query(aligned, k=1)
        targets = branch_points[idxs]
        R, t, s = umeyama_alignment(aligned, targets)
        aligned = s * np.dot(aligned, R.T) + t
        loss = np.mean(dists)
        if abs(prev_loss - loss) < 1e-7: break
        prev_loss = loss
    return aligned

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    centerlines_dir = os.path.abspath(os.path.join(base_dir, "../../../../BronchoSim/data/mesh/lungs/sim/centerlines"))
    results_file = os.path.join(base_dir, "branch_identification_results.txt")
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        sys.exit(1)

    # 1. Load branches
    print(f"Loading branches from {centerlines_dir}...")
    branches = {}
    branch_files = glob.glob(os.path.join(centerlines_dir, "b*_tum.txt"))
    for bf in branch_files:
        basename = os.path.basename(bf)
        branch_name = basename.replace("_tum.txt", "")
        points = load_tum(bf)
        if len(points) > 0:
            branches[branch_name] = points
    print(f"Loaded {len(branches)} branches.")

    # 2. Parse results file
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} results...")
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Format: [rgbd\paper_rgbd_b1] b5, b6, b8 (RMSE: 0.04214)
        match = re.search(r'\[(.*?)\] (.*?) \(RMSE:', line)
        if not match:
            print(f"Skipping malformed line: {line}")
            continue
            
        rel_path = match.group(1)
        combo_str = match.group(2)
        selected_branches = [b.strip() for b in combo_str.split(',')]
        
        # Resolve GT path
        # rel_path typically "rgbd\paper_rgbd_b1" (windows)
        # We need to ensure it works with current OS
        parts = rel_path.replace('\\', '/').split('/')
        gt_folder_path = os.path.join(base_dir, *parts)
        gt_file = os.path.join(gt_folder_path, "stamped_groundtruth.txt")
        
        if not os.path.exists(gt_file):
            print(f"GT file not found: {gt_file}")
            continue
            
        gt_points = load_tum(gt_file)
        if len(gt_points) < 2:
            print(f"Not enough points in {gt_file}")
            continue
            
        print(f"Plotting for {rel_path} with {selected_branches}")
        
        # 3. Align GT to selected branches
        combo_points = np.vstack([branches[b] for b in selected_branches if b in branches])
        aligned_gt = align_to_branches(gt_points, combo_points)
        
        # 4. Plot
        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter(off_screen=True)
        
        # A. All 21 branches (Gray, Context)
        for b_name, pts in branches.items():
            if len(pts) > 1:
                try:
                    line = pv.lines_from_points(pts)
                    pl.add_mesh(line, color='lightgray', opacity=0.3, line_width=2)
                except Exception:
                    pass
        
        # B. Selected 3 branches (Colored, Highlighted)
        colors = ['red', 'green', 'blue']
        for i, b_name in enumerate(selected_branches):
            if b_name not in branches: continue
            pts = branches[b_name]
            if len(pts) > 1:
                try:
                    line = pv.lines_from_points(pts)
                    tube = line.tube(radius=0.5)
                    pl.add_mesh(tube, color=colors[i % len(colors)], label=f"{b_name}")
                except Exception:
                    pass
        
        # C. Ground Truth (Black, Bold)
        if len(aligned_gt) > 1:
            try:
                line = pv.lines_from_points(aligned_gt)
                tube = line.tube(radius=0.7)
                pl.add_mesh(tube, color='black', label='Ground Truth')
            except Exception:
                pass
            
        pl.add_legend()
        pl.show_grid()
        pl.camera_position = 'iso'
        
        output_file = os.path.join(gt_folder_path, "vis_final.html")
        pl.export_html(output_file)
        print(f"  Saved {output_file}")
        pl.close()

if __name__ == "__main__":
    main()
