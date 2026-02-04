
"""
General 3D Trajectory Visualization Script
Plots Ground Truth vs Estimated Trajectory (e.g. ORB-SLAM3).
Aligns Estimate to GT using Sim3 (Scale+Rigid).

Usage:
    python make_3d_plot.py <input_folder>

    <input_folder> can be:
    1. A single sequence folder containing 'stamped_groundtruth.txt' and 'stamped_traj_estimate*.txt'.
    2. A parent directory containing multiple such sequence folders.

Output:
    Creates a 'visualization' subfolder in each processed directory with .png and .html plots.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

# ==========================================
# Alignment Utils
# ==========================================

def load_tum_with_stamps(path):
    """Load TUM format (timestamp x y z ...)."""
    try:
        data = np.loadtxt(path)
        if data.ndim == 1: data = data.reshape(1, -1)
        # Ensure at least 4 columns (t, x, y, z)
        if data.shape[1] < 4:
            print(f"[WARN] {os.path.basename(path)} has {data.shape[1]} cols. Adding dummy indices as time.")
            stamps = np.arange(len(data)).reshape(-1, 1)
            data = np.hstack([stamps, data[:, :3]])
        return data[:, :4]
    except Exception as e:
        print(f"[ERR] Error loading {path}: {e}")
        return np.array([])

def compute_umeyama(source, target, estimate_scale=False):
    """Compute Sim3 (R, t, s) or Rigid (R, t, 1.0) to align source to target."""
    n = source.shape[0]
    mu_s = np.mean(source, axis=0)
    mu_t = np.mean(target, axis=0)
    
    src_c = source - mu_s
    tgt_c = target - mu_t
    
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    
    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T
        
    scale = 1.0
    if estimate_scale:
        var_s = np.sum(np.square(src_c)) / n
        scale = 1.0 / var_s * np.trace(np.diag(S))
    
    t_vec = mu_t - scale * (R_mat @ mu_s)
    
    return R_mat, t_vec, scale

def apply_sim3(points, R, t, s):
    return (s * points @ R.T) + t

def associate_timestamps(src_stamps, tgt_stamps, max_diff=0.2):
    """
    Associate based on RELATIVE timestamps (t - t0).
    Assumes trajectories start at approximately the same event.
    """
    matches = []
    
    # Normalize time to start at 0
    src_s = src_stamps - src_stamps[0]
    tgt_s = tgt_stamps - tgt_stamps[0]
    
    # Sort just in case (usually sorted)
    src_idx = np.argsort(src_s)
    tgt_idx = np.argsort(tgt_s)
    src_s = src_s[src_idx]
    tgt_s = tgt_s[tgt_idx]
    
    tgt_ptr = 0
    n_tgt = len(tgt_s)
    
    for i, t_val in enumerate(src_s):
        # Sliding window
        while tgt_ptr < n_tgt and tgt_s[tgt_ptr] < t_val - max_diff:
            tgt_ptr += 1
            
        best_dist = max_diff
        best_match = -1
        
        curr = tgt_ptr
        while curr < n_tgt and tgt_s[curr] <= t_val + max_diff:
            dist = abs(tgt_s[curr] - t_val)
            if dist < best_dist:
                best_dist = dist
                best_match = curr
            curr += 1
            
        if best_match != -1:
            matches.append((src_idx[i], tgt_idx[best_match]))
            
    return matches

def align_trajectories(est_data, gt_data):
    """
    Align Est -> GT (Geometric Priority).
    1. Align Start (Translation).
    2. Approximate Rotation (Vector).
    3. Refine with Sim3 but force Scale=1.0 (since units appear consistent).
    """
    est_pts = est_data[:, 1:4]
    gt_pts = gt_data[:, 1:4]
    
    # A. Pin Start (Translation match)
    # This is critical if timestamps are offset or unreliable.
    t_shift = gt_pts[0] - est_pts[0]
    est_shifted = est_pts + t_shift
    
    # B. Try Timestamp matching with RELATIVE time
    # This guides the coarse alignment
    matches = associate_timestamps(est_data[:, 0], gt_data[:, 0], max_diff=0.5) # looser tolerance
    print(f"  [ALIGN] Time Matches (Relative): {len(matches)} / {len(est_data)}")
    
    # C. Compute Rotation
    # If we have matches, use them to solve rotation
    if len(matches) > 20:
        src_m = est_shifted[[m[0] for m in matches]]
        tgt_m = gt_pts[[m[1] for m in matches]]
        
        # Compute Rotation around centroid or start?
        # Let's align centroids of matched segments
        # Force Scale = 1.0 (Rigid)
        R_eg, t_eg, _ = compute_umeyama(src_m, tgt_m, estimate_scale=False)
        
        # Apply to original (unshifted) points
        # Sim3 normally: s * R * p + t
        # Here s=1. 
        est_aligned = est_pts @ R_eg.T + t_eg
    else:
        print("  [WARN] Fallback: Aligning Start Vectors")
        # Align initial direction vectors (first 50 points or so)
        n_init = min(50, len(est_pts), len(gt_pts))
        vec_est = est_shifted[n_init] - est_shifted[0]
        vec_gt  = gt_pts[n_init] - gt_pts[0]
        
        # Align vec_est to vec_gt -> R
        # (Simple 2-vector alignment)
        pts_a = np.array([est_shifted[0], est_shifted[0] + vec_est])
        pts_b = np.array([gt_pts[0], gt_pts[0] + vec_gt])
        R_init, _, _ = compute_umeyama(pts_a, pts_b, estimate_scale=False)
        
        # Apply shift + rot
        est_intermediate = (est_pts + t_shift) @ R_init.T
        
        # Refine with ICP?
        # Simple refinement: Centroid of whole trajectory? No, shapes might differ length.
        # Stick to start-pinned + vector
        est_aligned = (est_pts - est_pts[0]) @ R_init.T + gt_pts[0]

    return gt_pts, est_aligned

# ==========================================
# Plot
# ==========================================
def render_plot(plotter, gt_traj, est_traj, title):
    plotter.clear()
    
    # GT
    if len(gt_traj) > 1:
        plotter.add_mesh(pv.lines_from_points(gt_traj), color='blue', line_width=4, label='GT')
        plotter.add_point_labels([gt_traj[0]], ["Start"], point_size=8, text_color='blue', font_size=12)
    
    # Est
    if len(est_traj) > 1:
        plotter.add_mesh(pv.lines_from_points(est_traj), color='green', line_width=4, label='Pred')

    plotter.add_text(title, position='upper_left', font_size=12, color='black')
    plotter.add_legend(bcolor='white', border=True, size=(0.2, 0.15))

import math

# ... (imports)

# ... (alignment functions remain same)

def process_folder(folder, seq_id=1):
    """
    Process a single folder containing result txt files.
    Returns: (display_name, aligned_gt, aligned_est) if successful, else None.
    """
    folder_name = os.path.basename(folder)
    display_name = f"T{seq_id}"
    print(f"\nProcessing {folder_name} as {display_name}...")
    
    # Look for files
    input_gt_path = os.path.join(folder, "stamped_groundtruth.txt")
    est_path = os.path.join(folder, "stamped_traj_estimate0.txt")
    if not os.path.exists(est_path):
            est_path = os.path.join(folder, "stamped_traj_estimate.txt")
            
    if not os.path.exists(input_gt_path):
        print(f"  [SKIP] No GT found in {folder_name}")
        return None
    if not os.path.exists(est_path):
        print(f"  [SKIP] No Estimate found in {folder_name}")
        return None
        
    # Load
    input_gt = load_tum_with_stamps(input_gt_path)
    est = load_tum_with_stamps(est_path)
    
    if len(input_gt) == 0 or len(est) == 0:
        print("  [SKIP] Empty data")
        return None
        
    # Align
    aligned_gt, aligned_est = align_trajectories(est, input_gt)
    
    # Plot (Individual)
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.set_background('white')
    render_plot(plotter, aligned_gt, aligned_est, f"{display_name}")
    plotter.camera_position = 'xy'
    plotter.reset_camera()
    plotter.camera.zoom(1.2)
    
    # Save
    out_dir = os.path.join(folder, "visualization")
    os.makedirs(out_dir, exist_ok=True)
    
    # Use folder name for output filename
    safe_name = folder_name.replace("paper_combined_", "").replace("paper_", "")
    img_path = os.path.join(out_dir, f"{safe_name}.png")
    
    plotter.screenshot(img_path)
    print(f"  [SAVE] {img_path}")
    plotter.export_html(os.path.join(out_dir, f"{safe_name}.html"))
    plotter.close()
    
    return (display_name, aligned_gt, aligned_est)

def show_grid_view(results):
    """
    Show all results in a PyVista grid.
    results: list of (name, gt, est)
    """
    n_plots = len(results)
    if n_plots == 0:
        print("No results to visualize.")
        return

    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    
    print(f"\n[VIS] showing grid: {rows}x{cols} for {n_plots} plots")
    
    plotter = pv.Plotter(shape=(rows, cols), window_size=(1600, 1000), title="Batch Results")
    plotter.set_background('white')
    
    for i, (name, gt, est) in enumerate(results):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
        
        # Add Title
        plotter.add_text(name, font_size=14, color='black', position='upper_left')
        
        # Add Trajectories
        if len(gt) > 1:
            plotter.add_mesh(pv.lines_from_points(gt), color='blue', line_width=3, label='GT')
            # Mark start
            plotter.add_mesh(pv.PolyData(gt[0]), color='blue', point_size=8, render_points_as_spheres=True)
            
        if len(est) > 1:
            plotter.add_mesh(pv.lines_from_points(est), color='green', line_width=3, label='Pred')
            
        plotter.camera_position = 'xy'
        plotter.reset_camera()
        
        # Only add legend to first plot to save space/clutter? Or all?
        # User asked for legend in general. Let's add small legend to each.
        # Manual Legend using text to avoid API version issues with add_legend positioning
        # Moved further right (0.85) and separated vertically to avoid overlap
        plotter.add_text("GT", position=(0.70, 0.90), color='blue', font_size=10, viewport=True)
        plotter.add_text("Baseline", position=(0.70, 0.80), color='green', font_size=10, viewport=True)
        
    # plotter.link_views()  # Disabled to allow individual movement
    plotter.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Input directory (single sequence or parent batch folder)')
    parser.add_argument('--no-show', action='store_true', help='Do not show grid window at the end')
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input_path)
    
    if not os.path.exists(input_path):
        print(f"[ERR] Path not found: {input_path}")
        sys.exit(1)
        
    # Set PyVista theme
    pv.global_theme.allow_empty_mesh = True
    
    results = []
    
    # Check if single folder or batch
    if os.path.exists(os.path.join(input_path, "stamped_groundtruth.txt")):
        # Single mode
        res = process_folder(input_path, seq_id=1)
        if res: results.append(res)
    else:
        # Batch mode: Iterate subfolders
        subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]
        print(f"[INFO] Scanning {len(subfolders)} subfolders in {input_path}")
        
        # Sort to ensure T1, T2 order is consistent
        # Try to sort by number in folder name if possible, else alphabetical
        def natural_keys(text):
            import re
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
            
        sorted_subs = sorted(subfolders, key=lambda p: natural_keys(os.path.basename(p)))
        
        seq_counter = 1
        for sub in sorted_subs:
            if os.path.exists(os.path.join(sub, "stamped_groundtruth.txt")):
                res = process_folder(sub, seq_id=seq_counter)
                if res: 
                    results.append(res)
                    seq_counter += 1
                
    if not results:
        print("[WARN] No valid results processed.")
    elif not args.no_show:
        show_grid_view(results)

if __name__ == "__main__":
    main()
