"""
Evaluation script for BronchoLoc (ANT + BIRD) model.
Evaluates predictions on sequences in dataset/eval/trajectories/seq_bX_...
Applies Umeyama alignment and outputs trajectories in TUM format.

Copied directly from test.py structure.
"""
import os
import argparse
import time
import numpy as np
import torch
import cv2
import pyvista as pv
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Subset

from ant import ActionPredictor
from bird import BIRD, BIRD_CONFIGS
from ant_dataset import AntDataset
from constants import (
    MAP_QUERY_RADIUS, MAP_POINT_SPACING, DEFAULT_MAX_MAP_POINTS, 
    NORM_MAP_SCALE, load_window_config
)
from utils.utils import (
    load_centerline_points, filter_connected_component, 
    density_based_sample
)


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least-squares best-fit transform between corresponding 3D points.
    Implements Umeyama's method (1991).
    
    Args:
        x: (N, 3) source points (estimated trajectory)
        y: (N, 3) target points (ground truth trajectory)
        with_scale: if True, compute scale factor (Sim3), else only SE3
        
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scale factor (1.0 if with_scale=False)
    """
    assert x.shape == y.shape
    n, dim = x.shape
    
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    x_centered = x - mx
    y_centered = y - my
    
    sigma_x = np.sqrt((x_centered ** 2).sum() / n)
    cov = (y_centered.T @ x_centered) / n
    
    U, D, Vt = np.linalg.svd(cov)
    
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1
    
    rot = U @ S @ Vt
    
    if with_scale:
        scale = np.trace(np.diag(D) @ S) / (sigma_x ** 2)
    else:
        scale = 1.0
    
    trans = my - scale * rot @ mx
    
    return rot, trans, scale


def apply_transform(points, rot, trans, scale=1.0):
    """Apply rotation, translation, and scale to points."""
    return scale * (points @ rot.T) + trans


def save_trajectory_tum(filepath, timestamps, positions, quaternions):
    """
    Save trajectory in TUM format.
    Format: timestamp tx ty tz qx qy qz qw
    """
    with open(filepath, 'w') as f:
        for i in range(len(timestamps)):
            t = timestamps[i]
            p = positions[i]
            q = quaternions[i]
            # Assuming quaternions are [qw, qx, qy, qz], convert to TUM order [qx qy qz qw]
            if len(q) == 4:
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            else:
                qx, qy, qz, qw = 0, 0, 0, 1
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    print(f"[INFO] Saved TUM trajectory: {filepath}")


def compute_ate(gt_positions, pred_positions):
    """Compute Absolute Trajectory Error (ATE) after alignment."""
    errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
    return {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'min': np.min(errors)
    }


def render_evaluation_plot(plotter, lung_mesh, centerline_pts, 
                           gt_traj, pred_traj, aligned_pred_traj, 
                           seq_name, ate_stats):
    """Render 3D plot showing GT and predicted trajectories."""
    plotter.clear()
    
    if lung_mesh is not None:
        plotter.add_mesh(lung_mesh, color='wheat', opacity=0.1)
    
    if centerline_pts is not None:
        plotter.add_mesh(pv.PolyData(centerline_pts), color='black', opacity=0.15, 
                        point_size=2, render_points_as_spheres=True)
    
    if len(gt_traj) > 1:
        gt_line = pv.lines_from_points(gt_traj)
        plotter.add_mesh(gt_line, color='blue', line_width=4, label='GT Trajectory')
    
    if len(pred_traj) > 1:
        pred_line = pv.lines_from_points(pred_traj)
        plotter.add_mesh(pred_line, color='green', line_width=4, label='BIRD')
    
    # Add start label (no spheres)
    if len(gt_traj) > 0:
        plotter.add_point_labels([gt_traj[0]], ["Start"], point_size=8, 
                                text_color='blue', font_size=12)
    
    # Add info text with legend (since HTML export doesn't support legend widget)
    text = f"Sequence: {seq_name}\n"
    text += f"ATE RMSE: {ate_stats['rmse']:.2f} mm\n"
    text += f"ATE Mean: {ate_stats['mean']:.2f} mm\n"
    text += f"ATE Max: {ate_stats['max']:.2f} mm"
    plotter.add_text(text, position='upper_left', font_size=11, color='black')
    
    # Add legend text in lower right
    legend_text = "LEGEND\n"
    legend_text += "━━━━━━━━\n"
    legend_text += "Blue: GT\n"
    legend_text += "Green: BIRD"
    plotter.add_text(legend_text, position='lower_right', font_size=12, color='black')
    
    # Add legend widget for PNG (won't show in HTML)
    plotter.add_legend(bcolor='white', border=True, size=(0.2, 0.15))
    
    return plotter


def evaluate(args):
    """Main evaluation function - copied from test.py structure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Evaluation on {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset (window_size/frame_skip loaded from config inside dataset)
    # Use eval sequences directory
    eval_data_root = os.path.join(args.data_root, "eval", "trajectories")
    
    dataset_kwargs = dict(
        data_root=eval_data_root,
        mode='test',
        img_size=args.img_size,
        chain_mode=False,  # No chaining for eval
        augment=False
    )
    if args.frame_skip is not None:
        dataset_kwargs['frame_skip'] = args.frame_skip
        print(f"[INFO] Overriding frame_skip to {args.frame_skip}")
    
    full_dataset = AntDataset(**dataset_kwargs)
    
    window_size = full_dataset.window_size
    frame_skip = full_dataset.frame_skip
    print(f"[INFO] Using window_size={window_size}, frame_skip={frame_skip}")
    
    # Group samples by sequence
    seq_to_indices = {}
    for i, (vid_path, _, _) in enumerate(full_dataset.samples):
        seq_name = os.path.basename(os.path.dirname(vid_path))
        if seq_name not in seq_to_indices:
            seq_to_indices[seq_name] = []
        seq_to_indices[seq_name].append(i)
    
    print(f"[INFO] Found {len(seq_to_indices)} evaluation sequences")
    
    # Load ANT Model (copied from test.py)
    ant_model = ActionPredictor(
        window_size=window_size,
        mode=args.model_mode,
        img_size=args.img_size
    ).to(device)
    
    print(f"[INFO] Loading ANT checkpoint: {args.ant_checkpoint}")
    checkpoint = torch.load(args.ant_checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ant_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] ANT loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        ant_model.load_state_dict(checkpoint)
    ant_model.eval()
    
    # Load BIRD Model (copied from test.py)
    bird_model = None
    centerline_encoded = None
    centerline_normalized = None
    centerline_ds = None
    
    if args.bird_checkpoint:
        # Load centerline for BIRD
        centerline_path = os.path.join(args.data_root, "static", "centerline.npz")
        centerline_pts_full = load_centerline_points(centerline_path)
        
        if centerline_pts_full is not None:
            # Downsample centerline (like test.py)
            centerline_ds, _ = density_based_sample(
                centerline_pts_full, 
                min_distance=MAP_POINT_SPACING
            )
            print(f"[INFO] Downsampled centerline: {len(centerline_ds)} points")
            
            # Load BIRD model
            bird_config = BIRD_CONFIGS[args.model_mode]
            bird_model = BIRD(
                ant_mode=args.model_mode,
                **bird_config
            ).to(device)
            
            print(f"[INFO] Loading BIRD checkpoint: {args.bird_checkpoint}")
            bird_ckpt = torch.load(args.bird_checkpoint, map_location=device)
            
            if isinstance(bird_ckpt, dict) and 'bird_state_dict' in bird_ckpt:
                bird_model.load_state_dict(bird_ckpt['bird_state_dict'])
                print(f"[INFO] BIRD loaded from epoch {bird_ckpt.get('epoch', '?')}")
            elif isinstance(bird_ckpt, dict) and 'model_state_dict' in bird_ckpt:
                bird_model.load_state_dict(bird_ckpt['model_state_dict'])
                print(f"[INFO] BIRD loaded from epoch {bird_ckpt.get('epoch', '?')}")
            else:
                bird_model.load_state_dict(bird_ckpt)
            bird_model.eval()
            
            # Encode centerline (normalized, like train_bird.py)
            centerline_normalized = torch.tensor(centerline_ds / NORM_MAP_SCALE, dtype=torch.float32).to(device)
            with torch.no_grad():
                centerline_encoded = bird_model.encode_centerline(centerline_normalized)
            print(f"[INFO] BIRD ready with {len(centerline_ds)} centerline points")
    
    # Load 3D assets for visualization
    lung_mesh = None
    lung_path = os.path.join(args.data_root, "..", "patient", "lungs.obj")
    if os.path.exists(lung_path):
        print(f"[INFO] Loading lung mesh: {lung_path}")
        lung_mesh = pv.read(lung_path)
    
    centerline_path = os.path.join(args.data_root, "static", "centerline.npz")
    centerline_pts = load_centerline_points(centerline_path)
    if centerline_pts is not None:
        print(f"[INFO] Loaded {len(centerline_pts)} centerline points for visualization")
    
    # Initialize PyVista plotter
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.set_background('white')
    
    # Collect all results
    all_results = []
    
    # Number of runs per trajectory
    num_runs = args.runs
    
    # Process each sequence
    for seq_name in sorted(seq_to_indices.keys()):
        indices = seq_to_indices[seq_name]
        print(f"\n[EVAL] Processing: {seq_name} ({len(indices)} windows) - {num_runs} run(s)")
        
        # Get total frames in sequence (for timing calculation)
        first_sample_idx = indices[0]
        vid_path, _, _ = full_dataset.samples[first_sample_idx]
        vid_mmap = np.load(vid_path, mmap_mode='r')
        total_frames_in_seq = len(vid_mmap)
        del vid_mmap  # Close mmap
        
        # Create subset for this sequence
        seq_dataset = Subset(full_dataset, indices)
        seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Store results from multiple runs
        run_results = []
        
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"  Run {run_idx + 1}/{num_runs}")
            
            # Collect trajectory for this sequence
            all_timestamps = []
            all_gt_positions = []
            all_gt_quaternions = []
            all_pred_positions = []
            
            bird_mem_state = None  # Reset for each sequence/run
            prev_end_frame = -1
            
            # Start timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(seq_loader, desc=f"  {seq_name}" + (f" run {run_idx+1}" if num_runs > 1 else ""), leave=False)):
                    video = batch['video'].to(device)
                    gt_deltas = batch['actions'].to(device)
                    map_points = batch['map_points'].to(device)
                    map_mask = batch['map_mask'].to(device)
                    
                    first_pos = batch['first_frame_pos'].numpy()
                    first_quat = batch['first_frame_quat'].numpy()
                    raw_positions = batch['raw_positions'].numpy()
                    
                    # Forward pass - ANT model
                    if bird_model is not None:
                        pred_pos, delta_pos, delta_quat, visual_tokens, attn_probs = ant_model(
                            video, map_points=map_points, map_mask=map_mask, return_features=True
                        )
                    else:
                        pred_pos, pred_quat = ant_model(video, map_points=map_points, map_mask=map_mask)
                    
                    # Denormalize
                    gt_trans_local = gt_deltas.cpu().numpy() * NORM_MAP_SCALE
                    pred_pos_local = pred_pos.cpu().numpy() * NORM_MAP_SCALE
                    
                    B, T = video.shape[:2]
                    for b in range(B):
                        p0_gt = first_pos[b]
                        q0 = first_quat[b]
                        rot_0 = R.from_quat(q0)
                        
                        # Transform to global
                        gt_window_global = rot_0.apply(gt_trans_local[b]) + p0_gt
                        pred_window_global = rot_0.apply(pred_pos_local[b]) + p0_gt
                        
                        # BIRD inference
                        if bird_model is not None:
                            ant_pos_global = torch.tensor(pred_window_global / NORM_MAP_SCALE, dtype=torch.float32).unsqueeze(0).to(device)
                            vis_tokens_b = visual_tokens[b:b+1]
                            delta_pos_b = delta_pos[b:b+1]
                            delta_quat_b = delta_quat[b:b+1]
                            
                            p_refined, bird_mem_state, _, _ = bird_model(
                                ant_pos_global, delta_pos_b, delta_quat_b, vis_tokens_b,
                                centerline_encoded, centerline_normalized,
                                mem_state=bird_mem_state
                            )
                            
                            bird_window_global = p_refined.cpu().numpy()[0] * NORM_MAP_SCALE
                        else:
                            bird_window_global = pred_window_global
                        
                        # Store results (avoid duplicates from overlapping windows)
                        start_t = 0 if len(all_pred_positions) == 0 else 1
                        
                        for t in range(start_t, T):
                            timestamp = batch_idx * (T - 1) * frame_skip / 30.0 + t * frame_skip / 30.0
                            all_timestamps.append(timestamp)
                            all_gt_positions.append(gt_window_global[t])
                            all_gt_quaternions.append(first_quat[b])  # Use first frame quat
                            all_pred_positions.append(bird_window_global[t])
            
            # End timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            total_inference_time = end_time - start_time
            ms_per_frame = (total_inference_time * 1000) / total_frames_in_seq
            
            if len(all_pred_positions) < 3:
                print(f"[WARNING] Not enough points for {seq_name} run {run_idx+1}, skipping run")
                continue
            
            gt_pos = np.array(all_gt_positions)
            pred_pos = np.array(all_pred_positions)
            timestamps = np.array(all_timestamps)
            gt_quat = np.array(all_gt_quaternions)
            
            # Verify BIRD predictions are on centerline
            centerline_tree = cKDTree(centerline_pts)
            dists_to_centerline, _ = centerline_tree.query(pred_pos)
            max_dist = np.max(dists_to_centerline)
            mean_dist = np.mean(dists_to_centerline)
            if max_dist > 1.0:  # More than 1mm from centerline
                print(f"  [WARNING] BIRD predictions are up to {max_dist:.2f}mm from centerline! (mean: {mean_dist:.2f}mm)")
            
            # Umeyama alignment
            rot, trans, scale = umeyama_alignment(pred_pos, gt_pos, with_scale=args.with_scale)
            aligned_pred = apply_transform(pred_pos, rot, trans, scale)
            
            # Compute ATE for this run
            ate_stats = compute_ate(gt_pos, aligned_pred)
            run_results.append({
                'rmse': ate_stats['rmse'],
                'mean': ate_stats['mean'],
                'median': ate_stats['median'],
                'std': ate_stats['std'],
                'max': ate_stats['max'],
                'min': ate_stats['min'],
                'ms_per_frame': ms_per_frame,
            })
            
            if num_runs > 1:
                print(f"    Run {run_idx+1} - ATE RMSE: {ate_stats['rmse']:.2f} mm, Mean: {ate_stats['mean']:.2f} mm")
        
        # Skip sequence if no valid runs
        if len(run_results) == 0:
            print(f"[WARNING] No valid runs for {seq_name}, skipping")
            continue
        
        # Average results across all runs
        avg_rmse = np.mean([r['rmse'] for r in run_results])
        avg_mean = np.mean([r['mean'] for r in run_results])
        avg_median = np.mean([r['median'] for r in run_results])
        avg_std = np.mean([r['std'] for r in run_results])
        avg_max = np.mean([r['max'] for r in run_results])
        avg_min = np.mean([r['min'] for r in run_results])
        avg_ms = np.mean([r['ms_per_frame'] for r in run_results])
        
        # Also compute std across runs for reporting
        std_rmse = np.std([r['rmse'] for r in run_results]) if num_runs > 1 else 0.0
        std_mean = np.std([r['mean'] for r in run_results]) if num_runs > 1 else 0.0
        std_max = np.std([r['max'] for r in run_results]) if num_runs > 1 else 0.0
        std_ms = np.std([r['ms_per_frame'] for r in run_results]) if num_runs > 1 else 0.0
        
        all_results.append({
            'sequence': seq_name,
            'rmse': avg_rmse,
            'mean': avg_mean,
            'median': avg_median,
            'std': avg_std,
            'max': avg_max,
            'min': avg_min,
            'ms_per_frame': avg_ms,
            'total_frames': total_frames_in_seq,
            'num_runs': len(run_results),
            'rmse_std': std_rmse,
            'mean_std': std_mean,
            'max_std': std_max,
            'ms_std': std_ms,
        })
        
        if num_runs > 1:
            print(f"  [AVG] ATE RMSE: {avg_rmse:.2f}±{std_rmse:.2f} mm, Mean: {avg_mean:.2f}±{std_mean:.2f} mm, Max: {avg_max:.2f}±{std_max:.2f} mm")
            print(f"  [AVG] Inference: {avg_ms:.2f}±{std_ms:.2f} ms/frame ({total_frames_in_seq} frames, {len(run_results)} runs)")
        else:
            print(f"  ATE RMSE: {avg_rmse:.2f} mm, Mean: {avg_mean:.2f} mm, Max: {avg_max:.2f} mm")
            print(f"  Inference: {avg_ms:.2f} ms/frame ({total_frames_in_seq} frames)")
        
        # Save trajectories
        tum_path = os.path.join(args.output_dir, f"{seq_name}_pred.tum")
        save_trajectory_tum(tum_path, timestamps, aligned_pred, gt_quat)
        
        gt_tum_path = os.path.join(args.output_dir, f"{seq_name}_gt.tum")
        save_trajectory_tum(gt_tum_path, timestamps, gt_pos, gt_quat)
        
        # Render and save plot
        render_evaluation_plot(plotter, lung_mesh, centerline_pts,
                              gt_pos, pred_pos, aligned_pred, seq_name, ate_stats)
        
        plotter.camera_position = 'xy'
        plotter.camera.zoom(1.2)
        
        # Save PNG screenshot
        plot_path = os.path.join(args.output_dir, f"{seq_name}_plot.png")
        plotter.screenshot(plot_path)
        print(f"[INFO] Saved plot: {plot_path}")
        
        # Save interactive 3D HTML
        html_path = os.path.join(args.output_dir, f"{seq_name}_3d.html")
        plotter.export_html(html_path)
        print(f"[INFO] Saved 3D plot: {html_path}")
    
    plotter.close()
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Sequence':<30} {'RMSE':>10} {'Mean':>10} {'Max':>10} {'ms/frame':>12}")
    print("-" * 80)
    
    rmse_all = []
    mean_all = []
    max_all = []
    ms_all = []
    for r in all_results:
        if num_runs > 1:
            print(f"{r['sequence']:<30} {r['rmse']:>10.2f}±{r['rmse_std']:.1f} {r['mean']:>10.2f}±{r['mean_std']:.1f} {r['max']:>10.2f}±{r['max_std']:.1f} {r['ms_per_frame']:>12.2f}")
        else:
            print(f"{r['sequence']:<30} {r['rmse']:>10.2f} {r['mean']:>10.2f} {r['max']:>10.2f} {r['ms_per_frame']:>12.2f}")
        rmse_all.append(r['rmse'])
        mean_all.append(r['mean'])
        max_all.append(r['max'])
        ms_all.append(r['ms_per_frame'])
    
    print("-" * 80)
    if rmse_all:
        print(f"{'AVERAGE':<30} {np.mean(rmse_all):>10.2f} {np.mean(mean_all):>10.2f} {np.mean(max_all):>10.2f} {np.mean(ms_all):>12.2f}")
        print(f"{'STD':<30} {np.std(rmse_all):>10.2f} {np.std(mean_all):>10.2f} {np.std(max_all):>10.2f} {np.std(ms_all):>12.2f}")
    print("=" * 80)
    
    # Save summary CSV
    summary_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    with open(summary_path, 'w') as f:
        f.write("sequence,rmse,mean,median,std,max,min,ms_per_frame,total_frames,num_runs,rmse_std,mean_std,max_std,ms_std\n")
        for r in all_results:
            f.write(f"{r['sequence']},{r['rmse']:.4f},{r['mean']:.4f},{r['median']:.4f},{r['std']:.4f},{r['max']:.4f},{r['min']:.4f},{r['ms_per_frame']:.4f},{r['total_frames']},{r['num_runs']},{r['rmse_std']:.4f},{r['mean_std']:.4f},{r['max_std']:.4f},{r['ms_std']:.4f}\n")
    print(f"\n[INFO] Summary saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BronchoLoc model on centerline sequences")
    
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--ant_checkpoint', type=str, required=True, help="Path to ANT checkpoint")
    parser.add_argument('--bird_checkpoint', type=str, default=None, help="Path to BIRD checkpoint")
    parser.add_argument('--model_mode', type=str, default='s', choices=['xs', 's', 'b', 'm', 'l'])
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--frame_skip', type=int, default=None, help="Override frame_skip")
    parser.add_argument('--with_scale', action='store_true', help="Use Sim3 alignment (with scale) instead of SE3")
    parser.add_argument('--output_dir', type=str, default='./dataset/eval/trajectories/results')
    parser.add_argument('--runs', type=int, default=1, help="Number of times to run each trajectory (results are averaged)")
    
    args = parser.parse_args()
    evaluate(args)
