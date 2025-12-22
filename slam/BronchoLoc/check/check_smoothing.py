#!/usr/bin/env python
"""
Check Trajectory Smoothing

Interactive tool to test and visualize trajectory smoothing.
Shows BEFORE and AFTER in separate plots for clear comparison.

Usage:
    python check/check_smoothing.py <sequence_name>
    python check/check_smoothing.py 1 --window 300
"""

import os
import sys
import json
import argparse
import numpy as np
import pyvista as pv
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tum_trajectory(filepath):
    """Load TUM format trajectory (positions only)."""
    data = np.loadtxt(filepath, comments='#')
    if data[0, 0] > data[-1, 0]:
        data = np.flip(data, axis=0)
    
    timestamps = data[:, 0]
    positions = data[:, 1:4] * 1000  # m to mm
    return timestamps, positions


def load_transform(transform_path):
    """Load similarity transform from JSON."""
    with open(transform_path, 'r') as f:
        data = json.load(f)
    return data['scale'], np.array(data['rotation_matrix']), np.array(data['translation'])


def apply_transform(positions, scale, R_mat, t_vec):
    """Apply similarity transformation to positions."""
    return scale * (positions @ R_mat.T) + t_vec


def smooth_positions_savgol(positions, window_length=300, polyorder=3):
    """Smooth positions using Savitzky-Golay filter."""
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    if window_length > len(positions):
        window_length = len(positions) // 2 * 2 + 1
        if window_length <= polyorder:
            return positions.copy()
    
    smoothed = np.zeros_like(positions)
    for i in range(3):
        smoothed[:, i] = savgol_filter(positions[:, i], window_length, polyorder)
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Check trajectory smoothing")
    parser.add_argument('sequence_name', help='Phantom sequence name (e.g., 1, lb)')
    parser.add_argument('--window', type=int, default=300, help='Savgol window size (default: 300)')
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "dataset", "phantom", "data")
    patient_dir = os.path.join(base_dir, "patient")
    
    traj_path = os.path.join(data_dir, f"{args.sequence_name}_gt.txt")
    transform_path = os.path.join(data_dir, f"{args.sequence_name}_transform.json")
    mesh_path = os.path.join(patient_dir, "lungs.obj")
    
    # Validate
    for path, name in [(traj_path, "Trajectory"), (transform_path, "Transform")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)
    
    # Load
    print(f"Loading trajectory: {traj_path}")
    timestamps, positions = load_tum_trajectory(traj_path)
    print(f"  {len(timestamps)} poses")
    
    print(f"Loading transform: {transform_path}")
    scale, R_mat, t_vec = load_transform(transform_path)
    
    # Apply transform to align with 3D model
    print("Applying alignment transformation...")
    positions = apply_transform(positions, scale, R_mat, t_vec)
    
    # Load mesh
    mesh = pv.read(mesh_path) if os.path.exists(mesh_path) else None
    
    # Smooth
    print(f"\nSmoothing positions (Savgol window={args.window})...")
    positions_smooth = smooth_positions_savgol(positions, window_length=args.window)
    
    # Stats
    pos_diff = np.linalg.norm(positions - positions_smooth, axis=1)
    print(f"\nPosition change: mean={pos_diff.mean():.2f}mm, max={pos_diff.max():.2f}mm")
    
    # === SIDE-BY-SIDE COMPARISON ===
    print("\n>>> Showing ORIGINAL (left) vs SMOOTHED (right)...")
    p = pv.Plotter(shape=(1, 2), title=f"Smoothing Comparison: {args.sequence_name}")
    
    # --- LEFT: ORIGINAL ---
    p.subplot(0, 0)
    if mesh is not None:
        p.add_mesh(mesh, color='wheat', opacity=0.2)
    p.add_mesh(pv.lines_from_points(positions), color='blue', line_width=3)
    p.add_mesh(pv.Sphere(radius=1.0, center=positions[0]), color='lime')
    p.add_mesh(pv.Sphere(radius=1.0, center=positions[-1]), color='orange')
    p.add_text(f"ORIGINAL\n{len(positions)} poses", position='upper_left', font_size=12)
    p.add_axes()
    
    # --- RIGHT: SMOOTHED ---
    p.subplot(0, 1)
    if mesh is not None:
        p.add_mesh(mesh, color='wheat', opacity=0.2)
    p.add_mesh(pv.lines_from_points(positions_smooth), color='blue', line_width=3)
    p.add_mesh(pv.Sphere(radius=1.0, center=positions_smooth[0]), color='lime')
    p.add_mesh(pv.Sphere(radius=1.0, center=positions_smooth[-1]), color='orange')
    p.add_text(f"SMOOTHED\nwindow={args.window}\n"
               f"Δ: mean={pos_diff.mean():.2f}mm, max={pos_diff.max():.2f}mm", 
               position='upper_left', font_size=10)
    p.add_axes()
    
    p.link_views()
    p.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
