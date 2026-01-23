#!/usr/bin/env python3
"""
Check orientation compatibility between simulation and phantom sequences.
Visualizes trajectory positions with orientation frames to verify coordinate systems match.

Usage:
    python check/check_orientation.py --sim seq_b1_var0_1765525016 --phantom seq_phantom_1_part1_sq
"""

import numpy as np
import pyvista as pv
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.spatial.transform import Rotation as R


def load_trajectory(seq_path):
    """Load trajectory.npy: (N, 7) with [x, y, z, qx, qy, qz, qw]"""
    traj_file = os.path.join(seq_path, "trajectory.npy")
    if not os.path.exists(traj_file):
        raise FileNotFoundError(f"No trajectory.npy found at {traj_file}")
    
    data = np.load(traj_file)
    positions = data[:, :3]
    quaternions = data[:, 3:7]  # qx, qy, qz, qw
    return positions, quaternions


def create_coordinate_frame_arrows(position, quaternion, scale=5.0):
    """Create coordinate frame as 3 arrows at position with given orientation."""
    rot = R.from_quat(quaternion)  # scipy uses [x, y, z, w]
    
    x_global = rot.apply([1, 0, 0]) * scale
    y_global = rot.apply([0, 1, 0]) * scale
    z_global = rot.apply([0, 0, 1]) * scale
    
    return [
        (position, position + x_global, 'red'),      # X axis
        (position, position + y_global, 'green'),    # Y axis  
        (position, position + z_global, 'blue'),     # Z axis
    ]


def add_trajectory_with_frames(plotter, positions, quaternions, 
                                line_color='white', frame_interval=10, frame_scale=5.0):
    """Add trajectory line and coordinate frames at intervals."""
    
    # Add trajectory line
    if len(positions) > 1:
        line = pv.lines_from_points(positions)
        plotter.add_mesh(line, color=line_color, line_width=3)
    
    # Add coordinate frames at intervals
    n_points = len(positions)
    for i in range(0, n_points, frame_interval):
        frames = create_coordinate_frame_arrows(positions[i], quaternions[i], scale=frame_scale)
        for start, end, color in frames:
            direction = end - start
            length = np.linalg.norm(direction)
            if length > 0:
                arrow = pv.Arrow(start=start, direction=direction/length, scale=length)
                plotter.add_mesh(arrow, color=color, opacity=0.8)


def compare_sequences(sim_seq, phantom_seq, data_root, centerline_path, output_html):
    """Compare sim and phantom sequences side by side."""
    
    sim_path = os.path.join(data_root, "sequences", sim_seq)
    phantom_path = os.path.join(data_root, "sequences", phantom_seq)
    
    print(f"[INFO] Loading simulation: {sim_path}")
    sim_pos, sim_quat = load_trajectory(sim_path)
    print(f"  Loaded {len(sim_pos)} poses")
    print(f"  First quat (qx,qy,qz,qw): [{sim_quat[0][0]:.4f}, {sim_quat[0][1]:.4f}, {sim_quat[0][2]:.4f}, {sim_quat[0][3]:.4f}]")
    
    print(f"\n[INFO] Loading phantom: {phantom_path}")
    phantom_pos, phantom_quat = load_trajectory(phantom_path)
    print(f"  Loaded {len(phantom_pos)} poses")
    print(f"  First quat (qx,qy,qz,qw): [{phantom_quat[0][0]:.4f}, {phantom_quat[0][1]:.4f}, {phantom_quat[0][2]:.4f}, {phantom_quat[0][3]:.4f}]")
    
    # Orientation analysis
    print("\n" + "=" * 60)
    print("ORIENTATION ANALYSIS")
    print("=" * 60)
    
    sim_rot = R.from_quat(sim_quat[0])
    phantom_rot = R.from_quat(phantom_quat[0])
    
    print("\nFirst frame axis directions (global frame):")
    for axis_name, axis_vec in [('X', [1,0,0]), ('Y', [0,1,0]), ('Z', [0,0,1])]:
        sim_axis = sim_rot.apply(axis_vec)
        phantom_axis = phantom_rot.apply(axis_vec)
        dot = np.dot(sim_axis, phantom_axis)
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        print(f"  {axis_name}-axis:")
        print(f"    Sim:     [{sim_axis[0]:+.3f}, {sim_axis[1]:+.3f}, {sim_axis[2]:+.3f}]")
        print(f"    Phantom: [{phantom_axis[0]:+.3f}, {phantom_axis[1]:+.3f}, {phantom_axis[2]:+.3f}]")
        print(f"    Dot product: {dot:+.3f}, Angle: {angle:.1f}°")
    
    # Check motion direction consistency
    print("\nMotion direction check (first 5 frames):")
    for name, pos, quat in [("Sim", sim_pos, sim_quat), ("Phantom", phantom_pos, phantom_quat)]:
        print(f"\n  {name}:")
        for i in range(min(5, len(pos)-1)):
            delta = pos[i+1] - pos[i]
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0.01:
                delta_dir = delta / delta_norm
                rot = R.from_quat(quat[i])
                z_global = rot.apply([0, 0, 1])  # Forward in local
                dot = np.dot(delta_dir, z_global)
                print(f"    Frame {i}: delta={delta_norm:.2f}mm, Z-dot={dot:+.3f} ({'forward' if dot > 0.5 else 'backward' if dot < -0.5 else 'sideways'})")
    
    # Create visualization
    print("\n[INFO] Creating 3D visualization...")
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Load centerline
    if os.path.exists(centerline_path):
        centerline_data = np.load(centerline_path)
        centerline_pts = centerline_data['centerline_points']
        plotter.add_mesh(pv.PolyData(centerline_pts), color='lightgray', point_size=1, opacity=0.3)
    
    # Add trajectories
    frame_interval_sim = max(1, len(sim_pos) // 10)
    frame_interval_phantom = max(1, len(phantom_pos) // 10)
    
    add_trajectory_with_frames(plotter, sim_pos, sim_quat, 
                               line_color='cyan', frame_interval=frame_interval_sim, frame_scale=3.0)
    add_trajectory_with_frames(plotter, phantom_pos, phantom_quat,
                               line_color='magenta', frame_interval=frame_interval_phantom, frame_scale=3.0)
    
    # Legend
    legend = "LEGEND\n━━━━━━━━━━━━\n"
    legend += "Cyan: Simulation\n"
    legend += "Magenta: Phantom\n\n"
    legend += "Arrows:\n"
    legend += "Red=X, Green=Y, Blue=Z"
    plotter.add_text(legend, position='upper_left', font_size=10, color='black')
    
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)
    
    if output_html:
        plotter.export_html(output_html)
        print(f"[INFO] Saved: {output_html}")
    
    print("\n[INFO] Showing 3D view - Blue arrow should point along trajectory direction")
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare orientation between sim and phantom sequences")
    parser.add_argument('--sim', type=str, required=True,
                        help="Simulation sequence name (e.g., seq_b1_var0_1765525016)")
    parser.add_argument('--phantom', type=str, required=True,
                        help="Phantom sequence name (e.g., seq_phantom_1_part1_sq)")
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help="Path to dataset root")
    parser.add_argument('--centerline', type=str, default='./dataset/static/centerline.npz',
                        help="Path to centerline.npz")
    parser.add_argument('--output', type=str, default=None,
                        help="Optional: Save interactive HTML")
    
    args = parser.parse_args()
    compare_sequences(args.sim, args.phantom, args.data_root, args.centerline, args.output)
