#!/usr/bin/env python
"""
Refine Phantom Trajectory Alignment

Interactive tool to manually adjust the transformation computed by align_phantom_traj.py.
Allows translation, rotation, and scale adjustments with real-time visualization.

Usage:
    python refine_phantom_traj.py <sequence_name>
"""

import os
import sys
import json
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R


def load_trajectory(trajectory_path):
    """Load TUM format trajectory."""
    data = np.loadtxt(trajectory_path, comments='#')
    if data[0, 0] > data[-1, 0]:
        data = np.flip(data, axis=0)
    return data[:, 1:4] * 1000  # Convert to mm


def load_transform(transform_path):
    """Load existing transformation."""
    with open(transform_path, 'r') as f:
        data = json.load(f)
    return {
        'scale': data['scale'],
        'rotation_matrix': np.array(data['rotation_matrix']),
        'translation': np.array(data['translation']),
        'original_data': data
    }


def apply_transform(positions, scale, R_mat, translation):
    """Apply similarity transformation."""
    return scale * (positions @ R_mat.T) + translation


class TransformRefiner:
    def __init__(self, sequence_name, base_dir):
        self.sequence_name = sequence_name
        self.base_dir = base_dir
        
        # Paths
        data_dir = os.path.join(base_dir, "dataset", "phantom", "data")
        patient_dir = os.path.join(base_dir, "patient")
        
        self.trajectory_path = os.path.join(data_dir, f"{sequence_name}_gt.txt")
        self.transform_path = os.path.join(data_dir, f"{sequence_name}_transform.json")
        self.mesh_path = os.path.join(patient_dir, "lungs.obj")
        
        # Determine centerline path from CSV lookup
        csv_path = os.path.join(data_dir, "closest_centerline.csv")
        centerline_name = None
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2 and parts[0] == sequence_name:
                        centerline_name = parts[1]
                        break
        
        if centerline_name:
            self.centerline_path = os.path.join(patient_dir, "centerlines", f"{centerline_name}.vtp")
            print(f"Using centerline from CSV: {centerline_name}")
        else:
            self.centerline_path = os.path.join(patient_dir, "centerline.vtk")
            print(f"No CSV mapping for '{sequence_name}', using default centerline")
        
        # Load data
        self.positions_raw = load_trajectory(self.trajectory_path)
        self.transform_data = load_transform(self.transform_path)
        self.mesh = pv.read(self.mesh_path)
        
        if os.path.exists(self.centerline_path):
            self.centerline = pv.read(self.centerline_path)
        else:
            self.centerline = None
        
        # Current transform parameters
        self.scale = self.transform_data['scale']
        self.R_mat = self.transform_data['rotation_matrix']
        self.translation = self.transform_data['translation'].copy()
        
        # Adjustment deltas (added on top of original transform)
        self.delta_translation = np.zeros(3)
        self.delta_euler = np.zeros(3)  # degrees
        self.delta_scale = 0.0
        
        self.plotter = None
        
    def get_current_transform(self):
        """Get current combined transformation."""
        # Apply delta adjustments
        scale = self.scale + self.delta_scale
        
        # Combine rotations
        delta_R = R.from_euler('xyz', self.delta_euler, degrees=True).as_matrix()
        R_mat = delta_R @ self.R_mat
        
        translation = self.translation + self.delta_translation
        
        return scale, R_mat, translation
    
    def update_trajectory(self):
        """Update trajectory visualization."""
        scale, R_mat, translation = self.get_current_transform()
        positions_aligned = apply_transform(self.positions_raw, scale, R_mat, translation)
        
        # Update trajectory points
        step = max(1, len(positions_aligned) // 500)
        traj_points = positions_aligned[::step]
        
        self.plotter.remove_actor('trajectory')
        traj_poly = pv.PolyData(traj_points)
        self.plotter.add_mesh(traj_poly, color='blue', point_size=4,
                             render_points_as_spheres=True, name='trajectory')
        
        # Update start/end
        self.plotter.remove_actor('start')
        self.plotter.remove_actor('end')
        start = pv.Sphere(radius=1.0, center=positions_aligned[0])
        end = pv.Sphere(radius=1.0, center=positions_aligned[-1])
        self.plotter.add_mesh(start, color='lime', name='start')
        self.plotter.add_mesh(end, color='orange', name='end')
    
    def save_transform(self):
        """Save refined transformation."""
        scale, R_mat, translation = self.get_current_transform()
        
        # Update the original data
        data = self.transform_data['original_data'].copy()
        data['scale'] = float(scale)
        data['rotation_matrix'] = R_mat.tolist()
        data['translation'] = translation.tolist()
        
        # Update euler angles
        rot = R.from_matrix(R_mat)
        data['rotation_euler_xyz_deg'] = rot.as_euler('xyz', degrees=True).tolist()
        data['rotation_quaternion_xyzw'] = rot.as_quat().tolist()
        
        # Update 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = scale * R_mat
        T[:3, 3] = translation
        data['transformation_matrix_4x4'] = T.tolist()
        
        # Add refinement note
        data['refined'] = True
        data['refinement_deltas'] = {
            'translation': self.delta_translation.tolist(),
            'euler_deg': self.delta_euler.tolist(),
            'scale': float(self.delta_scale)
        }
        
        # Save
        with open(self.transform_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"SAVED: {self.transform_path}")
        print(f"  Scale: {scale:.6f}")
        print(f"  Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
        print(f"{'='*50}\n")
    
    def run(self):
        """Run interactive visualization."""
        self.plotter = pv.Plotter(title=f"Refine Alignment: {self.sequence_name}")
        
        # Add mesh
        self.plotter.add_mesh(self.mesh, color='lightblue', opacity=0.15)
        
        # Add centerline
        if self.centerline is not None:
            self.plotter.add_mesh(self.centerline, color='gray', point_size=2,
                                 render_points_as_spheres=True, opacity=0.4)
        
        # Initial trajectory
        self.update_trajectory()
        
        # Sliders for translation
        def update_tx(val):
            self.delta_translation[0] = val
            self.update_trajectory()
        def update_ty(val):
            self.delta_translation[1] = val
            self.update_trajectory()
        def update_tz(val):
            self.delta_translation[2] = val
            self.update_trajectory()
        
        self.plotter.add_slider_widget(update_tx, rng=[-20, 20], value=0,
                                       title="Translate X", pointa=(0.02, 0.9), pointb=(0.18, 0.9))
        self.plotter.add_slider_widget(update_ty, rng=[-20, 20], value=0,
                                       title="Translate Y", pointa=(0.02, 0.8), pointb=(0.18, 0.8))
        self.plotter.add_slider_widget(update_tz, rng=[-20, 20], value=0,
                                       title="Translate Z", pointa=(0.02, 0.7), pointb=(0.18, 0.7))
        
        # Sliders for rotation
        def update_rx(val):
            self.delta_euler[0] = val
            self.update_trajectory()
        def update_ry(val):
            self.delta_euler[1] = val
            self.update_trajectory()
        def update_rz(val):
            self.delta_euler[2] = val
            self.update_trajectory()
        
        self.plotter.add_slider_widget(update_rx, rng=[-15, 15], value=0,
                                       title="Rotate X", pointa=(0.02, 0.55), pointb=(0.18, 0.55))
        self.plotter.add_slider_widget(update_ry, rng=[-15, 15], value=0,
                                       title="Rotate Y", pointa=(0.02, 0.45), pointb=(0.18, 0.45))
        self.plotter.add_slider_widget(update_rz, rng=[-15, 15], value=0,
                                       title="Rotate Z", pointa=(0.02, 0.35), pointb=(0.18, 0.35))
        
        # Slider for scale
        def update_scale(val):
            self.delta_scale = val
            self.update_trajectory()
        
        self.plotter.add_slider_widget(update_scale, rng=[-0.1, 0.1], value=0,
                                       title="Scale Δ", pointa=(0.02, 0.2), pointb=(0.18, 0.2))
        
        # Save button via key
        def save_action():
            self.save_transform()
        
        self.plotter.add_key_event('s', save_action)
        
        self.plotter.add_text(
            f"Sequence: {self.sequence_name}\n\n"
            "Adjust sliders to refine alignment.\n\n"
            "S = Save transformation\n"
            "Q = Quit without saving",
            position='upper_right', font_size=10
        )
        
        self.plotter.add_axes()
        self.plotter.show()


def main():
    parser = argparse.ArgumentParser(description='Manually refine phantom trajectory alignment')
    parser.add_argument('sequence_name', help='Name of the sequence')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "dataset", "phantom", "data")
    
    transform_path = os.path.join(data_dir, f"{args.sequence_name}_transform.json")
    if not os.path.exists(transform_path):
        print(f"Error: Transform not found: {transform_path}")
        print(f"Run align_phantom_traj.py {args.sequence_name} first.")
        sys.exit(1)
    
    refiner = TransformRefiner(args.sequence_name, base_dir)
    refiner.run()


if __name__ == "__main__":
    main()
