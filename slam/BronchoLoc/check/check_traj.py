import os
import sys
import glob
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_centerline_points

class TrajectoryVisualizer:
    def __init__(self, data_root, cad_path, centerline_path=None):
        self.data_root = data_root
        self.cad_path = cad_path
        self.centerline_path = centerline_path
        self.cad_mesh = None
        self.centerline_points = None
        
        # Load CAD once
        if os.path.exists(self.cad_path):
            print(f"[INFO] Loading CAD: {self.cad_path}")
            self.cad_mesh = pv.read(self.cad_path)
        else:
            print(f"[WARNING] CAD file not found at {self.cad_path}. Visualizing trajectories only.")
        
        # Load Centerline once
        if self.centerline_path:
            self.centerline_points = load_centerline_points(self.centerline_path)
            if self.centerline_points is not None:
                print(f"[INFO] Loaded {len(self.centerline_points)} centerline points")

    def load_pose_file(self, filepath):
        """
        Robustly loads trajectory data. Handles:
        1. New BronchoSim Format: .npy (N, 7) -> [x, y, z, qx, qy, qz, qw]
        2. Old/Matrix Format: .npy (N, 4, 4)
        3. TUM Text Format: (N, 8)
        
        Returns: (N, 4, 4) Homogeneous Transformation Matrices
        """
        try:
            if filepath.endswith('.npy'):
                data = np.load(filepath)
            else:
                # Load text, ignore comments
                data = np.loadtxt(filepath, comments='#')

            # --- CASE 1: NEW BRONCHOSIM FORMAT (N, 7) ---
            # [x, y, z, qx, qy, qz, qw]
            if data.ndim == 2 and data.shape[1] == 7:
                N = data.shape[0]
                t = data[:, 0:3]
                q = data[:, 3:7] # [qx, qy, qz, qw]
                
                rot_matrices = R.from_quat(q).as_matrix()
                
                matrices = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
                matrices[:, :3, :3] = rot_matrices
                matrices[:, :3, 3] = t
                return matrices

            # --- CASE 2: TUM FORMAT (N, 8) ---
            # [timestamp, x, y, z, qx, qy, qz, qw]
            if data.ndim == 2 and data.shape[1] == 8:
                N = data.shape[0]
                t = data[:, 1:4]
                q = data[:, 4:8] 
                
                rot_matrices = R.from_quat(q).as_matrix()
                
                matrices = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
                matrices[:, :3, :3] = rot_matrices
                matrices[:, :3, 3] = t
                return matrices

            # --- CASE 3: MATRIX FORMAT (N, 4, 4) ---
            if data.ndim == 3 and data.shape[1] == 4 and data.shape[2] == 4:
                return data

            # --- CASE 4: FLATTENED MATRIX (N, 16) ---
            if data.ndim == 2 and data.shape[1] == 16:
                return data.reshape(-1, 4, 4)

            print(f"[WARNING] Unknown data shape: {data.shape}")
            return None

        except Exception as e:
            print(f"  [ERROR] Could not load {filepath}: {e}")
            return None

    def visualize_sequence(self, seq_name, seq_path):
        print(f"\n--- Visualizing {seq_name} ---")
        
        # Look for the new file format
        traj_path = os.path.join(seq_path, 'trajectory.npy')
        
        if not os.path.exists(traj_path):
            print(f"  [SKIP] trajectory.npy not found in {seq_path}")
            return

        # Load Data
        poses = self.load_pose_file(traj_path)
        if poses is None:
            return

        # Initialize Plotter
        p = pv.Plotter(title=f"Check: {seq_name}")
        
        # 1. Plot CAD (Ghostly)
        if self.cad_mesh:
            p.add_mesh(self.cad_mesh, color='wheat', opacity=0.25, label='Lungs CAD')
        
        # 2. Plot Centerline (Faint)
        if self.centerline_points is not None:
            p.add_mesh(pv.PolyData(self.centerline_points), color='black', opacity=0.2, point_size=3, render_points_as_spheres=True, label='Centerline')

        # 3. Plot Trajectory Line
        positions = poses[:, :3, 3]
        # Create a continuous line
        line = pv.lines_from_points(positions)
        p.add_mesh(line, color='blue', line_width=4, label='Recorded Trajectory')
        
        # 3. Plot Start and End Orientation
        # Start (Green Axes)
        self.add_axes_at_pose(p, poses[0], scale=15) 
        p.add_point_labels([positions[0]], ["Start"], point_size=10, text_color='black', always_visible=True)
        
        # End (Red Axes)
        self.add_axes_at_pose(p, poses[-1], scale=15)
        p.add_point_labels([positions[-1]], ["End"], point_size=10, text_color='black', always_visible=True)

        p.add_legend()
        p.add_axes()
        print(f"  > Showing plot... (Close window to verify next sequence)")
        p.show()

    def add_axes_at_pose(self, plotter, matrix, scale=5.0):
        """ Draws RGB axes at a specific 4x4 pose matrix """
        origin = matrix[:3, 3]
        x_axis = matrix[:3, 0] * scale
        y_axis = matrix[:3, 1] * scale
        z_axis = matrix[:3, 2] * scale
        
        plotter.add_arrows(np.array([origin]), np.array([x_axis]), color='red', show_scalar_bar=False)
        plotter.add_arrows(np.array([origin]), np.array([y_axis]), color='green', show_scalar_bar=False)
        plotter.add_arrows(np.array([origin]), np.array([z_axis]), color='blue', show_scalar_bar=False)

    def run(self):
        # Find all sequence folders in the root
        # Assumes structure: DATASET_ROOT/seq_12345/trajectory.npy
        seq_pattern = os.path.join(self.data_root, "seq_*")
        sequences = sorted(glob.glob(seq_pattern))
        
        if not sequences:
            print(f"[ERROR] No sequences found matching {seq_pattern}")
            print(f"        Ensure DATASET_ROOT points to the folder containing 'seq_X' folders.")
            return

        print(f"Found {len(sequences)} sequences.")
        
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            self.visualize_sequence(seq_name, seq_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None, help="Filter sequences by name (substring match)")
    parser.add_argument("--centerline_path", type=str, default='./dataset/static/centerline.npz', help="Path to centerline centerline file")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    # BASE_DIR is the parent of check folder (BronchoLoc root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_ROOT = os.path.join(BASE_DIR, "dataset", "sequences")
    CAD_FILE = os.path.join(BASE_DIR, "patient", "lungs.obj")
    Centerline_FILE = os.path.join(BASE_DIR, args.centerline_path) if not os.path.isabs(args.centerline_path) else args.centerline_path
    # ---------------------
    
    viz = TrajectoryVisualizer(DATASET_ROOT, CAD_FILE, Centerline_FILE)
    
    # Monkey patch run to support filter (or modify run method, but this is cleaner for now without changing class signature)
    original_run = viz.run
    
    def run_with_filter():
        seq_pattern = os.path.join(viz.data_root, "seq_*")
        sequences = sorted(glob.glob(seq_pattern))
        
        if not sequences:
            print(f"[ERROR] No sequences found matching {seq_pattern}")
            return

        if args.filter:
            sequences = [s for s in sequences if args.filter in os.path.basename(s)]
            print(f"Filtered to {len(sequences)} sequences matching '{args.filter}'")

        print(f"Found {len(sequences)} sequences.")
        
        for seq_path in sequences:
            seq_name = os.path.basename(seq_path)
            viz.visualize_sequence(seq_name, seq_path)
            
    viz.run = run_with_filter
    viz.run()