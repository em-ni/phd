import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from constants import MAP_QUERY_RADIUS, NORM_MAP_SCALE, DEFAULT_MAX_MAP_POINTS
from utils import filter_connected_component, load_centerline_points, farthest_point_sample

def get_stats(data_root):
    """
    Computes global statistics (min, max, center, scale) for all trajectories in the dataset.
    This validation step ensures we know the bounding box of the data for normalization purposes.
    The output should be used to set constants like NORM_MAP_SCALE.
    """
    traj_files = glob.glob(os.path.join(data_root, "seq_*", "trajectory.npy"))
    
    all_mins = []
    all_maxs = []
    
    print(f"[INFO] Scanning {len(traj_files)} sequences...")
    
    for f in tqdm(traj_files):
        # Load only xyz columns (0,1,2) of the trajectory data.
        # Format is likely [x, y, z, qx, qy, qz, qw].
        data = np.load(f)[:, :3] 
        all_mins.append(data.min(axis=0))
        all_maxs.append(data.max(axis=0))
    
    # Global stats across all sequences
    global_min = np.min(all_mins, axis=0)
    global_max = np.max(all_maxs, axis=0)
    
    # Calculate scale to fit in [-1, 1]
    # Center = (Max + Min) / 2
    # Scale = (Max - Min) / 2 (half-range)
    center = (global_max + global_min) / 2
    scale = (global_max - global_min) / 2
    
    # Use the largest scale dimension to preserve aspect ratio during normalization
    max_scale = scale.max()
    
    # Add 10% padding so we don't hit the edge - points at exactly the boundary might be clipped.
    max_scale *= 1.1
    
    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"Global Min (mm): {global_min}")
    print(f"Global Max (mm): {global_max}")
    print("-" * 20)
    print(f"Normalization Center: {center}")
    print(f"Normalization Scale:  {max_scale}")
    print("="*40)
    print("\nCopy these values into your constants.py!")

class AntDataset(Dataset):
    """
    PyTorch Dataset for the ANT model.
    Handles loading video frames, trajectories, and static airway maps, 
    and generating training samples consisting of video clips, map points, and target actions.
    """
    def __init__(self, data_root, mode='train', max_map_points=DEFAULT_MAX_MAP_POINTS, img_size=128, chain_mode=False):
        """
        Args:
            data_root (str): Path to the directory containing sequence folders.
            mode (str): 'train' or 'test'. Used for logging.
            max_map_points (int): Maximum number of map points to pass to model.
                                  If ball contains more, FPS downsampling is applied.
            img_size (int): Spatial resolution to resize video frames to (img_size x img_size).
            chain_mode (bool): If True, windows overlap by 1 frame (for chained prediction testing).
        """
        self.data_root = data_root
        self.mode = mode
        self.max_map_points = max_map_points
        self.img_size = img_size
        self.chain_mode = chain_mode
        self.samples = []
        
        # Load window config from file
        from constants import load_window_config
        self.window_size, self.frame_skip = load_window_config()
        print(f"[DATASET] Loaded config: window_size={self.window_size}, frame_skip={self.frame_skip}")
        
        # --- LOAD MAP (CENTERLINE) ---
        # Assume static data is in parent of 'sequences' or 'test'
        # data_root is usually .../dataset/sequences
        parent_dir = os.path.dirname(self.data_root)
        graph_path = os.path.join(parent_dir, "static", "centerline.npz")
        
        self.map_tree = None
        self.map_points = None
        self.map_points = load_centerline_points(graph_path)
        if self.map_points is not None:
            print(f"[DATASET] Loaded {len(self.map_points)} centerline points from {graph_path}")
            # Build a KDTree for fast spatial queries (finding nearest map points).
            self.map_tree = cKDTree(self.map_points)
        else:
            print(f"[WARNING] Map inputs will be zero.")

        # Index the dataset to find all valid sliding windows.
        self._index_dataset()

    def _index_dataset(self):
        """
        Scans all sequence directories and creates a list of valid (video_path, traj_path, start_index) tuples.
        Each tuple represents one training sample.
        """
        seq_dirs = sorted(glob.glob(os.path.join(self.data_root, "seq_*")))
        if not seq_dirs: return

        print(f"[DATASET] Indexing {self.mode} data...")
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir): continue
            vid_path = os.path.join(seq_dir, "video.npy")
            traj_path = os.path.join(seq_dir, "trajectory.npy")
            
            if not (os.path.exists(vid_path) and os.path.exists(traj_path)): continue
            
            try:
                # Use mmap_mode to read shape without loading data to RAM
                vid_data = np.load(vid_path, mmap_mode='r')
                N_vid = vid_data.shape[0]
                
                # Create sliding windows
                # Total span in video: window_size frames with frame_skip gap between each
                # Last frame index = start + (window_size - 1) * frame_skip
                # So we need at least start + (window_size - 1) * frame_skip + 1 frames
                effective_len = (self.window_size - 1) * self.frame_skip + 1
                
                if N_vid >= effective_len:
                    # Step size depends on mode:
                    # - Normal: effective_len (non-overlapping windows)
                    # - Chain mode: (window_size - 1) * frame_skip (overlap by 1 sampled frame)
                    if self.chain_mode:
                        # Overlap by 1 frame: last frame of window N = first frame of window N+1
                        step_size = (self.window_size - 1) * self.frame_skip
                    else:
                        step_size = effective_len
                    
                    for start_idx in range(0, N_vid - effective_len + 1, step_size):
                        self.samples.append((vid_path, traj_path, start_idx))
            except Exception as e:
                print(f"Error: {e}")

        print(f"[DATASET] Found {len(self.samples)} valid windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads a single sequence window (video + trajectory + map).
        """
        vid_path, traj_path, start_idx = self.samples[idx]
        
        # Load data using mmap to save memory
        vid_mmap = np.load(vid_path, mmap_mode='r')
        traj_mmap = np.load(traj_path, mmap_mode='r')
        
        # Select frames: start, start+frame_skip, start+2*frame_skip, ...
        # for a total of window_size frames
        frame_indices = [start_idx + i * self.frame_skip for i in range(self.window_size)]
        video_clip = vid_mmap[frame_indices]
        traj_clip  = traj_mmap[frame_indices].copy()
        
        # Resize video frames to self.img_size
        resized_frames = []
        for frame in video_clip:
            if frame.shape[0] == 3: 
                # If channel-first (C, H, W), transpose to (H, W, C) for cv2
                frame = np.transpose(frame, (1, 2, 0))
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                # Transpose back to (C, H, W)
                frame = np.transpose(frame, (2, 0, 1))
            else:
                # If grayscale or HWC already (check needed usually, but assuming HWC if not 3 first)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
            resized_frames.append(frame)
        video_clip = np.array(resized_frames)
        
        # Convert to Tensor
        video_tensor = torch.from_numpy(video_clip).float()
        
        # Ensure (C, H, W) format for PyTorch Conv2D
        if video_tensor.shape[-1] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        # Normalize Video to [-1, 1] (Assuming input was 0-255)
        video_tensor = (video_tensor / 127.5) - 1.0
        
        # --- ACTION EXTRACTION (Continuous Regression) ---
        # 1. Get Global Poses (Pos + Quat)
        # traj_clip: (T, 7) -> [x, y, z, qx, qy, qz, qw]
        positions = traj_clip[:, :3]
        quats = traj_clip[:, 3:]
        
        # --- MAP EXTRACTION ---
        # Query map points ONCE at T=0 (first frame of the window)
        # This defines the "ball" of candidates that the agent sees at the start of the window
        p0 = positions[0]
        q0 = quats[0]
        rot_0 = R.from_quat(q0)
        inv_rot_0 = rot_0.inv()
        
        # 1. Query Global Points (All centerline points within radius)
        if self.map_tree is not None:
            # Use query_ball_point to get ALL points within radius
            ball_indices = self.map_tree.query_ball_point(p0, r=MAP_QUERY_RADIUS)
            
            if len(ball_indices) > 0:
                # Get the fixed set of global points for this window
                raw_neighbors = self.map_points[ball_indices] # (M, 3)
                
                # Apply DBSCAN Filtering
                # Important: removes points from disconnected parallel airways
                window_map_points_global, _ = filter_connected_component(p0, raw_neighbors)
            else:
                window_map_points_global = np.zeros((0, 3), dtype=np.float32)
        else:
            window_map_points_global = np.zeros((0, 3), dtype=np.float32)

        # 2. Downsample Map Points for Model Input
        # Use FPS to reduce candidates while maintaining good spatial coverage
        # This makes training faster while still covering the full trajectory
        if len(window_map_points_global) > self.max_map_points:
            # Find closest point to p0 to use as FPS starting point
            dists = np.linalg.norm(window_map_points_global - p0, axis=1)
            start_idx = np.argmin(dists)
            # Downsample using Farthest Point Sampling
            model_map_points_global, _ = farthest_point_sample(
                window_map_points_global, self.max_map_points, start_idx=start_idx
            )
        else:
            model_map_points_global = window_map_points_global

        # 3. Transform Map Points to Local Frame of Start (T=0)
        # The map input is CONSTANT for the whole window (relative to p0)
        # This gives the model a fixed "local map" context.
        num_valid_points = 0
        if len(model_map_points_global) > 0:
            # Vector from camera 0 to point
            rel_vecs = model_map_points_global - p0
            
            # Rotate by R0^T (to align with camera 0 orientation)
            local_points = inv_rot_0.apply(rel_vecs) # (M, 3)
            
            # Pad to max_map_points
            out_points = np.zeros((self.max_map_points, 3), dtype=np.float32)
            num_valid_points = len(local_points)
            out_points[:num_valid_points] = local_points
            
            # Normalize map points to [-1, 1] range
            out_points = out_points / NORM_MAP_SCALE
            
            # Repeat for window_size frames (since map is static relative to window start)
            map_points_tensor = torch.from_numpy(out_points).float().unsqueeze(0).repeat(self.window_size, 1, 1) # (T, K, 3)
        else:
            map_points_tensor = torch.zeros(self.window_size, self.max_map_points, 3)
        
        # Create mask for valid points (1 = valid, 0 = padding)
        map_mask = torch.zeros(self.max_map_points, dtype=torch.bool)
        map_mask[:num_valid_points] = True
        # Repeat mask for all frames
        map_mask_tensor = map_mask.unsqueeze(0).repeat(self.window_size, 1)  # (T, K)

        # 4. Compute Actions (Targets)
        # Target is the position of the nearest candidate point in the local frame of START (T=0).
        # We want to predict where the true centerline point is for each frame i, 
        # but expressed in the coordinate system of frame 0. This is a sequence-to-sequence regression task.
        actions = []
        for i in range(self.window_size):
            p_curr = positions[i]
            
            # Find nearest neighbor to current position from the DOWNSAMPLED map points.
            # IMPORTANT: Use model_map_points_global (what the model sees), not window_map_points_global!
            # This ensures the GT is always within the convex hull of the model's candidates.
            if len(model_map_points_global) > 0:
                dists = np.linalg.norm(model_map_points_global - p_curr, axis=1)
                nn_idx = np.argmin(dists)
                target_global = model_map_points_global[nn_idx]
            else:
                target_global = p_curr # Fallback to identity (learn nothing) if no map
            
            # Transform this global target into the Local Frame of START (T=0)
            # Vector from Camera 0 to Target
            target_local = inv_rot_0.apply(target_global - p0)
            
            # Normalize to match the map points (which are also normalized)
            target_local = target_local / NORM_MAP_SCALE
            
            # We construct a 6D action vector (3 pos + 3 rot), but currently only use position.
            # Zeros for rotation placeholders.
            action = np.concatenate([target_local, np.zeros(3)])
            actions.append(action)
            
        action_tensor = torch.tensor(np.array(actions, dtype=np.float32)).float() # (T, 6)

        return {
            "video": video_tensor,
            "actions": action_tensor, # (T, 6) - local frame of T=0
            "map_points": map_points_tensor, # (T, K, 3) - local frame of T=0
            "map_mask": map_mask_tensor,  # (T, K) - True for valid points
            # For visualization: first frame's global pose to transform back
            "first_frame_pos": torch.from_numpy(p0.astype(np.float32)),  # (3,)
            "first_frame_quat": torch.from_numpy(q0.astype(np.float32))  # (4,)
        }

if __name__ == "__main__":
    import argparse
    import pyvista as pv
    
    parser = argparse.ArgumentParser(description="Debug visualization for AntDataset windows")
    parser.add_argument('--data_root', type=str, default='./dataset/sequences')
    parser.add_argument('--seq_filter', type=str, default=None, help="Filter sequence by name (substring match)")
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--frame_skip', type=int, default=10)
    parser.add_argument('--lung_path', type=str, default='./patient/lungs.obj')
    parser.add_argument('--stats', action='store_true', help="Run dataset statistics instead of visualization")
    args = parser.parse_args()
    
    if args.stats:
        get_stats(args.data_root)
    else:
        # Debug visualization mode
        print("[DEBUG] Visualizing windows for dataset sequences...")
        
        # Find sequences
        seq_dirs = sorted(glob.glob(os.path.join(args.data_root, "seq_*")))
        if args.seq_filter:
            seq_dirs = [s for s in seq_dirs if args.seq_filter in os.path.basename(s)]
        
        if not seq_dirs:
            print(f"[ERROR] No sequences found in {args.data_root}")
            exit(1)
            
        print(f"[INFO] Found {len(seq_dirs)} sequence(s)")
        
        # Load centerline once
        parent_dir = os.path.dirname(args.data_root)
        graph_path = os.path.join(parent_dir, "static", "centerline.npz")
        centerline_pts = load_centerline_points(graph_path)
        
        # Load lung mesh once
        lung_mesh = None
        if os.path.exists(args.lung_path):
            lung_mesh = pv.read(args.lung_path)
        
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            traj_path = os.path.join(seq_dir, "trajectory.npy")
            
            if not os.path.exists(traj_path):
                continue
                
            # Load trajectory
            traj = np.load(traj_path)
            positions = traj[:, :3]
            
            # Calculate windows for this sequence
            effective_len = (args.window_size - 1) * args.frame_skip + 1
            windows = []
            for start_idx in range(0, len(positions) - effective_len + 1, effective_len):
                frame_indices = [start_idx + i * args.frame_skip for i in range(args.window_size)]
                windows.append(frame_indices)
            
            if not windows:
                print(f"[SKIP] {seq_name}: Too short for any windows")
                continue
                
            print(f"\n[INFO] {seq_name}: {len(windows)} windows, {len(positions)} frames")
            
            # Create plotter
            p = pv.Plotter(title=f"Windows: {seq_name}")
            
            # Draw lungs
            if lung_mesh:
                p.add_mesh(lung_mesh, color='wheat', opacity=0.15, label='Lungs')
            
            # Draw full centerline (very faint)
            if centerline_pts is not None:
                p.add_mesh(pv.PolyData(centerline_pts), color='black', opacity=0.1, 
                          point_size=2, render_points_as_spheres=True, label='Centerline')
            
            # Draw full trajectory
            if len(positions) > 1:
                traj_line = pv.lines_from_points(positions)
                p.add_mesh(traj_line, color='blue', opacity=0.3, line_width=2, label='Full Trajectory')
            
            # Draw each window with different colors
            colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink']
            
            for w_idx, frame_indices in enumerate(windows):
                color = colors[w_idx % len(colors)]
                window_positions = positions[frame_indices]
                p0 = window_positions[0]
                
                # Draw ball wireframe for this window
                sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=p0, theta_resolution=12, phi_resolution=12)
                p.add_mesh(sphere, style='wireframe', color=color, opacity=0.3)
                
                # Draw window path
                if len(window_positions) > 1:
                    window_line = pv.lines_from_points(window_positions)
                    p.add_mesh(window_line, color=color, line_width=3)
                
                # Draw window points
                p.add_mesh(pv.PolyData(window_positions), color=color, point_size=8, 
                          render_points_as_spheres=True)
                
                # Add label at start of window
                p.add_point_labels([p0], [f"W{w_idx}"], point_size=6, 
                                   text_color=color, always_visible=True, font_size=12)
                
                # Print window info
                print(f"  Window {w_idx}: frames {frame_indices[0]}-{frame_indices[-1]}, "
                      f"start pos [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}]")
            
            # Camera setup
            center = np.mean(positions, axis=0)
            p.camera.position = (center[0], center[1] - 150, center[2] + 50)
            p.camera.focal_point = center
            p.camera.up = (0, 0, 1)
            
            p.add_legend()
            p.add_axes()
            p.show()