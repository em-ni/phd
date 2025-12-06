import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from constants import MAP_QUERY_RADIUS, NORM_MAP_SCALE

# Heuristic threshold for connectivity
# This threshold is used by DBSCAN to determine if two points belong to the same cluster.
# Points closer than 2.0 units (likely mm) are considered connected.
CONNECTIVITY_THRESHOLD = 2.0 

def filter_connected_component(center_point, neighbors):
    """
    Filters neighbors to keep only those in the same cluster as the center_point.
    This is crucial for identifying the correct bronchial branch in a dense airway tree.
    Uses DBSCAN for density-based clustering.
    
    Args:
        center_point (np.array): The reference point (current camera position), shape (3,).
        neighbors (np.array): Array of candidate map points within a query radius, shape (N, 3).
        
    Returns:
        np.array: A subset of 'neighbors' that belong to the same connected component as 'center_point'.
    """
    if len(neighbors) == 0:
        return np.array([])
        
    # 1. Run DBSCAN on all points (neighbors)
    # DBSCAN groups points that are closely packed together (points with many nearby neighbors).
    # eps=CONNECTIVITY_THRESHOLD defines the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples=1 ensures every point is part of a cluster (no noise points by definition here, though usually outliers are -1).
    clustering = DBSCAN(eps=CONNECTIVITY_THRESHOLD, min_samples=1).fit(neighbors)
    labels = clustering.labels_
    
    # 2. Find label of the center point (or closest point to it)
    # Since 'center_point' might not be exactly in 'neighbors' (it's the query point), 
    # we find the neighbor closest to the center_point to determine which cluster the camera is "inside".
    dists = np.linalg.norm(neighbors - center_point, axis=1)
    center_idx = np.argmin(dists)
    center_label = labels[center_idx]
    
    # 3. Select points with the same label
    # We filter out all points that belong to different disjoint clusters (e.g., adjacent airway branches that are close in Euclidean space but not connected).
    mask = (labels == center_label)
    connected_indices = np.where(mask)[0]
    
    return neighbors[connected_indices]

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

class DeepLungDataset(Dataset):
    """
    PyTorch Dataset for the Deep-Lung-ST model.
    Handles loading video frames, trajectories, and static airway maps, 
    and generating training samples consisting of video clips, map points, and target actions.
    """
    def __init__(self, data_root, t_frames=16, mode='train', stride=1, map_points_k=32, img_size=128):
        """
        Args:
            data_root (str): Path to the directory containing sequence folders.
            t_frames (int): Number of temporal frames in one sample sequence (window size).
            mode (str): 'train' or 'test'. Used for logging.
            stride (int): Step size for sampling windows (augmentation/downsampling).
            map_points_k (int): Number of nearest map points to retrieve for the graph.
            img_size (int): Spatial resolution to resize video frames to (img_size x img_size).
        """
        self.data_root = data_root
        self.t_frames = t_frames
        self.mode = mode
        self.stride = stride
        self.map_points_k = map_points_k
        self.img_size = img_size
        self.samples = []
        
        # --- LOAD MAP (CENTERLINE) ---
        # Assume static data is in parent of 'sequences' or 'test'
        # data_root is usually .../dataset/sequences
        parent_dir = os.path.dirname(self.data_root)
        graph_path = os.path.join(parent_dir, "static", "deep_lung_graph.npz")
        
        self.map_tree = None
        self.map_points = None
        
        if os.path.exists(graph_path):
            print(f"[DATASET] Loading Centerline Map from {graph_path}")
            gdata = np.load(graph_path)
            # node_pos: (N_nodes, 3) - Represents the 3D coordinates of the airway centerline graph nodes.
            # We use these discrete points as the "map" that the model attends to.
            # We can also use edge points for denser map if available, 
            # but node_pos is a good start. 
            self.map_points = gdata['node_pos']
            # Build a KDTree for fast spatial queries (finding nearest map points).
            self.map_tree = cKDTree(self.map_points)
        else:
            print(f"[WARNING] Map file not found at {graph_path}. Map inputs will be zero.")

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
                
                # Create sliding windows of length t_frames
                if N_vid >= self.t_frames:
                    for start_idx in range(0, N_vid - self.t_frames + 1, self.stride):
                        self.samples.append((vid_path, traj_path, start_idx))
            except Exception as e:
                print(f"Error: {e}")

        print(f"[DATASET] Found {len(self.samples)} valid windows.")

    def __len__(self):
        return len(self.samples)

    def _get_local_map(self, current_pos, current_quat):
        """
        Finds nearest map points and transforms them to local camera frame.
        NOTE: This helper function seems unused in __getitem__ below, which implements similar logic inline 
        for the whole window relative to T=0. It might be a leftover or utility.
        
        Args:
            current_pos (np.array): Camera position (3,)
            current_quat (np.array): Camera orientation quaternion (4,) [x, y, z, w]
            
        Returns:
            torch.Tensor: (K, 3) normalized local map points.
        """
        K = self.map_points_k
        
        if self.map_tree is None:
            return torch.zeros(K, 3)
            
        # 1. Query Nearest Neighbors
        # Find K nearest points in the global map within radius MAP_QUERY_RADIUS
        dists, indices = self.map_tree.query(current_pos, k=K, distance_upper_bound=MAP_QUERY_RADIUS)
        
        # Handle cases where fewer than K points found (indices will be N_points for invalid slots)
        valid_mask = indices < len(self.map_points)
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            return torch.zeros(K, 3)
            
        local_points_global = self.map_points[valid_indices] # (M, 3)
        
        # 2. Transform to Local Frame
        # We want points expressed relative to the camera.
        # T_global = T_camera * T_local
        # P_global = R_cam * P_local + t_cam
        # P_local = R_cam^T * (P_global - t_cam)
        
        rot_cam = R.from_quat(current_quat)
        inv_rot_cam = rot_cam.inv()
        
        # Vector from camera to point (Global Frame translation)
        rel_vecs = local_points_global - current_pos
        
        # Rotate into camera frame
        local_points = inv_rot_cam.apply(rel_vecs) # (M, 3)
        
        # 3. Pad or Truncate
        out_points = np.zeros((K, 3), dtype=np.float32)
        M = min(len(local_points), K)
        out_points[:M] = local_points[:M]
        
        # Normalize to [-1, 1] for neural network input stability
        out_points = out_points / NORM_MAP_SCALE
        
        return torch.from_numpy(out_points).float()

    def __getitem__(self, idx):
        """
        Loads a single sequence window (video + trajectory + map).
        """
        vid_path, traj_path, start_idx = self.samples[idx]
        
        # Load data using mmap to save memory
        vid_mmap = np.load(vid_path, mmap_mode='r')
        traj_mmap = np.load(traj_path, mmap_mode='r')
        end_idx = start_idx + self.t_frames
        
        # Slice the window
        video_clip = vid_mmap[start_idx : end_idx] 
        traj_clip  = traj_mmap[start_idx : end_idx].copy()
        
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
        # Query map points ONCE at T=0
        # This defines the "ball" of candidates that the agent sees at the start of the window.
        # We assume the local map context doesn't change drastically over the short window,
        # or we want to predict relative motion within this initial reference frame.
        p0 = positions[0]
        q0 = quats[0]
        rot_0 = R.from_quat(q0)
        inv_rot_0 = rot_0.inv()
        
        # 1. Query Global Points (Indices)
        if self.map_tree is not None:
            # Radius search for candidates
            dists, indices = self.map_tree.query(p0, k=self.map_points_k, distance_upper_bound=MAP_QUERY_RADIUS)
            
            # Handle valid indices
            valid_mask = indices < len(self.map_points)
            valid_indices = indices[valid_mask]
            
            if len(valid_indices) > 0:
                # Get the fixed set of global points for this window
                raw_neighbors = self.map_points[valid_indices] # (M, 3)
                
                # Apply DBSCAN Filtering
                # Important: removes points from disconnected parallel airways
                window_map_points_global = filter_connected_component(p0, raw_neighbors)
            else:
                window_map_points_global = np.zeros((0, 3), dtype=np.float32)
        else:
            window_map_points_global = np.zeros((0, 3), dtype=np.float32)

        # 2. Transform Map Points to Local Frame of Start (T=0)
        # The map input is CONSTANT for the whole window (relative to p0)
        # This gives the model a fixed "local map" context.
        if len(window_map_points_global) > 0:
            # Vector from camera 0 to point
            rel_vecs = window_map_points_global - p0
            
            # Rotate by R0^T (to align with camera 0 orientation)
            local_points = inv_rot_0.apply(rel_vecs) # (M, 3)
            
            # Pad or Truncate to K
            out_points = np.zeros((self.map_points_k, 3), dtype=np.float32)
            M = min(len(local_points), self.map_points_k)
            out_points[:M] = local_points[:M]
            
            # Normalize map points to [-1, 1] range
            out_points = out_points / NORM_MAP_SCALE
            
            # Repeat for T frames (since map is static relative to window start)
            # The model receives the same map context at every timestep because it's estimating 
            # the trajectory WITHIN this context.
            map_points_tensor = torch.from_numpy(out_points).float().unsqueeze(0).repeat(self.t_frames, 1, 1) # (T, K, 3)
        else:
            map_points_tensor = torch.zeros(self.t_frames, self.map_points_k, 3)

        # 4. Compute Actions (Targets)
        # Target is the position of the nearest candidate point in the local frame of START (T=0).
        # We want to predict where the true centerline point is for each frame i, 
        # but expressed in the coordinate system of frame 0. This is a sequence-to-sequence regression task.
        actions = []
        for i in range(self.t_frames):
            p_curr = positions[i]
            
            # Find nearest neighbor to current position from the fixed window map points.
            # This represents the "correct" node we should be at or moving towards.
            if len(window_map_points_global) > 0:
                dists = np.linalg.norm(window_map_points_global - p_curr, axis=1)
                nn_idx = np.argmin(dists)
                target_global = window_map_points_global[nn_idx]
            else:
                target_global = p_curr # Fallback to identity (learn nothing) if no map
            
            # Transform this global target into the Local Frame of START (T=0)
            # Vector from Camera 0 to Target
            target_local = inv_rot_0.apply(target_global - p0)
            
            # We construct a 6D action vector (3 pos + 3 rot), but currently only use position.
            # Zeros for rotation placeholders.
            action = np.concatenate([target_local, np.zeros(3)])
            actions.append(action)
            
        action_tensor = torch.tensor(np.array(actions, dtype=np.float32)).float() # (T, 6)

        return {
            "video": video_tensor,
            "actions": action_tensor, # (T, 6)
            "map_points": map_points_tensor # (T, K, 3)
        }

if __name__ == "__main__":
    # If run as a script, compute dataset stats
    get_stats("./dataset/sequences")