"""
Utility functions shared across BronchoLoc scripts.
"""
import os
import numpy as np
from sklearn.cluster import DBSCAN
from constants import CONNECTIVITY_THRESHOLD


def load_centerline_points(graph_path):
    """
    Loads centerline points from a graph .npz file.
    
    Args:
        graph_path: Path to the .npz file containing centerline data.
        
    Returns:
        np.array: (N, 3) array of centerline points, or None if not found.
    """
    if not os.path.exists(graph_path):
        print(f"[WARNING] Graph file not found at {graph_path}")
        return None
        
    gdata = np.load(graph_path)
    
    if 'centerline_points' in gdata:
        return gdata['centerline_points']
    elif 'node_pos' in gdata:
        return gdata['node_pos']
    else:
        print("[WARNING] No centerline points found in graph file.")
        return None


def filter_connected_component(center_point, neighbors):
    """
    Filters neighbors to keep only those in the same cluster as the center_point.
    Uses DBSCAN for density-based clustering.
    
    Args:
        center_point: (3,) array - the reference point (e.g., camera position).
        neighbors: (N, 3) array - candidate map points within query radius.
        
    Returns:
        tuple: (connected_neighbors, connected_indices)
    """
    if len(neighbors) == 0:
        return np.array([]), np.array([])
        
    # Run DBSCAN clustering
    clustering = DBSCAN(eps=CONNECTIVITY_THRESHOLD, min_samples=1).fit(neighbors)
    labels = clustering.labels_
    
    # Find label of the point closest to center
    dists = np.linalg.norm(neighbors - center_point, axis=1)
    center_idx = np.argmin(dists)
    center_label = labels[center_idx]
    
    # Select points with the same label
    mask = (labels == center_label)
    connected_indices = np.where(mask)[0]
    
    return neighbors[connected_indices], connected_indices


def farthest_point_sample(points, num_points, start_idx=0):
    """
    Farthest Point Sampling (FPS) for downsampling while maintaining good coverage.
    Iteratively selects the point farthest from already selected points.
    
    Args:
        points: (N, 3) array of 3D points
        num_points: Number of points to keep
        start_idx: Index of starting point (0 = first point, useful to ensure center is included)
        
    Returns:
        tuple: (sampled_points, sampled_indices)
    """
    N = len(points)
    if num_points >= N:
        return points, np.arange(N)
    
    # Start with the specified point (usually the one closest to camera)
    selected_idx = [start_idx]
    distances = np.full(N, np.inf)
    
    for _ in range(num_points - 1):
        # Update distances to nearest selected point
        last_selected = points[selected_idx[-1]]
        new_distances = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select point with maximum distance to any selected point
        next_idx = np.argmax(distances)
        selected_idx.append(next_idx)
    
    selected_idx = np.array(selected_idx)
    return points[selected_idx], selected_idx