"""
Utility functions shared across BronchoLoc scripts.
"""
import os
import numpy as np
from sklearn.cluster import DBSCAN
from constants import CONNECTIVITY_THRESHOLD


def load_centerline_points(centerline_path):
    """
    Loads centerline points from a centerline .npz file.
    
    Args:
        centerline_path: Path to the .npz file containing centerline data.
        
    Returns:
        np.array: (N, 3) array of centerline points, or None if not found.
    """
    if not os.path.exists(centerline_path):
        print(f"[WARNING] Centerline file not found at {centerline_path}")
        return None
        
    gdata = np.load(centerline_path)
    
    if 'centerline_points' in gdata:
        return gdata['centerline_points']
    elif 'node_pos' in gdata:
        return gdata['node_pos']
    else:
        print("[WARNING] No centerline points found in centerline file.")
        return None


def load_centerline_poses(centerline_path):
    """
    Loads centerline with full poses (position + quaternion) from TUM format file.
    TUM format: timestamp x y z qx qy qz qw (space-separated)
    
    Args:
        centerline_path: Path to .txt (TUM) file containing centerline poses.
        
    Returns:
        np.array: (N, 7) array of [x, y, z, qx, qy, qz, qw], or None if not found.
    """
    if not os.path.exists(centerline_path):
        print(f"[WARNING] Centerline file not found at {centerline_path}")
        return None
    
    try:
        data = np.loadtxt(centerline_path)
        # Skip first column (timestamp), take columns 1-7 as pose
        poses = data[:, 1:8].astype(np.float32)
        return poses
    except Exception as e:
        print(f"[WARNING] Failed to load TUM centerline: {e}")
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


def density_based_sample(points, min_distance=2.0, start_idx=0, max_points=None):
    """
    Density-based downsampling: selects points at a constant distance interval.
    Ensures uniform point density regardless of the number of branches in the region.
    
    Uses greedy FPS-like approach but stops when all remaining points are within
    min_distance of already selected points.
    
    Args:
        points: (N, 3) array of 3D points
        min_distance: Minimum distance between selected points (mm). Default 2mm.
        start_idx: Index of starting point (usually closest to camera).
        max_points: Optional cap on number of points (for memory limits).
        
    Returns:
        tuple: (sampled_points, sampled_indices)
    """
    N = len(points)
    if N == 0:
        return np.array([]), np.array([])
    
    # Start with the specified point
    selected_idx = [start_idx]
    distances = np.full(N, np.inf)
    
    while True:
        # Update distances to nearest selected point
        last_selected = points[selected_idx[-1]]
        new_distances = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Find point with maximum distance to any selected point
        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx]
        
        # Stop if all points are within min_distance of selected points
        if max_dist < min_distance:
            break
            
        # Stop if we've reached max_points limit
        if max_points is not None and len(selected_idx) >= max_points:
            break
            
        selected_idx.append(max_dist_idx)
    
    selected_idx = np.array(selected_idx)
    return points[selected_idx], selected_idx


def find_centerline_path(start_point, end_point, centerline_pts, centerline_tree, max_points=100):
    """
    Finds the path along the centerline between two points.
    Uses a greedy approach: starting from start_point, iteratively move to the 
    nearest neighbor that brings us closer to end_point until we reach it.
    
    Args:
        start_point: (3,) array - starting position (predicted point at frame t)
        end_point: (3,) array - ending position (predicted point at frame t+1)
        centerline_pts: (N, 3) array of all centerline points
        centerline_tree: cKDTree built from centerline_pts for fast queries
        max_points: Maximum number of points to include in path (safety limit)
        
    Returns:
        np.array: (M, 3) array of centerline points forming the path (includes start and end)
    """
    if centerline_tree is None or centerline_pts is None:
        # No centerline available, return direct interpolation
        return np.array([start_point, end_point])
    
    # Find nearest centerline points to start and end
    _, start_idx = centerline_tree.query(start_point)
    _, end_idx = centerline_tree.query(end_point)
    
    # If they're the same point, just return it
    if start_idx == end_idx:
        return np.array([centerline_pts[start_idx]])
    
    # Get the actual centerline positions
    start_cl = centerline_pts[start_idx]
    end_cl = centerline_pts[end_idx]
    
    # Greedy path finding: move along centerline towards end_cl
    # Use connectivity threshold to find neighbors
    path = [start_idx]
    visited = {start_idx}
    current_idx = start_idx
    
    # Distance from current point to end
    dist_to_end = np.linalg.norm(centerline_pts[current_idx] - end_cl)
    
    iteration = 0
    while current_idx != end_idx and iteration < max_points:
        iteration += 1
        current_pos = centerline_pts[current_idx]
        
        # Query nearby points (within connectivity threshold)
        neighbor_indices = centerline_tree.query_ball_point(current_pos, r=CONNECTIVITY_THRESHOLD * 1.5)
        
        # Filter out visited points
        unvisited = [idx for idx in neighbor_indices if idx not in visited]
        
        if not unvisited:
            # No unvisited neighbors, we're stuck - try expanding search
            neighbor_indices = centerline_tree.query_ball_point(current_pos, r=CONNECTIVITY_THRESHOLD * 3.0)
            unvisited = [idx for idx in neighbor_indices if idx not in visited]
            
            if not unvisited:
                # Still stuck, break and return what we have
                break
        
        # Find the neighbor that gets us closest to end_cl
        best_idx = None
        best_dist = np.inf
        for idx in unvisited:
            dist = np.linalg.norm(centerline_pts[idx] - end_cl)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        if best_idx is None:
            break
            
        # Move to best neighbor
        path.append(best_idx)
        visited.add(best_idx)
        current_idx = best_idx
        
        # Check if we've reached close enough to end
        if best_dist < CONNECTIVITY_THRESHOLD:
            # Make sure end_idx is included
            if end_idx not in visited:
                path.append(end_idx)
            break
    
    # Convert indices to points
    path_points = centerline_pts[path]
    
    return path_points


def interpolate_trajectory(predictions, centerline_pts, centerline_tree):
    """
    Interpolate between consecutive predictions to create a smooth trajectory
    that follows the centerline.
    
    Args:
        predictions: (T, 3) array of predicted positions
        centerline_pts: (N, 3) array of all centerline points
        centerline_tree: cKDTree built from centerline_pts
        
    Returns:
        np.array: (M, 3) array of interpolated trajectory points
    """
    if len(predictions) < 2:
        return predictions
    
    full_trajectory = []
    
    for i in range(len(predictions) - 1):
        start = predictions[i]
        end = predictions[i + 1]
        
        # Find path along centerline
        path = find_centerline_path(start, end, centerline_pts, centerline_tree)
        
        # Add all points except the last (to avoid duplicates)
        if i == len(predictions) - 2:
            # Last segment: include end point
            full_trajectory.extend(path)
        else:
            # Not last segment: exclude end to avoid duplicate with next segment's start
            full_trajectory.extend(path[:-1] if len(path) > 1 else path)
    
    return np.array(full_trajectory)