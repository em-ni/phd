import os
import argparse
import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from constants import MAP_QUERY_RADIUS

# Heuristic threshold for connectivity
# Increased to 2.0mm to be robust against gaps in the centerline point cloud
CONNECTIVITY_THRESHOLD = 2.0 

def filter_connected_component(center_point, neighbors):
    """
    Filters neighbors to keep only those in the same cluster as the center_point.
    Uses DBSCAN.
    """
    if len(neighbors) == 0:
        return np.array([])
        
    # 1. Run DBSCAN on all points (neighbors)
    # We include the center point implicitly by finding the closest neighbor to it
    # (or we could append it, but let's stick to the existing neighbors list)
    
    clustering = DBSCAN(eps=CONNECTIVITY_THRESHOLD, min_samples=1).fit(neighbors)
    labels = clustering.labels_
    
    # 2. Find label of the center point (or closest point to it)
    dists = np.linalg.norm(neighbors - center_point, axis=1)
    center_idx = np.argmin(dists)
    center_label = labels[center_idx]
    
    # 3. Select points with the same label
    mask = (labels == center_label)
    connected_indices = np.where(mask)[0]
    
    return neighbors[connected_indices], connected_indices

def visualize_ball(args):
    # 1. Load Lungs (Optional Context)
    lung_mesh = None
    if os.path.exists(args.lung_obj):
        print(f"[INFO] Loading Lungs: {args.lung_obj}")
        lung_mesh = pv.read(args.lung_obj)
    else:
        print(f"[WARNING] Lung object not found at {args.lung_obj}")

    # 2. Load Centerline
    if not os.path.exists(args.graph_path):
        print(f"[ERROR] Graph not found at {args.graph_path}")
        return

    print(f"[INFO] Loading Graph: {args.graph_path}")
    gdata = np.load(args.graph_path)
    
    if 'centerline_points' in gdata:
        points = gdata['centerline_points']
    elif 'node_pos' in gdata:
        points = gdata['node_pos']
    else:
        print("[ERROR] No points found in .npz file")
        return
        
    print(f"      Total Points: {len(points)}")
    
    # 3. Pick Random Point
    idx = np.random.randint(0, len(points))
    center_point = points[idx]
    print(f"      Selected Point [{idx}]: {center_point}")
    
    # 4. Find Neighbors
    tree = cKDTree(points)
    indices = tree.query_ball_point(center_point, r=MAP_QUERY_RADIUS)
    neighbors = points[indices]
    print(f"      Neighbors found (Raw): {len(neighbors)}")
    
    # 5. Filter Connected Component
    connected_neighbors, visited_indices = filter_connected_component(center_point, neighbors)
    print(f"      Connected Neighbors: {len(connected_neighbors)}")
    
    # Identify disconnected for viz
    mask = np.ones(len(neighbors), dtype=bool)
    mask[visited_indices] = False
    disconnected_neighbors = neighbors[mask]
    
    # 6. Visualize
    p = pv.Plotter(title="Centerline Ball Visualization")
    
    # Draw Lungs
    if lung_mesh:
        p.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
        
    # Draw Full Centerline (Faint)
    p.add_mesh(pv.PolyData(points), color='black', opacity=0.2, point_size=3, render_points_as_spheres=True, label='Full Centerline')
    
    # Draw Selected Point (Green)
    p.add_mesh(pv.Sphere(radius=2.0, center=center_point), color='green', label='Selected Point')
    
    # Draw Sphere (Wireframe)
    sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=center_point, theta_resolution=20, phi_resolution=20)
    p.add_mesh(sphere, style='wireframe', color='gray', opacity=0.5, label=f'Radius {MAP_QUERY_RADIUS}mm')
    
    # Draw Connected Neighbors (Red)
    if len(connected_neighbors) > 0:
        p.add_mesh(pv.PolyData(connected_neighbors), color='red', point_size=8, render_points_as_spheres=True, label='Connected')

    # Draw Disconnected Neighbors (Blue)
    if len(disconnected_neighbors) > 0:
        p.add_mesh(pv.PolyData(disconnected_neighbors), color='blue', point_size=6, render_points_as_spheres=True, label='Disconnected')
    
    # Camera
    p.camera.position = (center_point[0], center_point[1] - 150, center_point[2] + 30)
    p.camera.focal_point = center_point
    p.camera.up = (0, 0, 1)
    p.camera.zoom(1.0)
    
    p.add_legend()
    p.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, default='./dataset/static/deep_lung_graph.npz')
    parser.add_argument('--lung_obj', type=str, default='./patient/lungs.obj')
    args = parser.parse_args()
    
    visualize_ball(args)
