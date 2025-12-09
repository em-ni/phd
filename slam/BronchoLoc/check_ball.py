import os
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from constants import MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS
from utils import filter_connected_component, load_centerline_points, farthest_point_sample


def visualize_ball(args):
    # 1. Load Lungs (Optional Context)
    lung_mesh = None
    if os.path.exists(args.lung_obj):
        print(f"[INFO] Loading Lungs: {args.lung_obj}")
        lung_mesh = pv.read(args.lung_obj)
    else:
        print(f"[WARNING] Lung object not found at {args.lung_obj}")

    # 2. Load Centerline
    points = load_centerline_points(args.graph_path)
    if points is None:
        print(f"[ERROR] Could not load centerline from {args.graph_path}")
        return
        
    print(f"[INFO] Loaded {len(points)} centerline points")
    
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
    
    # 6. Apply FPS Downsampling (same as dataset)
    if len(connected_neighbors) > args.max_points:
        dists = np.linalg.norm(connected_neighbors - center_point, axis=1)
        start_idx = np.argmin(dists)
        fps_points, fps_indices = farthest_point_sample(
            connected_neighbors, args.max_points, start_idx=start_idx
        )
        print(f"      FPS Downsampled: {len(fps_points)} points (from {len(connected_neighbors)})")
    else:
        fps_points = connected_neighbors
        fps_indices = np.arange(len(connected_neighbors))
        print(f"      No FPS needed (only {len(connected_neighbors)} points)")
    
    # 7. Visualize
    p = pv.Plotter(title="Centerline Ball + FPS Visualization")
    
    # Draw Lungs
    if lung_mesh:
        p.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
        
    # Draw Full Centerline (Faint)
    p.add_mesh(pv.PolyData(points), color='black', opacity=0.2, point_size=3, render_points_as_spheres=True, label='Full Centerline')
    
    # Draw Selected Point (Green)
    p.add_mesh(pv.Sphere(radius=2.0, center=center_point), color='green', label='Query Center')
    
    # Draw Sphere (Wireframe)
    sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=center_point, theta_resolution=20, phi_resolution=20)
    p.add_mesh(sphere, style='wireframe', color='gray', opacity=0.5, label=f'Radius {MAP_QUERY_RADIUS}mm')
    
    # Draw Connected Neighbors (Faint - these are filtered but before FPS)
    if len(connected_neighbors) > 0:
        p.add_mesh(pv.PolyData(connected_neighbors), color='orange', opacity=0.4, point_size=5, 
                  render_points_as_spheres=True, label=f'Connected ({len(connected_neighbors)})')
    
    # Draw FPS Downsampled Points (Bright Red - what model sees)
    if len(fps_points) > 0:
        p.add_mesh(pv.PolyData(fps_points), color='red', point_size=10, 
                  render_points_as_spheres=True, label=f'FPS Model Input ({len(fps_points)})')

    # Draw Disconnected Neighbors (Blue)
    if len(disconnected_neighbors) > 0:
        p.add_mesh(pv.PolyData(disconnected_neighbors), color='blue', point_size=4, opacity=0.3,
                  render_points_as_spheres=True, label=f'Disconnected ({len(disconnected_neighbors)})')
    
    # Camera
    p.camera.position = (center_point[0], center_point[1] - 150, center_point[2] + 30)
    p.camera.focal_point = center_point
    p.camera.up = (0, 0, 1)
    p.camera.zoom(1.0)
    
    p.add_legend()
    p.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, default='./dataset/static/centerline.npz')
    parser.add_argument('--lung_obj', type=str, default='./patient/lungs.obj')
    parser.add_argument('--max_points', type=int, default=DEFAULT_MAX_MAP_POINTS,
                       help='Max points after FPS (same as dataset max_map_points)')
    args = parser.parse_args()
    
    visualize_ball(args)
