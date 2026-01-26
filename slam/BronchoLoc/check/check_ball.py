import os
import sys
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import MAP_QUERY_RADIUS, DEFAULT_MAX_MAP_POINTS, MAP_POINT_SPACING
from utils.utils import filter_connected_component, load_centerline_points, density_based_sample

# temp
MAP_QUERY_RADIUS = 15

def visualize_ball(args):
    # 1. Load Lungs (Optional Context)
    lung_mesh = None
    if os.path.exists(args.lung_obj):
        print(f"[INFO] Loading Lungs: {args.lung_obj}")
        lung_mesh = pv.read(args.lung_obj)
    else:
        print(f"[WARNING] Lung object not found at {args.lung_obj}")

    # 2. Load Centerline
    points = load_centerline_points(args.centerline_path)
    if points is None:
        print(f"[ERROR] Could not load centerline from {args.centerline_path}")
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
    
    # 6. Apply Density-based Downsampling (same as dataset)
    if len(connected_neighbors) > 0:
        dists = np.linalg.norm(connected_neighbors - center_point, axis=1)
        start_idx = np.argmin(dists)
        fps_points, fps_indices = density_based_sample(
            connected_neighbors, 
            min_distance=MAP_POINT_SPACING, 
            start_idx=start_idx,
            max_points=args.max_points
        )
        print(f"      Density-based Downsampled: {len(fps_points)} points (spacing={MAP_POINT_SPACING}mm)")
    else:
        fps_points = connected_neighbors
        fps_indices = np.arange(len(connected_neighbors))
        print(f"      No downsampling needed (0 points)")
    
    # 7. Visualize
    p = pv.Plotter(title="Centerline Ball + FPS Visualization")
    
    # Draw Lungs
    if lung_mesh:
        p.add_mesh(lung_mesh, color='wheat', opacity=0.1, label='Lungs')
        
    # Draw Full Centerline (Faint)
    p.add_mesh(pv.PolyData(points), color='black', opacity=1, point_size=3, render_points_as_spheres=True, label='Full Centerline')
    
    # Draw Selected Point (Green)
    p.add_mesh(pv.Sphere(radius=1.0, center=center_point), color='red', label='Query Center')
    
    # Draw Sphere (Wireframe)
    sphere = pv.Sphere(radius=MAP_QUERY_RADIUS, center=center_point, theta_resolution=20, phi_resolution=20)
    p.add_mesh(sphere, style='wireframe', color='gray', opacity=0.5, label=f'Radius {MAP_QUERY_RADIUS}mm')
    
    # Draw Connected Neighbors (Faint - these are filtered but before FPS)
    if len(connected_neighbors) > 0:
        p.add_mesh(pv.PolyData(connected_neighbors), color='light blue', opacity=0.4, point_size=5, 
                  render_points_as_spheres=True, label=f'Connected ({len(connected_neighbors)})')
    
    # Draw Density-based Downsampled Points (Bright Red - what model sees)
    if len(fps_points) > 0:
        p.add_mesh(pv.PolyData(fps_points), color='green', point_size=10, 
                  render_points_as_spheres=True, label=f'Density Sample ({len(fps_points)})')

    # Draw Disconnected Neighbors (Blue)
    if len(disconnected_neighbors) > 0:
        p.add_mesh(pv.PolyData(disconnected_neighbors), color='blue', point_size=6, opacity=0.3,
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
    parser.add_argument('--centerline_path', type=str, default='../dataset/static/centerline.npz')
    parser.add_argument('--lung_obj', type=str, default='../patient/lungs.obj')
    parser.add_argument('--max_points', type=int, default=DEFAULT_MAX_MAP_POINTS,
                       help='Max points after FPS (same as dataset max_map_points)')
    args = parser.parse_args()
    
    visualize_ball(args)
