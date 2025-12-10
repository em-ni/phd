"""
Script to downsample centerline points for faster training.
Reduces the density of points while preserving the overall structure.
"""
import os
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def voxel_downsample(points, voxel_size):
    """
    Downsample points using voxel grid filtering.
    Each voxel keeps its centroid.
    
    Args:
        points: (N, 3) array of 3D points
        voxel_size: Size of voxel in mm
        
    Returns:
        Downsampled points (M, 3)
    """
    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Create unique voxel keys
    # Use a dictionary to accumulate points in each voxel
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])
    
    # Compute centroid of each voxel
    downsampled = []
    for key, pts in voxel_dict.items():
        centroid = np.mean(pts, axis=0)
        downsampled.append(centroid)
    
    return np.array(downsampled)


def uniform_downsample(points, keep_every_n):
    """
    Simple uniform downsampling - keep every N-th point.
    
    Args:
        points: (N, 3) array of 3D points
        keep_every_n: Keep every N-th point
        
    Returns:
        Downsampled points
    """
    return points[::keep_every_n]


def farthest_point_sampling(points, num_points):
    """
    Farthest Point Sampling (FPS) for better coverage.
    Iteratively selects the point farthest from already selected points.
    
    Args:
        points: (N, 3) array of 3D points
        num_points: Number of points to keep
        
    Returns:
        Downsampled points (num_points, 3)
    """
    N = len(points)
    if num_points >= N:
        return points
    
    # Start with a random point
    selected_idx = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    
    for _ in range(num_points - 1):
        # Update distances to nearest selected point
        last_selected = points[selected_idx[-1]]
        new_distances = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select point with maximum distance to any selected point
        next_idx = np.argmax(distances)
        selected_idx.append(next_idx)
    
    return points[selected_idx]


def main():
    parser = argparse.ArgumentParser(description="Downsample centerline points")
    parser.add_argument('--input', type=str, default='../dataset/static/centerline.npz',
                       help='Input centerline file')
    parser.add_argument('--output', type=str, default='../dataset/static/centerline_downsampled.npz',
                       help='Output centerline file')
    parser.add_argument('--method', type=str, default='voxel', choices=['voxel', 'uniform', 'fps'],
                       help='Downsampling method')
    parser.add_argument('--voxel_size', type=float, default=2.0,
                       help='Voxel size in mm (for voxel method)')
    parser.add_argument('--keep_every_n', type=int, default=3,
                       help='Keep every N-th point (for uniform method)')
    parser.add_argument('--num_points', type=int, default=2000,
                       help='Target number of points (for fps method)')
    parser.add_argument('--lung_path', type=str, default='../patient/lungs.obj',
                       help='Path to lung mesh for visualization')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Visualize before and after')
    parser.add_argument('--save', action='store_true',
                       help='Save the downsampled centerline')
    args = parser.parse_args()
    
    # Load centerline
    print(f"[INFO] Loading centerline from {args.input}")
    data = np.load(args.input)
    
    if 'centerline_points' in data:
        points = data['centerline_points']
    elif 'node_pos' in data:
        points = data['node_pos']
    else:
        print("[ERROR] No centerline points found in file")
        return
    
    print(f"[INFO] Original: {len(points)} points")
    
    # Downsample
    if args.method == 'voxel':
        print(f"[INFO] Voxel downsampling with voxel_size={args.voxel_size}mm")
        downsampled = voxel_downsample(points, args.voxel_size)
    elif args.method == 'uniform':
        print(f"[INFO] Uniform downsampling, keeping every {args.keep_every_n}-th point")
        downsampled = uniform_downsample(points, args.keep_every_n)
    elif args.method == 'fps':
        print(f"[INFO] Farthest Point Sampling to {args.num_points} points")
        downsampled = farthest_point_sampling(points, args.num_points)
    
    print(f"[INFO] Downsampled: {len(downsampled)} points")
    print(f"[INFO] Reduction: {100 * (1 - len(downsampled)/len(points)):.1f}%")
    
    # Compute average nearest neighbor distance
    tree_orig = cKDTree(points)
    tree_down = cKDTree(downsampled)
    
    # Original spacing
    dists_orig, _ = tree_orig.query(points, k=2)
    avg_spacing_orig = np.mean(dists_orig[:, 1])
    
    # Downsampled spacing
    dists_down, _ = tree_down.query(downsampled, k=2)
    avg_spacing_down = np.mean(dists_down[:, 1])
    
    print(f"[INFO] Average spacing - Original: {avg_spacing_orig:.2f}mm, Downsampled: {avg_spacing_down:.2f}mm")
    
    # Save
    if args.save:
        print(f"[INFO] Saving to {args.output}")
        np.savez(args.output, centerline_points=downsampled)
        print("[INFO] Saved successfully!")
    
    # Visualize
    if args.visualize:
        print("[INFO] Visualizing...")
        
        # Load lung mesh
        lung_mesh = None
        if os.path.exists(args.lung_path):
            lung_mesh = pv.read(args.lung_path)
        
        # Create side-by-side comparison
        p = pv.Plotter(shape=(1, 2), title="Centerline Downsampling Comparison")
        
        # Left: Original
        p.subplot(0, 0)
        p.add_title(f"Original ({len(points)} points)")
        if lung_mesh:
            p.add_mesh(lung_mesh, color='wheat', opacity=0.1)
        p.add_mesh(pv.PolyData(points), color='blue', point_size=4, 
                  render_points_as_spheres=True)
        
        # Right: Downsampled
        p.subplot(0, 1)
        p.add_title(f"Downsampled ({len(downsampled)} points)")
        if lung_mesh:
            p.add_mesh(lung_mesh, color='wheat', opacity=0.1)
        p.add_mesh(pv.PolyData(downsampled), color='red', point_size=6, 
                  render_points_as_spheres=True)
        
        p.link_views()
        p.show()


if __name__ == "__main__":
    main()
