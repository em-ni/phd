#!/usr/bin/env python
"""
Centerline Variations Visualization Script
==========================================

Generates centerline variations for all VTP centerlines in the first 10 folders
inside selected_airways (from 1003 to 1356), and displays them in a grid layout.

Usage:
    python visualize_centerline_variations.py
    python visualize_centerline_variations.py --num-variations 3
    python visualize_centerline_variations.py --num-folders 5
"""

import os
import sys
import glob
import math
import argparse
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------------
# Centerline variation functions (adapted from centerline_variations.py)
# ----------------------------------------------------------------------------

def load_centerline(path: str) -> vtk.vtkPolyData:
    """Load centerline from VTP file."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def extract_points(polydata: vtk.vtkPolyData) -> np.ndarray:
    """
    Extract points from VTK polydata as numpy array, following line cell connectivity order.
    
    VTP centerline files store points in arbitrary order, but the LINE CELL
    contains the actual traversal order. We must extract points in that order.
    """
    n_points = polydata.GetNumberOfPoints()
    
    # Check if there are line cells
    lines = polydata.GetLines()
    if lines.GetNumberOfCells() > 0:
        # Get the first line cell (which contains the path order)
        lines.InitTraversal()
        cell = vtk.vtkIdList()
        lines.GetNextCell(cell)
        
        n_cell_points = cell.GetNumberOfIds()
        points = np.zeros((n_cell_points, 3))
        
        for i in range(n_cell_points):
            point_id = cell.GetId(i)
            points[i] = polydata.GetPoint(point_id)
        
        return points
    else:
        # Fallback: no line cells, use raw point order
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            points[i] = polydata.GetPoint(i)
        return points


def save_points_as_vtp(points: np.ndarray, output_path: str):
    """
    Save points as a VTP file with line connectivity.
    """
    n_points = len(points)
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p[0], p[1], p[2])
    
    # Create line connectivity
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(n_points)
    for i in range(n_points):
        line.GetPointIds().SetId(i, i)
    
    # Create cells
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    
    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)
    
    # Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()


def add_smooth_noise(points: np.ndarray, 
                     amplitude: float = 0.05,
                     smoothness: float = 10.0,
                     preserve_endpoints: bool = True,
                     taper_end_ratio: float = 1.0) -> np.ndarray:
    """
    Add smooth noise to trajectory points.
    
    Args:
        points: Nx3 array of 3D points
        amplitude: Noise amplitude as fraction of trajectory length
        smoothness: Higher = smoother noise (gaussian sigma)
        preserve_endpoints: If True, don't modify start and end points
        taper_end_ratio: Amplitude ratio at end vs start (0.3 = 30% of start amplitude at end)
    
    Returns:
        Nx3 array of modified points
    """
    n_points = len(points)
    
    # Calculate trajectory length for scaling
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(lengths)
    
    # Scale amplitude by trajectory size
    noise_scale = amplitude * total_length
    
    # Generate random noise for each dimension
    noise = np.random.randn(n_points, 3)
    
    # Apply Gaussian smoothing to make noise smooth (low-pass filter)
    sigma = max(1, smoothness)
    for dim in range(3):
        noise[:, dim] = gaussian_filter1d(noise[:, dim], sigma)
    
    # Normalize and scale
    noise = noise / np.max(np.abs(noise) + 1e-6) * noise_scale
    
    # Apply position-dependent amplitude tapering (larger at start, smaller at end)
    if taper_end_ratio < 1.0:
        taper = np.linspace(1.0, taper_end_ratio, n_points)
        noise = noise * taper[:, np.newaxis]
    
    # Apply noise
    new_points = points + noise
    
    # Apply final smoothing pass to remove any remaining high-frequency artifacts
    final_smooth_sigma = max(2, smoothness / 3)
    for dim in range(3):
        new_points[:, dim] = gaussian_filter1d(new_points[:, dim], final_smooth_sigma)
    
    # Preserve endpoints if requested
    if preserve_endpoints:
        # Fade noise near endpoints
        fade_length = max(3, n_points // 10)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        for i in range(min(fade_length, n_points)):
            new_points[i] = points[i] * (1 - fade_in[i]) + new_points[i] * fade_in[i]
        
        for i in range(min(fade_length, n_points)):
            idx = n_points - 1 - i
            blend = fade_out[fade_length - 1 - i]
            new_points[idx] = points[idx] * (1 - blend) + new_points[idx] * blend
    
    return new_points


def get_allowed_radius_per_point(points: np.ndarray, mesh: vtk.vtkPolyData, 
                                   safety_factor: float = 0.3,
                                   max_deviation: float = 1.5,
                                   max_distance_threshold: float = 10.0) -> np.ndarray:
    """
    For each point on the centerline, find the distance to the mesh surface.
    This tells us how far we can deviate at each point without hitting the wall.
    
    Args:
        points: Nx3 centerline points
        mesh: VTK mesh of the airway
        safety_factor: Fraction of radius to allow (0.3 = stay within 30% of tube radius)
        max_deviation: Maximum allowed deviation in mm (caps deviation)
        max_distance_threshold: If distance to mesh > this, point is at opening, set deviation to 0
    
    Returns:
        Nx1 array of allowed deviation radius for each point
    """
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    
    allowed_radii = np.zeros(len(points))
    
    for i in range(len(points)):
        pt = points[i]
        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        
        cell_locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
        dist_to_wall = np.sqrt(dist2.get())
        
        # If point is too far from mesh surface, it's at an opening (trachea, branch end)
        # Set allowed deviation to 0 to keep variation exactly on original path
        if dist_to_wall > max_distance_threshold:
            allowed_radii[i] = 0.0
        else:
            # Allow movement up to safety_factor of the distance to wall
            # Cap at max_deviation
            allowed_radii[i] = min(dist_to_wall * safety_factor, max_deviation)
    
    return allowed_radii


def add_constrained_noise(points: np.ndarray, 
                          allowed_radii: np.ndarray,
                          smoothness: float = 15.0,
                          preserve_endpoints: bool = True) -> np.ndarray:
    """
    Add smooth noise constrained by allowed radius at each point.
    """
    n_points = len(points)
    
    # Generate random direction for each point (unit vectors)
    noise_direction = np.random.randn(n_points, 3)
    
    # Apply Gaussian smoothing to make directions smooth
    sigma = max(1, smoothness)
    for dim in range(3):
        noise_direction[:, dim] = gaussian_filter1d(noise_direction[:, dim], sigma)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(noise_direction, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0  # Avoid division by zero
    noise_direction = noise_direction / norms
    
    # Generate random magnitudes (scaled by allowed radius)
    magnitudes = np.random.rand(n_points) * allowed_radii
    
    # Smooth the magnitudes too
    magnitudes = gaussian_filter1d(magnitudes, sigma / 2)
    
    # Apply noise
    new_points = points + noise_direction * magnitudes[:, np.newaxis]
    
    # Preserve endpoints if requested
    if preserve_endpoints:
        fade_length = max(3, n_points // 10)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        for i in range(min(fade_length, n_points)):
            new_points[i] = points[i] * (1 - fade_in[i]) + new_points[i] * fade_in[i]
        
        for i in range(min(fade_length, n_points)):
            idx = n_points - 1 - i
            blend = fade_out[fade_length - 1 - i]
            new_points[idx] = points[idx] * (1 - blend) + new_points[idx] * blend
    
    return new_points


def generate_variations_in_memory(points: np.ndarray, 
                                   num_variations: int = 5,
                                   amplitude: float = 0.03,
                                   smoothness: float = 15.0,
                                   constraint_mesh: vtk.vtkPolyData = None,
                                   max_attempts: int = 20) -> list:
    """
    Generate smooth variations of centerline points.
    
    Uses simple fixed-amplitude smooth noise. No complex mesh constraints.
    The noise is in the PERPENDICULAR direction to the path to keep it realistic.
    
    Returns list of Nx3 numpy arrays, including the original as first element.
    """
    variations = [points.copy()]  # Original first
    n_points = len(points)
    
    if n_points < 3:
        return variations
    
    # Calculate path tangent at each point (direction of the path)
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    for i in range(1, n_points - 1):
        tangents[i] = points[i+1] - points[i-1]
    
    # Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms[tangent_norms < 1e-6] = 1.0
    tangents = tangents / tangent_norms
    
    for var_idx in range(num_variations):
        # Generate smooth random perturbation
        sigma = smoothness * (0.8 + 0.4 * np.random.random())
        
        # Generate random 3D noise
        noise = np.random.randn(n_points, 3)
        
        # Smooth the noise
        for dim in range(3):
            noise[:, dim] = gaussian_filter1d(noise[:, dim], sigma)
        
        # Remove tangential component to keep noise perpendicular to path
        # Project noise onto tangent and subtract
        tangent_component = np.sum(noise * tangents, axis=1, keepdims=True) * tangents
        perpendicular_noise = noise - tangent_component
        
        # Normalize and scale to higher amplitude (3mm at start, tapering to smaller at end)
        max_noise = np.max(np.abs(perpendicular_noise))
        if max_noise > 1e-6:
            perpendicular_noise = perpendicular_noise / max_noise
        
        # Create amplitude taper: larger at start (trachea), smaller at end (branches)
        # Start with 7mm, taper to 0.5mm at the end
        amplitude_taper = np.linspace(7.0, 0.5, n_points)
        perpendicular_noise = perpendicular_noise * amplitude_taper[:, np.newaxis]
        
        # Apply noise
        new_points = points + perpendicular_noise
        
        # Small fade at very start (first 5 points) to avoid discontinuity
        fade_start_length = min(5, n_points)
        fade_in = np.linspace(0.3, 1, fade_start_length)  # Start at 30% not 0%
        for i in range(fade_start_length):
            new_points[i] = points[i] * (1 - fade_in[i]) + new_points[i] * fade_in[i]
        
        # Fade out at end
        fade_end_length = max(3, n_points // 10)
        fade_out = np.linspace(1, 0, fade_end_length)
        for i in range(min(fade_end_length, n_points)):
            idx = n_points - 1 - i
            blend = fade_out[fade_end_length - 1 - i]
            new_points[idx] = points[idx] * (1 - blend) + new_points[idx] * blend
        
        variations.append(new_points)
    
    return variations


# ----------------------------------------------------------------------------
# Visualization functions
# ----------------------------------------------------------------------------

# Nice color palette for trajectories - vivid colors for light background
TRAJECTORY_COLORS = [
    '#c41e3a',  # Cardinal Red
    '#228b22',  # Forest Green
    '#1e90ff',  # Dodger Blue
    '#ff8c00',  # Dark Orange
    '#8b008b',  # Dark Magenta
    '#008b8b',  # Dark Cyan
    '#b8860b',  # Dark Goldenrod
    '#4b0082',  # Indigo
    '#2e8b57',  # Sea Green
    '#dc143c',  # Crimson
    '#00008b',  # Dark Blue
    '#8b4513',  # Saddle Brown
    '#006400',  # Dark Green
    '#9932cc',  # Dark Orchid
    '#ff4500',  # Orange Red
    '#191970',  # Midnight Blue
]


def points_to_pyvista_line(points: np.ndarray) -> pv.PolyData:
    """Convert numpy points array to PyVista line mesh."""
    n_points = len(points)
    # Create line connectivity: 0-1, 1-2, 2-3, ...
    lines = np.zeros((n_points - 1, 3), dtype=int)
    lines[:, 0] = 2  # Each line segment has 2 points
    lines[:, 1] = np.arange(n_points - 1)
    lines[:, 2] = np.arange(1, n_points)
    
    return pv.PolyData(points, lines=lines.flatten())


def visualize_all_folders(selected_airways_dir: str, 
                          num_folders: int = 10, 
                          num_variations: int = 5,
                          amplitude: float = 0.03,
                          smoothness: float = 15.0,
                          save_trajectories: bool = False):
    """
    Visualize centerline variations for all folders in a grid.
    
    Args:
        selected_airways_dir: Path to selected_airways directory
        num_folders: Number of folders to process
        num_variations: Number of variations per centerline
        amplitude: Noise amplitude
        smoothness: Noise smoothness
        save_trajectories: If True, save variations as VTP files
    """
    # Get list of folders (sorted)
    folders = sorted([f for f in os.listdir(selected_airways_dir) 
                     if os.path.isdir(os.path.join(selected_airways_dir, f))])
    
    # Limit to first num_folders
    folders = folders[:num_folders]
    
    print(f"Processing {len(folders)} folders: {folders}")
    print(f"Generating {num_variations} variations per centerline")
    print("-" * 60)
    
    # Calculate grid dimensions - expand vertically first (fewer columns)
    n_plots = len(folders)
    # Use 2 columns for up to 6 items, 3 columns for more
    if n_plots <= 2:
        cols = n_plots
    elif n_plots <= 6:
        cols = 2
    else:
        cols = 3
    rows = int(math.ceil(n_plots / cols))
    
    print(f"Grid layout: {rows} rows x {cols} columns")
    
    # Create plotter with grid layout - narrow window for tighter vertical cells
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(800, 1200),
        title="Centerline Variations - All Airways"
    )
    
    for folder_idx, folder_name in enumerate(folders):
        folder_path = os.path.join(selected_airways_dir, folder_name)
        
        r = folder_idx // cols
        c = folder_idx % cols
        plotter.subplot(r, c)
        
        print(f"\n[{folder_idx+1}/{n_plots}] Processing folder: {folder_name}")
        
        # Load the 3D mesh model (folder_name.vtp) for visualization and constraint
        mesh_path = os.path.join(folder_path, f"{folder_name}.vtp")
        constraint_mesh = None
        if os.path.exists(mesh_path):
            try:
                # Load for constraint checking (VTK)
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(mesh_path)
                reader.Update()
                constraint_mesh = reader.GetOutput()
                
                # Load for visualization (PyVista)
                mesh = pv.read(mesh_path)
                plotter.add_mesh(
                    mesh,
                    color='#aabbcc',
                    opacity=0.25,
                    smooth_shading=True
                )
                print(f"  Loaded mesh: {folder_name}.vtp (constraint + visualization)")
            except Exception as e:
                print(f"  Error loading mesh: {e}")
        else:
            print(f"  Warning: mesh file {folder_name}.vtp not found")
        
        # Find all b*.vtp files (centerlines, not the full airway mesh)
        centerline_files = sorted(glob.glob(os.path.join(folder_path, "b*.vtp")))
        
        if not centerline_files:
            print(f"  No centerline files (b*.vtp) found in {folder_name}")
            plotter.add_text(f"{folder_name}\nNo centerlines", 
                           position=(0.05, 0.85), viewport=True,
                           font_size=12, color='red')
            continue
        
        print(f"  Found {len(centerline_files)} centerline files")
        
        color_idx = 0
        total_trajectories = 0
        
        for clf in centerline_files:
            clf_basename = os.path.basename(clf)
            
            try:
                # Load centerline
                polydata = load_centerline(clf)
                points = extract_points(polydata)
                
                if len(points) < 5:
                    continue
                
                # Generate variations constrained to mesh
                variations = generate_variations_in_memory(
                    points, 
                    num_variations=num_variations,
                    amplitude=amplitude,
                    smoothness=smoothness,
                    constraint_mesh=constraint_mesh
                )
                
                # Save variations as VTP files if requested
                if save_trajectories:
                    # Get centerline base name (e.g., b1 from b1.vtp)
                    cl_name = os.path.splitext(clf_basename)[0]
                    for var_idx, var_points in enumerate(variations):
                        if var_idx == 0:
                            continue  # Skip original
                        # Save as t{centerline_num}_v{variation_num}.vtp
                        var_filename = f"t{cl_name}_v{var_idx}.vtp"
                        var_path = os.path.join(folder_path, var_filename)
                        save_points_as_vtp(var_points, var_path)
                
                # Add only variation trajectories to the plot (skip original at index 0)
                for var_idx, var_points in enumerate(variations):
                    # Skip the original centerline (index 0)
                    if var_idx == 0:
                        continue
                    
                    line_mesh = points_to_pyvista_line(var_points)
                    
                    # Create tube for better visibility
                    tube = line_mesh.tube(radius=0.3)
                    
                    # Use cycling colors
                    color = TRAJECTORY_COLORS[color_idx % len(TRAJECTORY_COLORS)]
                    
                    plotter.add_mesh(
                        tube,
                        color=color,
                        smooth_shading=True,
                        opacity=1.0
                    )
                    
                    total_trajectories += 1
                
                color_idx += 1
                
            except Exception as e:
                print(f"  Error processing {clf_basename}: {e}")
        
        print(f"  Added {total_trajectories} trajectories (original + variations)")
        
        # Calculate total trajectories (centerlines × num_variations)
        total_generated = len(centerline_files) * num_variations
        
        # Add label - folder name, centerlines count, total trajectories count on separate lines
        plotter.add_text(
            f"{folder_name}\n{len(centerline_files)} CL\n{total_generated} Traj",
            position=(0.02, 0.82),
            viewport=True,
            font_size=8,
            color='black',
            shadow=False
        )
        
        # Set camera - use isometric view and reset for each subplot independently
        plotter.view_isometric()
        plotter.reset_camera()
    
    # Fill empty subplots
    for i in range(n_plots, rows * cols):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
    
    print("\n" + "-" * 60)
    print("Rendering visualization...")
    print("Controls: Left-drag to rotate, Right-drag to zoom, Middle-drag to pan")
    
    # Don't link views - each airway has different size/orientation
    # plotter.link_views()
    plotter.set_background('white')
    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate and visualize centerline variations for selected airways"
    )
    
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for selected_airways (default: auto-detect)"
    )
    
    parser.add_argument(
        "--num-folders", "-f",
        type=int,
        default=10,
        help="Number of folders to process (default: 10)"
    )
    
    parser.add_argument(
        "--num-variations", "-n",
        type=int,
        default=5,
        help="Number of variations per centerline (default: 5)"
    )
    
    parser.add_argument(
        "--amplitude", "-a",
        type=float,
        default=0.03,
        help="Noise amplitude (default: 0.03)"
    )
    
    parser.add_argument(
        "--smoothness", "-s",
        type=float,
        default=15.0,
        help="Noise smoothness (default: 15.0)"
    )
    
    parser.add_argument(
        "--save",
        action='store_true',
        default=False,
        help="Save generated trajectories as VTP files (t{branch}_v{variation}.vtp)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect base directory
    if args.base_dir is None:
        script_dir = Path(__file__).parent
        base_dir = script_dir / "selected_airways"
        if not base_dir.exists():
            print(f"Error: selected_airways directory not found at {base_dir}")
            print("Please specify --base-dir")
            sys.exit(1)
    else:
        base_dir = Path(args.base_dir)
    
    print(f"Using selected_airways directory: {base_dir}")
    
    visualize_all_folders(
        str(base_dir),
        num_folders=args.num_folders,
        num_variations=args.num_variations,
        amplitude=args.amplitude,
        smoothness=args.smoothness,
        save_trajectories=args.save
    )


if __name__ == "__main__":
    main()
