#!/usr/bin/env python
"""
Trajectory Visualization Script
================================

Visualizes all computed trajectories (t*.vtp files) across selected airways
in an optimized grid layout for displaying many trajectories efficiently.

Optimizations for large-scale visualization:
- Efficient PyVista multiblock rendering
- Reduced geometry complexity (thin tubes, low polygon count)
- Progressive loading with progress indicators
- Optional mesh culling for performance
- Batched actor creation

Usage:
    python visualize_trajectories.py
    python visualize_trajectories.py --num-folders 5
    python visualize_trajectories.py --num-folders -1  # All folders
    python visualize_trajectories.py --hide-mesh       # Hide 3D mesh for speed
    python visualize_trajectories.py --tube-radius 0.15
"""

import os
import sys
import glob
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pyvista as pv
import vtk

# Disable VTK warnings for cleaner output
vtk.vtkObject.GlobalWarningDisplayOff()


# ----------------------------------------------------------------------------
# Color Palettes - vivid colors that work on light/dark backgrounds
# ----------------------------------------------------------------------------

# Nice color palette for trajectories - vivid colors for light background
# Same as generate_trajectories.py and pipeline_demo.py for consistency
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

MESH_COLOR = '#c0c8d0'
MESH_OPACITY = 0.20


# ----------------------------------------------------------------------------
# I/O Functions
# ----------------------------------------------------------------------------

def load_vtp_points(path: str) -> np.ndarray:
    """
    Load points from a VTP file, following line cell connectivity order.
    Optimized for speed with minimal VTK overhead.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    
    lines = polydata.GetLines()
    if lines.GetNumberOfCells() > 0:
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
        n_points = polydata.GetNumberOfPoints()
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            points[i] = polydata.GetPoint(i)
        return points


def points_to_pyvista_line(points: np.ndarray) -> pv.PolyData:
    """Convert numpy points array to PyVista line mesh."""
    n_points = len(points)
    lines = np.zeros((n_points - 1, 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(n_points - 1)
    lines[:, 2] = np.arange(1, n_points)
    
    return pv.PolyData(points, lines=lines.flatten())


# ----------------------------------------------------------------------------
# Folder Discovery
# ----------------------------------------------------------------------------

def find_trajectory_files(folder_path: str, traj_per_branch: int = -1) -> List[str]:
    """
    Find trajectory VTP files (t*.vtp pattern) in a folder.
    
    Files follow naming: tb{branch}_v{variation}.vtp (e.g. tb1_v1.vtp, tball_v3.vtp).
    When traj_per_branch > 0, only the first N variations per branch are returned.
    """
    all_files = sorted(glob.glob(os.path.join(folder_path, "t*.vtp")))
    
    if traj_per_branch < 0 or traj_per_branch == 0:
        return all_files
    
    # Group by branch: extract branch name before "_v" suffix
    import re
    branch_counts: Dict[str, int] = {}
    filtered = []
    for f in all_files:
        basename = os.path.splitext(os.path.basename(f))[0]  # e.g. "tb1_v3"
        match = re.match(r'^(t.+?)_v\d+$', basename)
        branch = match.group(1) if match else basename
        count = branch_counts.get(branch, 0)
        if count < traj_per_branch:
            filtered.append(f)
            branch_counts[branch] = count + 1
    return filtered


def find_centerline_files(folder_path: str) -> List[str]:
    """Find all centerline VTP files (b*.vtp pattern) in a folder."""
    return sorted(glob.glob(os.path.join(folder_path, "b*.vtp")))


def get_folder_stats(folder_path: str, traj_per_branch: int = -1) -> Dict:
    """Get trajectory and centerline counts for a folder."""
    folder_name = os.path.basename(folder_path)
    all_trajectories = find_trajectory_files(folder_path)
    shown_trajectories = find_trajectory_files(folder_path, traj_per_branch)
    centerlines = find_centerline_files(folder_path)
    mesh_path = os.path.join(folder_path, f"{folder_name}.vtp")
    
    return {
        'name': folder_name,
        'path': folder_path,
        'n_trajectories': len(all_trajectories),
        'n_shown_trajectories': len(shown_trajectories),
        'n_centerlines': len(centerlines),
        'trajectory_files': shown_trajectories,
        'centerline_files': centerlines,
        'mesh_exists': os.path.exists(mesh_path),
        'mesh_path': mesh_path if os.path.exists(mesh_path) else None
    }


# ----------------------------------------------------------------------------
# Optimized Visualization
# ----------------------------------------------------------------------------

def create_trajectory_tube(points: np.ndarray, radius: float = 0.15, 
                           n_sides: int = 6, taper: bool = True) -> pv.PolyData:
    """
    Create a tube mesh from trajectory points.
    
    When taper=True, the tube starts very thin (where trajectories overlap
    near the trachea) and grows progressively thicker toward the distal end,
    making individual branches easier to distinguish.
    """
    if len(points) < 2:
        return None
    
    line_mesh = points_to_pyvista_line(points)
    
    if taper:
        # Quadratic taper: stays thin near the start, flares toward the end
        t = np.linspace(0.0, 1.0, len(points))
        line_mesh['radius'] = (1 + 10 * t**2) * radius
        
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(line_mesh)
        tube_filter.SetNumberOfSides(n_sides)
        tube_filter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tube_filter.Update()
        return pv.wrap(tube_filter.GetOutput())
    else:
        return line_mesh.tube(radius=radius, n_sides=n_sides)


def visualize_trajectories(selected_airways_dir: str,
                           num_folders: int = 10,
                           show_mesh: bool = True,
                           show_centerlines: bool = False,
                           tube_radius: float = 0.15,
                           light_mode: bool = False,
                           grid_shape: Optional[Tuple[int, int]] = None,
                           traj_per_branch: int = -1):
    """
    Visualize all computed trajectories across selected airways in a grid.
    
    Optimized for displaying many trajectories efficiently:
    - Low-polygon tubes (6 sides instead of 20)
    - Optional mesh hiding for speed
    - Batched loading with progress
    - Efficient multi-subplot rendering
    
    Args:
        selected_airways_dir: Path to selected_airways directory
        num_folders: Number of folders to process (-1 for all)
        show_mesh: Whether to show the 3D airway mesh
        show_centerlines: Whether to show original centerlines
        tube_radius: Radius of trajectory tubes (smaller = faster)
        max_cols: Maximum number of columns in grid
    """
    # Get list of folders
    folders = sorted([f for f in os.listdir(selected_airways_dir) 
                     if os.path.isdir(os.path.join(selected_airways_dir, f))])
    
    if num_folders >= 0:
        folders = folders[:num_folders]
    
    # Get stats for each folder
    print(f"Scanning {len(folders)} folders...")
    folder_stats = []
    total_trajectories = 0
    
    for folder_name in folders:
        folder_path = os.path.join(selected_airways_dir, folder_name)
        stats = get_folder_stats(folder_path, traj_per_branch=traj_per_branch)
        folder_stats.append(stats)
        total_trajectories += stats['n_trajectories']
    
    # Filter to folders with trajectories
    folder_stats = [s for s in folder_stats if s['n_shown_trajectories'] > 0]
    
    if not folder_stats:
        print("No trajectory files found! Run generate_trajectories.py --save first.")
        return
    
    n_plots = len(folder_stats)
    print(f"Found {total_trajectories} total trajectories across {n_plots} folders")
    print("-" * 60)
    
    # Calculate grid dimensions
    if grid_shape is not None:
        rows, cols = grid_shape
        if rows * cols < n_plots:
            print(f"Warning: grid {rows}x{cols} = {rows*cols} cells but have {n_plots} plots, some will be skipped")
            folder_stats = folder_stats[:rows * cols]
            n_plots = len(folder_stats)
    else:
        # Auto-compute: optimized for common screen sizes (16:9 aspect ratio)
        GRID_LAYOUTS = {
            1: (1, 1), 2: (2, 1), 3: (3, 2), 4: (4, 2), 5: (5, 3), 6: (6, 3),
            7: (4, 2), 8: (4, 2), 9: (5, 2), 10: (5, 2), 11: (6, 2), 12: (6, 2),
            13: (5, 3), 14: (5, 3), 15: (5, 3), 16: (4, 4), 17: (6, 3), 18: (6, 3),
            19: (5, 4), 20: (5, 4), 21: (7, 3), 22: (6, 4), 23: (6, 4), 24: (6, 4),
            25: (5, 5), 26: (7, 4), 27: (7, 4), 28: (7, 4), 29: (6, 5), 30: (6, 5),
        }
        
        if n_plots in GRID_LAYOUTS:
            cols, rows = GRID_LAYOUTS[n_plots]
        else:
            cols = int(math.ceil(math.sqrt(n_plots * 1.5)))
            rows = int(math.ceil(n_plots / cols))
        
        # Ensure we have enough cells
        while rows * cols < n_plots:
            rows += 1
    
    print(f"Grid layout: {rows} rows x {cols} columns ({rows * cols} cells for {n_plots} plots)")
    
    # Window size: square cells that fit on common screens (1920x1080, 2560x1440)
    cell_size = 300  # pixels per cell (square)
    window_w = min(1800, cols * cell_size)
    window_h = min(1000, rows * cell_size)
    
    # Create plotter with optimized settings
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(window_w, window_h),
        title="Trajectory Visualization - All Airways"
    )
    
    # Process each folder
    for idx, stats in enumerate(folder_stats):
        r = idx // cols
        c = idx % cols
        plotter.subplot(r, c)
        
        folder_name = stats['name']
        folder_path = stats['path']
        
        shown_str = (f" (showing {stats['n_shown_trajectories']})"
                     if stats['n_shown_trajectories'] < stats['n_trajectories'] else "")
        print(f"[{idx+1}/{n_plots}] {folder_name}: "
              f"{stats['n_trajectories']} trajectories{shown_str}, "
              f"{stats['n_centerlines']} centerlines")
        
        # Load and display mesh (if requested)
        if show_mesh and stats['mesh_path']:
            try:
                mesh = pv.read(stats['mesh_path'])
                plotter.add_mesh(
                    mesh,
                    color=MESH_COLOR,
                    opacity=MESH_OPACITY,
                    smooth_shading=True
                )
            except Exception as e:
                print(f"  Warning: Could not load mesh: {e}")
        
        # Load and display centerlines (if requested)
        if show_centerlines:
            for cl_file in stats['centerline_files']:
                try:
                    points = load_vtp_points(cl_file)
                    if len(points) >= 2:
                        tube = create_trajectory_tube(points, radius=tube_radius * 0.8, n_sides=6)
                        if tube:
                            plotter.add_mesh(tube, color='#333333', opacity=0.5)
                except Exception:
                    pass
        
        # Load and display trajectories
        color_idx = 0
        for traj_file in stats['trajectory_files']:
            try:
                points = load_vtp_points(traj_file)
                if len(points) >= 2:
                    tube = create_trajectory_tube(points, radius=tube_radius, n_sides=6)
                    if tube:
                        color = TRAJECTORY_COLORS[color_idx % len(TRAJECTORY_COLORS)]
                        plotter.add_mesh(tube, color=color, smooth_shading=True)
                        color_idx += 1
            except Exception as e:
                print(f"  Warning: Could not load {os.path.basename(traj_file)}: {e}")
        
        # Add label (positioned lower to avoid cell border cutoff)
        label_color = 'black' if light_mode else 'white'
        plotter.add_text(
            f"{folder_name}\n{stats['n_centerlines']} CL\n{stats['n_trajectories']} Traj",
            position=(0.02, 0.68),
            viewport=True,
            font_size=7,
            color=label_color,
            shadow=False
        )
        
        # Set camera
        plotter.view_isometric()
        plotter.reset_camera()
    
    # Fill empty subplots
    for i in range(n_plots, rows * cols):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
    
    print("-" * 60)
    print("Rendering visualization...")
    print("Controls: Left-drag=rotate, Right-drag=zoom, Middle-drag=pan")
    
    if light_mode:
        plotter.set_background('white')
    else:
        plotter.set_background('#101030')  # Dark blue, same as pipeline_demo.py
    plotter.show()


def print_summary(selected_airways_dir: str, num_folders: int = -1):
    """Print a summary of available trajectories without visualizing."""
    folders = sorted([f for f in os.listdir(selected_airways_dir) 
                     if os.path.isdir(os.path.join(selected_airways_dir, f))])
    
    if num_folders >= 0:
        folders = folders[:num_folders]
    
    print(f"{'Folder':<10} {'Centerlines':<12} {'Trajectories':<12} {'Mesh':<6}")
    print("-" * 42)
    
    total_cl = 0
    total_traj = 0
    
    for folder_name in folders:
        folder_path = os.path.join(selected_airways_dir, folder_name)
        stats = get_folder_stats(folder_path)
        
        mesh_status = "Yes" if stats['mesh_exists'] else "No"
        print(f"{folder_name:<10} {stats['n_centerlines']:<12} "
              f"{stats['n_trajectories']:<12} {mesh_status:<6}")
        
        total_cl += stats['n_centerlines']
        total_traj += stats['n_trajectories']
    
    print("-" * 42)
    print(f"{'TOTAL':<10} {total_cl:<12} {total_traj:<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize computed trajectories across selected airways"
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
        help="Number of folders to process (default: 10, use -1 for all)"
    )
    
    parser.add_argument(
        "--hide-mesh",
        action='store_true',
        help="Hide 3D airway mesh for faster rendering"
    )
    
    parser.add_argument(
        "--show-centerlines",
        action='store_true',
        help="Show original centerlines (darker, thinner)"
    )
    
    parser.add_argument(
        "--tube-radius", "-r",
        type=float,
        default=0.15,
        help="Trajectory tube radius in mm (default: 0.15, smaller=faster)"
    )
    
    parser.add_argument(
        "--light-mode",
        action='store_true',
        help="Use light background with black text (for paper figures)"
    )
    
    parser.add_argument(
        "--grid", "-g",
        type=str,
        default=None,
        help="Grid shape as RxC, e.g. 3x3 (default: auto-compute)"
    )
    
    parser.add_argument(
        "--traj-per-branch", "-t",
        type=int,
        default=-1,
        help="Number of trajectory variations to show per branch (default: -1 = all, use 1 for one per branch)"
    )

    
    parser.add_argument(
        "--summary",
        action='store_true',
        help="Just print summary, don't visualize"
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
    
    if args.summary:
        print_summary(str(base_dir), args.num_folders)
    else:
        # Parse grid shape if provided
        grid_shape = None
        if args.grid:
            parts = args.grid.lower().split('x')
            if len(parts) == 2:
                grid_shape = (int(parts[0]), int(parts[1]))
            else:
                print(f"Error: --grid must be in RxC format, e.g. 3x3")
                sys.exit(1)
        
        visualize_trajectories(
            str(base_dir),
            num_folders=args.num_folders,
            show_mesh=not args.hide_mesh,
            show_centerlines=args.show_centerlines,
            tube_radius=args.tube_radius,
            light_mode=args.light_mode,
            grid_shape=grid_shape,
            traj_per_branch=args.traj_per_branch
        )


if __name__ == "__main__":
    main()
