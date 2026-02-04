#!/usr/bin/env python

"""
Airway Visualization Script
============================

Visualizes all STL variations for a processed CT scan in a PyVista grid.

Usage:
    python visualize_airways.py <folder_name>
    python visualize_airways.py 1003
    python visualize_airways.py --base-dir /path/to/processed_airways 1003

The script looks for all .stl files in the specified folder and displays them
in an auto-sized grid layout.
"""

import os
import sys
import glob
import math
import argparse

import pyvista as pv


def visualize_folder(folder_path):
    """
    Visualize all STL files in a folder using PyVista grid layout.
    
    Args:
        folder_path: Path to the folder containing STL files
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Find all STL files in the folder
    stl_files = sorted(glob.glob(os.path.join(folder_path, "*.stl")))
    
    if not stl_files:
        print(f"No STL files found in '{folder_path}'.")
        return

    n_files = len(stl_files)
    
    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n_files)))
    rows = int(math.ceil(n_files / cols))
    
    print(f"Visualizing {n_files} STL files from '{folder_path}'")
    print(f"Grid layout: {rows} rows x {cols} columns")
    print("-" * 50)
    
    # Colors
    DEFAULT_COLOR = '#7a8b99'  # Soft blue-grey
    SELECTED_COLOR = '#ff7f50'  # Coral/Orange
    
    # Create plotter with grid layout
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(1600, 1000),
        title=f"Airways - {os.path.basename(folder_path)}"
    )
    
    # Store all actors and their info
    all_actors = []
    selected_actor = [None]
    
    for i, stl_file in enumerate(stl_files):
        r = i // cols
        c = i % cols
        
        plotter.subplot(r, c)
        
        try:
            mesh = pv.read(stl_file)
            
            # Get filename for label
            filename = os.path.basename(stl_file)
            
            # Extract parameters from filename for cleaner label
            # Expected format: folder_s1.0_t0.01_g100.stl
            parts = filename.replace('.stl', '').split('_')
            
            # Parse s, t, g values
            sigma = thresh = gamma = None
            for p in parts:
                if p.startswith('s') and not p.startswith('stl'):
                    try:
                        sigma = float(p[1:])
                    except ValueError:
                        pass
                elif p.startswith('t'):
                    try:
                        thresh = float(p[1:])
                    except ValueError:
                        pass
                elif p.startswith('g'):
                    try:
                        gamma = float(p[1:])
                    except ValueError:
                        pass
            
            # Build formatted label with parameter names
            if sigma is not None and thresh is not None and gamma is not None:
                label = f"s={sigma}  t={thresh}  g={int(gamma)}"
            else:
                label = filename
            
            # Add mesh with blue-grey color
            actor = plotter.add_mesh(
                mesh,
                name=f"mesh_{i}",
                color=DEFAULT_COLOR,
                smooth_shading=True,
                show_edges=False,
                opacity=1.0,
                pickable=True
            )
            
            all_actors.append({
                'actor': actor,
                'filename': filename,
                'label': label,
                'index': i
            })
            
            # Add label at top of subplot using viewport coordinates
            plotter.add_text(
                label,
                position=(0.05, 0.85),
                viewport=True,
                font_size=10,
                color='black',
                font='arial'
            )
            
            # Set consistent view
            plotter.camera_position = 'xy'
            plotter.reset_camera()
            
            print(f"  [{i+1}/{n_files}] {filename}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load {filename}: {e}")
            plotter.add_text(
                f"Error:\n{os.path.basename(stl_file)}",
                position='upper_edge',
                font_size=8,
                color='red'
            )
    
    # Fill empty subplots if grid is not fully used
    for i in range(n_files, rows * cols):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
    # Current selection index
    current_idx = [0]
    
    def update_selection(new_idx):
        """Update which mesh is highlighted."""
        if not all_actors:
            return
            
        # Wrap around
        new_idx = new_idx % len(all_actors)
        
        # Reset previous selection
        old_actor = all_actors[current_idx[0]]['actor']
        old_actor.GetProperty().SetColor(pv.Color(DEFAULT_COLOR).float_rgb)
        
        # Highlight new selection
        new_actor = all_actors[new_idx]['actor']
        new_actor.GetProperty().SetColor(pv.Color(SELECTED_COLOR).float_rgb)
        
        current_idx[0] = new_idx
        print(f"Selected [{new_idx+1}/{len(all_actors)}]: {all_actors[new_idx]['filename']}")
        plotter.render()
    
    def next_mesh():
        """Select next mesh."""
        update_selection(current_idx[0] + 1)
    
    def prev_mesh():
        """Select previous mesh."""
        update_selection(current_idx[0] - 1)
    
    # Highlight first mesh initially
    if all_actors:
        all_actors[0]['actor'].GetProperty().SetColor(pv.Color(SELECTED_COLOR).float_rgb)
    
    print("-" * 50)
    print("Rendering visualization...")
    print("Controls: Press 'n' for Next, 'p' for Previous mesh")
    
    # Add keyboard callbacks
    plotter.add_key_event('n', next_mesh)
    plotter.add_key_event('p', prev_mesh)
    
    # Show the visualization
    plotter.link_views()
    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize processed airway STL files in a grid"
    )
    
    parser.add_argument(
        "folder",
        help="Name of the folder containing STL files (e.g., '1003')"
    )
    
    parser.add_argument(
        "--base-dir",
        default="processed_airways",
        help="Base directory where processed folders are stored (default: processed_airways)"
    )
    
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Treat 'folder' as an absolute path instead of relative to base-dir"
    )
    
    args = parser.parse_args()
    
    if args.absolute:
        folder_path = args.folder
    else:
        folder_path = os.path.join(args.base_dir, args.folder)
    
    visualize_folder(folder_path)


if __name__ == "__main__":
    main()
