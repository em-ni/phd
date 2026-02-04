#!/usr/bin/env python

"""
Process Best Models Script
==========================

Gathers all `best_*.stl` files from the processed dataset.
Converts them to VTP format.
Saves them to a unified `selected_airways` folder.
Visualizes the final collection.

Usage:
    python process_best_models.py --base-dir processed_airways --output-dir selected_airways
"""

import os
import glob
import math
import argparse
import pyvista as pv
import shutil

def process_models(base_dir, output_dir, args):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    print(f"Scanning '{base_dir}' for 'best_*.stl' files...")
    
    # Recursive search or just 1-level deep? 
    # build_dataset structures it as base_dir/scan_id/files...
    # So we look in subdirectories.
    
    search_pattern = os.path.join(base_dir, "*", "best_*.stl")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No 'best_*.stl' files found.")
        return

    print(f"Found {len(files)} models. Converting and copying...")
    
    converted_files = []
    
    for i, stl_path in enumerate(files):
        filename = os.path.basename(stl_path)
        
        # Determine output name
        # Remove "best_" prefix? Or keep it? Use parent folder name as ID?
        # Usually user wants neat names. Let's use the folder name (Scan ID) if possible.
        # e.g. 1003/best_1003_s1.0...stl -> selected_airways/1003.vtp
        
        # Parse Scan ID to create subfolder
        info = parse_filename_info(filename)
        scan_id = info.get('id', 'Unknown')
        
        # Create subfolder: output_dir/1003
        scan_dir = os.path.join(output_dir, scan_id)
        if not os.path.exists(scan_dir):
            os.makedirs(scan_dir)
            
        # Target name: 1003.vtp
        vtp_name = f"{scan_id}.vtp"
        vtp_path = os.path.join(scan_dir, vtp_name)
        
        if os.path.exists(vtp_path):
             print(f"[{i+1}/{len(files)}] {vtp_name} exists. Skipping conversion.")
             converted_files.append(vtp_path)
             continue
        
        print(f"[{i+1}/{len(files)}] {filename} -> {vtp_name}")
        
        try:
            # Convert STL to VTP using PyVista
            mesh = pv.read(stl_path)
            mesh.save(vtp_path)
            converted_files.append(vtp_path)
        except Exception as e:
            print(f"  [Error] Failed to convert {stl_path}: {e}")

    print("-" * 50)
    print(f"Total available models: {len(converted_files)}.")
    
    if converted_files:
        visualize_results(converted_files, output_dir, limit=args.limit)

def parse_filename_info(filename):
    """
    Extracts parameters from filename.
    Expected format: best_1003_s1.0_t0.001_g100.stl (or similar)
    """
    name = filename.replace('.vtp', '').replace('.stl', '').replace('best_', '')
    parts = name.split('_')
    
    # Heuristic parsing
    info = {'id': parts[0] if parts else 'Unknown'}
    
    for p in parts:
        if p.startswith('s') and not p.startswith('stl'):
            info['sigma'] = p[1:]
        elif p.startswith('t'):
            info['thresh'] = p[1:]
        elif p.startswith('g'):
            info['gamma'] = p[1:]
            
    return info

def visualize_results(files, title_dir, limit=None):
    """
    Visualize the list of VTP files in a grid.
    """
    if limit is not None and limit > 0:
        print(f"Limiting visualization to {limit} files.")
        files = files[:limit]
        
    n_files = len(files)
    cols = int(math.ceil(math.sqrt(n_files)))
    rows = int(math.ceil(n_files / cols))
    
    print(f"\nVisualizing results in {rows}x{cols} grid...")
    
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(1600, 1000),
        title=f"Selected Airways ({title_dir})"
    )
    
    for i, file_path in enumerate(files):
        r = i // cols
        c = i % cols
        
        plotter.subplot(r, c)
        
        try:
            mesh = pv.read(file_path)
            filename = os.path.basename(file_path)
            
            # Parse info
            info = parse_filename_info(filename)
            label = f"{info.get('id', '?')}"
            
            # Color based on index or random? Let's use a nice default.
            plotter.add_mesh(mesh, color='#ADD8E6', smooth_shading=True) # LightBlue
            
            # Add text box
            plotter.add_text(
                label, 
                position='upper_left',
                font_size=8,
                color='black',
                font='arial',
                shadow=False
            )
            plotter.reset_camera()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    plotter.show()

def main():
    parser = argparse.ArgumentParser(description="Process and convert best models to VTP.")
    parser.add_argument("--base-dir", default="processed_airways", help="Base directory of processed scans")
    parser.add_argument("--output-dir", default="selected_airways", help="Output directory for VTP files")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of models to visualize (default: all)")
    
    args = parser.parse_args()
    
    process_models(args.base_dir, args.output_dir, args)

if __name__ == "__main__":
    main()
