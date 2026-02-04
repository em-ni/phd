#!/usr/bin/env python

"""
Manual Model Selection Script
=============================

Allows the user to manually select the best airway model from a set of generated STL files.
Iterates through all folders in the base directory.

Usage:
    python select_best_model.py --base-dir processed_airways
"""

import os
import sys
import glob
import math
import shutil
import argparse
import pyvista as pv

def visualize_and_select(folder_path):
    """
    Visualize STL files in a folder and allow user to select one.
    
    Returns:
        bool: True if user wants to continue to next folder, False (quit).
    """
    if not os.path.exists(folder_path):
        return True

    # Find all STL files (excluding already selected 'best_' ones to avoid clutter)
    stl_files = sorted(glob.glob(os.path.join(folder_path, "*.stl")))
    stl_files = [f for f in stl_files if not os.path.basename(f).startswith("best_")]
    
    if not stl_files:
        print(f"No STL files found in '{folder_path}'.")
        return True

    n_files = len(stl_files)
    
    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n_files)))
    rows = int(math.ceil(n_files / cols))
    
    print(f"\nOpening '{os.path.basename(folder_path)}' ({n_files} models)...")
    
    # Colors
    DEFAULT_COLOR = '#7a8b99'  # Soft blue-grey
    SELECTED_COLOR = '#ff7f50'  # Coral/Orange
    
    # Create plotter with grid layout
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(1600, 1000),
        title=f"Select Best Model - {os.path.basename(folder_path)}"
    )
    
    all_actors = []
    
    # State dictionary to be modified by callbacks
    state = {
        'idx': 0,
        'action': None  # 'select', 'skip', 'quit'
    }
    
    # Load meshes
    for i, stl_file in enumerate(stl_files):
        r = i // cols
        c = i % cols
        
        plotter.subplot(r, c)
        
        try:
            mesh = pv.read(stl_file)
            filename = os.path.basename(stl_file)
            
            # Simplified label
            label = filename.replace('.stl', '')
            
            actor = plotter.add_mesh(
                mesh,
                color=DEFAULT_COLOR,
                smooth_shading=True
            )
            
            plotter.add_text(
                label[-20:], # Show last part if too long
                position=(0.05, 0.9),
                viewport=True,
                font_size=8,
                color='black'
            )
            
            all_actors.append({
                'actor': actor,
                'file': stl_file,
                'name': filename
            })
            
            if i == 0:
                plotter.reset_camera()
            else:
                # Share camera from first subplot? PyVista links views automatically usually?
                # We will explicit link later or just reset all
                plotter.reset_camera()
                
        except Exception as e:
            print(f"Error loading {stl_file}: {e}")

    # Fill empty subplots
    for i in range(n_files, rows * cols):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
    
    # --- Interaction Logic ---
    
    previous_idx = [0]

    def update_highlight():
        if not all_actors:
            return
            
        # Wrap index
        idx = state['idx'] % len(all_actors)
        state['idx'] = idx # Save wrapped
        
        # Reset previous
        prev = previous_idx[0]
        if prev != idx:
            all_actors[prev]['actor'].GetProperty().SetColor(pv.Color(DEFAULT_COLOR).float_rgb)
        
        # Highlight current
        item = all_actors[idx]
        item['actor'].GetProperty().SetColor(pv.Color(SELECTED_COLOR).float_rgb)
        
        previous_idx[0] = idx
        
        print(f"\rHighlighted: {item['name']}   ", end='', flush=True)
        plotter.render()

    def next_mesh():
        state['idx'] += 1
        update_highlight()
        
    def prev_mesh():
        state['idx'] -= 1
        update_highlight()
        
    def confirm_selection():
        print("\n[DEBUG] Selection confirmed. Processing...")
        try:
            if all_actors:
                idx = state['idx'] % len(all_actors)
                selected_file = all_actors[idx]['file']
                filename = all_actors[idx]['name']
                
                # Construct new name
                dirname = os.path.dirname(selected_file)
                new_name = f"best_{filename}"
                dest_path = os.path.join(dirname, new_name)
                
                print(f"[DEBUG] Saving {filename} -> {new_name}")
                shutil.copy2(selected_file, dest_path)
                print(f"Successfully saved: best_{filename}")
                
            state['action'] = 'select'
        except Exception as e:
            print(f"[ERROR] Failed to save file: {e}")
        
        print("[DEBUG] Force exiting subprocess (Success)...")
        os._exit(0) # Force exit with success code to continue to next folder
        
    def quit_script():
        print("\n[DEBUG] Quit requested.")
        state['action'] = 'quit'
        print("[DEBUG] Force exiting subprocess (Quit)...")
        os._exit(99) # Force exit with Quit code
        
    def skip_folder():
        print("\n[DEBUG] Skip requested.")
        state['action'] = 'skip'
        print("[DEBUG] Force exiting subprocess (Skip)...")
        os._exit(0) # Force exit with success code (just skip this one)

    # Register keys
    plotter.add_key_event('Right', next_mesh)
    plotter.add_key_event('n', next_mesh)
    plotter.add_key_event('Left', prev_mesh)
    plotter.add_key_event('p', prev_mesh)
    plotter.add_key_event('Return', confirm_selection)
    plotter.add_key_event('Enter', confirm_selection) # Try numeric pad enter or specific key
    plotter.add_key_event('r', confirm_selection) # Backup key
    plotter.add_key_event('q', quit_script)
    plotter.add_key_event('s', skip_folder)
    
    # Init highlight
    update_highlight()
    
    print("\nControls:")
    print("  [Left/Right] or [p/n]: Navigate")
    print("  [Enter] or [r]:        Select current model (Save as best_...)")
    print("  [s]:                   Skip this folder")
    print("  [q]:                   Quit script")
    
    plotter.link_views()
    plotter.show()
    
    # --- Post-Interaction Handling ---
    print(f"[DEBUG] Window closed. Action: {state['action']}")
    
    if state['action'] == 'quit':
        return False
        
    return True

import subprocess

def main():
    parser = argparse.ArgumentParser(description="Manual selection of best airway models.")
    parser.add_argument("--base-dir", default="processed_airways", help="Base directory containing scan folders")
    parser.add_argument("--folder", help="Specific folder to process (internal use)")
    args = parser.parse_args()
    
    # --- Subprocess Mode (Single Folder) ---
    if args.folder:
        folder_path = args.folder
        if not os.path.exists(folder_path):
            sys.exit(0) # Skip invalid
            
        print(f"Processing: {folder_path}")
        # Run visualization
        # We need to capture the return value from visualize_and_select
        # We will use exit codes:
        # 0: Continue (Selected or Skipped)
        # 99: Quit
        
        try:
            should_continue = visualize_and_select(folder_path)
            sys.exit(0 if should_continue else 99)
        except Exception as e:
            print(f"Error processing {folder_path}: {e}")
            sys.exit(0) # Error but try next?

    # --- Main Driver Mode ---
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory '{args.base_dir}' does not exist.")
        return
        
    # Get all subdirectories
    subdirs = sorted([
        os.path.join(args.base_dir, d) 
        for d in os.listdir(args.base_dir) 
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])
    
    print(f"Found {len(subdirs)} folders in '{args.base_dir}'.")
    
    for i, folder in enumerate(subdirs):
        folder_name = os.path.basename(folder)
        
        # Check if already has a best model
        existing_best = glob.glob(os.path.join(folder, "best_*.stl"))
        if existing_best:
            print(f"[{i+1}/{len(subdirs)}] {folder_name}: 'best_' model already exists. Skipping.")
            continue
            
        print(f"[{i+1}/{len(subdirs)}] Processing {folder_name}...")
        
        # Run subprocess
        cmd = [sys.executable, __file__, "--folder", folder]
        
        try:
            result = subprocess.run(cmd)
            if result.returncode == 99:
                print("Quit requested.")
                break
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error running subprocess: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
