#!/usr/bin/env python
"""
Dataset Building Pipeline Demo Script
======================================

This script performs the complete dataset building pipeline for a single CT scan,
designed for video recording. Each step runs sequentially with visualization.

Pipeline Steps:
1. CT to 3D STL Models (all 16 Frangi parameter combinations)
2. Best Model Selection (interactive grid selection)
3. Centerline Computation (interactive point picking + VMTK)
4. Trajectory Generation (variation generation + visualization)

Usage:
    python pipeline_demo.py --input <path_to_dicom_folder>
    python pipeline_demo.py --input archive/CT_Scans/EXP1_blind/1003
"""

import os
import sys
import glob
import math
import shutil
import argparse
import tempfile

import numpy as np
import pyvista as pv
import vtk
import trimesh
import SimpleITK as sitk

# ============================================================================
# Import functions from existing pipeline scripts
# ============================================================================

# From CT_to_3D.py
from CT_to_3D import (
    loadLargestSeries,
    extract_airways_tubular,
    sitk2vtk,
    extractSurface,
    cleanMesh,
    smoothMesh,
    writeSTL,
    smart_repair,
    TRIPLETS  # All 16 parameter combinations
)

# From robust_centerlines.py
from robust_centerlines import (
    pick_points,
    run_vmtk_branch
)

# From generate_trajectories.py
from generate_trajectories import (
    load_centerline,
    extract_points,
    generate_variations_in_memory,
    points_to_pyvista_line,
    save_points_as_vtp,
    TRAJECTORY_COLORS
)


# ============================================================================
# CT VISUALIZATION
# ============================================================================

def visualize_ct_scan(img):
    """
    Visualize CT scan as a grid of axial slices before processing.
    Uses SimpleITK image and displays with PyVista.
    """
    print("\nVisualizing CT scan...")
    
    # Get image as numpy array (z, y, x)
    arr = sitk.GetArrayFromImage(img)
    depth, height, width = arr.shape
    
    # Select slices to show (evenly spaced, 16 slices for 4x4 grid)
    num_slices = 16
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    # Create a 4x4 grid plotter
    rows, cols = 4, 4
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(1200, 1200),
        title="CT Scan - Axial Slices"
    )
    
    for i, slice_idx in enumerate(indices):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
        
        # Get slice and normalize to 0-255 for display
        slice_data = arr[slice_idx, :, :].astype(np.float32)
        
        # Window/level for lung (typical: W=1500, L=-600)
        window_center = -600
        window_width = 1500
        vmin = window_center - window_width / 2
        vmax = window_center + window_width / 2
        
        slice_data = np.clip(slice_data, vmin, vmax)
        slice_data = ((slice_data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        # Create a plane mesh with the slice as texture
        plane = pv.Plane(
            center=(width / 2, height / 2, 0),
            direction=(0, 0, 1),
            i_size=width,
            j_size=height,
            i_resolution=1,
            j_resolution=1
        )
        
        # Add as image (using scalar mapping)
        grid = pv.ImageData(dimensions=(width, height, 1))
        grid.point_data['values'] = slice_data.flatten(order='F')
        
        plotter.add_mesh(
            grid,
            cmap='gray',
            show_scalar_bar=False
        )
        
        # Show slice number
        plotter.add_text(
            f"Slice {slice_idx}/{depth-1}",
            position=(0.05, 0.9),
            viewport=True,
            font_size=8,
            color='white'
        )
        
        plotter.view_xy()
        plotter.camera.zoom(1.0)
    
    plotter.set_background('black')
    
    # Add a white banner at top using 2D actor
    from vtk import vtkTextActor
    banner = vtkTextActor()
    banner.SetInput("   CT Scan Preview   ")
    banner.GetTextProperty().SetFontSize(24)
    banner.GetTextProperty().SetColor(0, 0, 0)  # Black text
    banner.GetTextProperty().SetBackgroundColor(1, 1, 1)  # White background
    banner.GetTextProperty().SetBackgroundOpacity(1.0)
    banner.GetTextProperty().SetBold(True)
    banner.SetPosition(10, plotter.window_size[1] - 40)
    plotter.renderer.AddActor2D(banner)
    
    plotter.show(full_screen=True)


# ============================================================================
# STEP 1: CT to 3D STL Models
# ============================================================================

def step1_ct_to_3d(dicom_folder, output_dir, show_ct=True):
    """
    Step 1: Convert CT scan to 16 candidate 3D STL models.
    Returns list of generated STL file paths.
    """
    print("\n" + "=" * 70)
    print("STEP 1: CT to 3D STL Models")
    print("=" * 70)
    
    folder_name = os.path.basename(os.path.normpath(dicom_folder))
    scan_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(scan_output_dir, exist_ok=True)
    
    # Load DICOM
    print(f"\nLoading DICOM from: {dicom_folder}")
    img, modality = loadLargestSeries(dicom_folder)
    if img is None:
        print("[ERROR] Failed to load DICOM series.")
        return []
    print(f"Loaded CT scan with modality: {modality}")
    print(f"Image size: {img.GetSize()}")
    print(f"Spacing: {img.GetSpacing()}")
    
    # Show CT scan visualization
    if show_ct:
        visualize_ct_scan(img)
    
    generated_files = []
    total = len(TRIPLETS)
    
    for i, (sigma, thresh, gamma) in enumerate(TRIPLETS):
        count = i + 1
        filename = f"{folder_name}_s{sigma}_t{thresh}_g{int(gamma)}.stl"
        output_file = os.path.join(scan_output_dir, filename)
        
        print(f"\n--- [{count}/{total}] Sigma={sigma}, Thresh={thresh}, Gamma={gamma} ---")
        
        if os.path.exists(output_file):
            print(f"[Skip] Already exists: {filename}")
            generated_files.append(output_file)
            continue
        
        try:
            # Extract airways using Frangi vesselness
            print("  Extracting airways...")
            mask = extract_airways_tubular(img, sigma=sigma, vessel_thresh=thresh, gamma=gamma)
            
            # Check if mask is empty
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(mask)
            if stats.GetNumberOfLabels() == 0:
                print("  [Warning] Empty mask. Skipping.")
                continue
            
            # Convert to mesh
            print("  Converting to mesh...")
            vtkimg = sitk2vtk(mask)
            mesh_vtk = extractSurface(vtkimg, isovalue=0.5)
            if mesh_vtk is None or mesh_vtk.GetNumberOfPoints() == 0:
                print("  [Warning] Empty mesh. Skipping.")
                continue
            
            mesh_vtk = cleanMesh(mesh_vtk)
            mesh_vtk = smoothMesh(mesh_vtk, nIterations=10)
            
            # Save temp STL for Trimesh
            temp_stl = os.path.join(scan_output_dir, f"temp_{count}.stl")
            writeSTL(mesh_vtk, temp_stl)
            
            # Repair topology
            print("  Repairing topology...")
            mesh_tri = trimesh.load(temp_stl)
            if isinstance(mesh_tri, trimesh.Scene):
                if len(mesh_tri.geometry) == 0:
                    print("  [Warning] Empty scene. Skipping.")
                    os.remove(temp_stl)
                    continue
                mesh_tri = mesh_tri.dump(concatenate=True)
            
            if len(mesh_tri.vertices) == 0:
                print("  [Warning] No vertices. Skipping.")
                os.remove(temp_stl)
                continue
            
            mesh_fixed = smart_repair(mesh_tri, target_euler=2, smoothing_iters=100)
            
            # Save final mesh
            print(f"  Saving: {filename}")
            mesh_fixed.export(output_file)
            generated_files.append(output_file)
            
            if os.path.exists(temp_stl):
                os.remove(temp_stl)
                
        except Exception as e:
            print(f"  [Error] {e}")
            import traceback
            traceback.print_exc()
    # Step 1 complete - models will be shown in Step 2 selection
    print(f"\n[Step 1 Complete] Generated {len(generated_files)} models.")
    
    return generated_files, scan_output_dir


def visualize_step1_results(files):
    """Show all generated STL models in a grid layout."""
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return
    
    n_files = len(files)
    cols = int(math.ceil(math.sqrt(n_files)))
    rows = int(math.ceil(n_files / cols))
    
    plotter = pv.Plotter(
        shape=(rows, cols),
        window_size=(1600, 1000),
        title="Step 1: Generated 3D Models"
    )
    
    for i, stl_file in enumerate(files):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
        
        try:
            mesh = pv.read(stl_file)
            plotter.add_mesh(mesh, color='#7a8b99', smooth_shading=True)
            
            # Show parameter values from filename
            basename = os.path.basename(stl_file).replace('.stl', '')
            # Extract just the parameters part (last part after folder name)
            parts = basename.split('_')
            if len(parts) >= 3:
                label = '_'.join(parts[-3:])  # s1.0_t0.001_g100
            else:
                label = basename[-20:]
            
            plotter.add_text(label, position=(0.05, 0.9), viewport=True, font_size=8, color='black')
            plotter.reset_camera()
        except Exception as e:
            print(f"Error loading {stl_file}: {e}")
    
    # Fill empty subplots
    for i in range(n_files, rows * cols):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
    
    plotter.link_views()
    plotter.add_text(
        "Step 1 Complete: Review generated models. Close window to continue.",
        position='upper_edge', font_size=12, color='blue'
    )
    plotter.show()


# ============================================================================
# STEP 2: Best Model Selection
# ============================================================================

def step2_select_best(stl_files, scan_output_dir):
    """
    Step 2: Interactive selection of the best model from the 16 candidates.
    Runs the selection in a subprocess to avoid PyVista window blocking issues.
    Returns path to the selected best model.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Best Model Selection")
    print("=" * 70)
    
    if not stl_files:
        print("[ERROR] No models to select from.")
        return None, None
    
    print(f"\nOpening selection view for {len(stl_files)} models...")
    print("\nControls:")
    print("  [Left/Right] or [p/n]: Navigate between models")
    print("  [Enter] or [r]:        Select highlighted model")
    print("  [s]:                   Skip")
    print("  [q]:                   Quit")
    
    # Run the selection using the original script as subprocess
    import subprocess
    cmd = [sys.executable, "select_best_model.py", "--folder", scan_output_dir]
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode == 99:
            print("Selection cancelled.")
            return None, None
    except Exception as e:
        print(f"Error running selection: {e}")
        return None, None
    
    # Check if best model was created
    best_files = glob.glob(os.path.join(scan_output_dir, "best_*.stl"))
    if not best_files:
        print("[ERROR] No best model was selected.")
        return None, None
    
    best_stl = best_files[0]  # Take the first (should only be one)
    print(f"\nSelected: {os.path.basename(best_stl)}")
    
    # Convert to VTP for centerline processing
    vtp_path = os.path.join(scan_output_dir, f"{os.path.basename(scan_output_dir)}.vtp")
    mesh = pv.read(best_stl)
    mesh.save(vtp_path)
    print(f"Saved VTP version: {os.path.basename(vtp_path)}")
    print("\n[Step 2 Complete] Best model selected.")
    
    return best_stl, vtp_path


# ============================================================================
# STEP 3: Centerline Visualization (Load Existing)
# ============================================================================

def step3_compute_centerlines(vtp_path, centerlines_dir):
    """
    Step 3: Show point picking for demo, then load existing centerlines.
    vtp_path: path to the mesh VTP for visualization
    centerlines_dir: folder containing ball.vtp and b*.vtp centerline files
    Returns list of branch centerline files.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Centerline Extraction")
    print("=" * 70)
    
    if not vtp_path or not os.path.exists(vtp_path):
        print("[ERROR] No VTP file for centerline visualization.")
        return None
    
    mesh = pv.read(vtp_path)
    
    # Pick source point (inlet) - for demo visualization
    print("\nStep 3a: Pick ONE Source Point (Inlet)")
    print("  - Hover over the trachea entrance and press SPACE")
    print("  - Close window when done")
    source_points = pick_points(mesh, "Pick ONE Inlet Point (Hover + Space, then close)")
    
    if not source_points:
        print("[Warning] No source point picked. Continuing with existing centerlines...")
    else:
        source_point = source_points[-1]
        print(f"Selected Source: {source_point}")
    
    # Pick target points (outlets) - for demo visualization
    print("\nStep 3b: Pick Target Points (Branch Outlets)")
    print("  - Pick multiple points at branch endings")
    print("  - Close window when done")
    target_points = pick_points(mesh, "Pick Outlet Points (Hover + Space, then close)")
    
    if not target_points:
        print("[Warning] No target points picked. Continuing with existing centerlines...")
    else:
        print(f"Selected {len(target_points)} target points.")
    
    # Load existing centerlines from the specified folder
    print(f"\nStep 3c: Loading existing centerlines from {centerlines_dir}...")
    
    # Check for ball.vtp (combined centerlines)
    ball_path = os.path.join(centerlines_dir, "ball.vtp")
    if not os.path.exists(ball_path):
        print(f"[ERROR] No ball.vtp found in {centerlines_dir}")
        return None
    
    # Also get individual branches for step 4
    existing_branches = sorted(glob.glob(os.path.join(centerlines_dir, "b*.vtp")))
    existing_branches = [f for f in existing_branches if os.path.basename(f) != "ball.vtp"]
    
    print(f"Loaded combined centerlines: ball.vtp")
    print(f"Found {len(existing_branches)} individual branches for trajectory generation")
    
    # Show centerlines overlaid on mesh
    print("\nShowing centerlines...")
    p = pv.Plotter(title="Step 3: Centerlines")
    p.set_background('#080820')
    p.add_mesh(mesh, color='white', opacity=0.75)
    
    # Load and display the combined centerlines
    cl = pv.read(ball_path)
    p.add_mesh(cl, color='red', line_width=4, render_lines_as_tubes=True)
    
    # Add banner text
    from vtk import vtkTextActor
    banner = vtkTextActor()
    banner.SetInput("   Centerlines Loaded   ")
    banner.GetTextProperty().SetFontSize(24)
    banner.GetTextProperty().SetColor(0, 0, 0)
    banner.GetTextProperty().SetBackgroundColor(1, 1, 1)
    banner.GetTextProperty().SetBackgroundOpacity(1.0)
    banner.GetTextProperty().SetBold(True)
    banner.SetPosition(10, p.window_size[1] - 40)
    p.renderer.AddActor2D(banner)
    
    p.show(full_screen=True)
    
    print(f"\n[Step 3 Complete] Loaded centerlines")
    
    return existing_branches


# ============================================================================
# STEP 4: Trajectory Generation
# ============================================================================

def step4_generate_trajectories(branch_files, vtp_path, num_variations=5):
    """
    Step 4: Generate trajectory variations from centerlines.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Trajectory Generation")
    print("=" * 70)
    
    if not branch_files:
        print("[ERROR] No centerline branches to process.")
        return
    
    scan_dir = os.path.dirname(branch_files[0])
    mesh = pv.read(vtp_path)
    
    # Load constraint mesh for VTK
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    constraint_mesh = reader.GetOutput()
    
    print(f"\nProcessing {len(branch_files)} branches with {num_variations} variations each...")
    
    all_variations = []
    
    for clf in branch_files:
        clf_basename = os.path.basename(clf)
        print(f"\n  Processing: {clf_basename}")
        
        try:
            polydata = load_centerline(clf)
            points = extract_points(polydata)
            
            print(f"    Points in centerline: {len(points)}")
            
            if len(points) < 5:
                print(f"    Too few points ({len(points)}). Skipping.")
                continue
            
            # Generate variations
            variations = generate_variations_in_memory(
                points,
                num_variations=num_variations,
                amplitude=0.03,
                smoothness=15.0,
                constraint_mesh=constraint_mesh
            )
            
            # Save variations
            cl_name = os.path.splitext(clf_basename)[0]
            for var_idx, var_points in enumerate(variations):
                if var_idx == 0:
                    continue  # Skip original
                var_filename = f"t{cl_name}_v{var_idx}.vtp"
                var_path = os.path.join(scan_dir, var_filename)
                save_points_as_vtp(var_points, var_path)
            
            all_variations.append((clf, variations))
            print(f"    Generated {len(variations)-1} variations")
            
        except Exception as e:
            print(f"    [Error] {e}")
            import traceback
            traceback.print_exc()
    
    if not all_variations:
        print("\n[ERROR] No variations were generated.")
        return
    
    # Visualize all trajectories
    total_traj = sum(len(v) - 1 for _, v in all_variations)
    print(f"\n[Step 4 Complete] Generated {total_traj} trajectories. Showing...")
    
    p = pv.Plotter(title="Step 4: Generated Trajectories")
    p.set_background('#101030')
    p.add_mesh(mesh, color='#aabbcc', opacity=0.3, smooth_shading=True)
    
    color_idx = 0
    for clf, variations in all_variations:
        for var_idx, var_points in enumerate(variations):
            if var_idx == 0:
                continue  # Skip original
            
            line_mesh = points_to_pyvista_line(var_points)
            tube = line_mesh.tube(radius=0.4)
            color = TRAJECTORY_COLORS[color_idx % len(TRAJECTORY_COLORS)]
            p.add_mesh(tube, color=color, smooth_shading=True)
        color_idx += 1
    
    # Add banner text
    from vtk import vtkTextActor
    banner = vtkTextActor()
    banner.SetInput(f"   {total_traj} Trajectories Generated   ")
    banner.GetTextProperty().SetFontSize(24)
    banner.GetTextProperty().SetColor(0, 0, 0)
    banner.GetTextProperty().SetBackgroundColor(1, 1, 1)
    banner.GetTextProperty().SetBackgroundOpacity(1.0)
    banner.GetTextProperty().SetBold(True)
    banner.SetPosition(10, p.window_size[1] - 40)
    p.renderer.AddActor2D(banner)
    
    p.show(full_screen=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dataset Building Pipeline Demo - For Video Recording"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to DICOM folder (e.g., archive/CT_Scans/EXP1_blind/1003)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="demo_output",
        help="Output directory (default: demo_output)"
    )
    
    parser.add_argument(
        "--num-variations", "-n",
        type=int,
        default=5,
        help="Number of trajectory variations per centerline (default: 5)"
    )
    
    parser.add_argument(
        "--processed-dir", "-p",
        default=None,
        help="Path to existing processed folder with STL models (skips Step 1 generation)"
    )
    
    parser.add_argument(
        "--centerlines-dir", "-c",
        default=None,
        help="Path to folder containing existing centerlines (ball.vtp, b*.vtp). E.g., selected_airways/1003"
    )
    
    parser.add_argument(
        "--skip-ct-preview",
        action="store_true",
        help="Skip CT scan preview visualization"
    )
    
    args = parser.parse_args()
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input folder does not exist: {args.input}")
        return
    
    print("=" * 70)
    print("DATASET BUILDING PIPELINE DEMO")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.processed_dir:
        print(f"Using existing models: {args.processed_dir}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: CT to 3D (or use existing)
    if args.processed_dir and os.path.exists(args.processed_dir):
        # Use existing processed folder
        print("\n" + "=" * 70)
        print("STEP 1: Using Existing STL Models")
        print("=" * 70)
        
        scan_output_dir = args.processed_dir
        stl_files = sorted(glob.glob(os.path.join(scan_output_dir, "*.stl")))
        stl_files = [f for f in stl_files if not os.path.basename(f).startswith("best_")]
        
        print(f"Found {len(stl_files)} existing STL models in {scan_output_dir}")
        
        # Optionally show CT scan
        if not args.skip_ct_preview:
            print("\nLoading CT for preview...")
            img, _ = loadLargestSeries(args.input)
            if img:
                visualize_ct_scan(img)
        
        # Models will be shown in Step 2 selection
    else:
        # Generate models from CT
        stl_files, scan_output_dir = step1_ct_to_3d(args.input, args.output, show_ct=not args.skip_ct_preview)
    
    if not stl_files:
        print("\n[ABORT] Step 1 failed - no models generated.")
        return
    
    # Step 2: Select best model
    best_stl, best_vtp = step2_select_best(stl_files, scan_output_dir)
    
    if not best_vtp:
        print("\n[ABORT] Step 2 failed - no model selected.")
        return
    
    # Step 3: Load/Show centerlines
    centerlines_dir = args.centerlines_dir if args.centerlines_dir else scan_output_dir
    branch_files = step3_compute_centerlines(best_vtp, centerlines_dir)
    
    if not branch_files:
        print("\n[ABORT] Step 3 failed - no centerlines found.")
        return
    
    # Step 4: Generate trajectories
    step4_generate_trajectories(branch_files, best_vtp, num_variations=args.num_variations)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {scan_output_dir}")
    print(f"  - {len(stl_files)} STL models generated")
    print(f"  - Best model: {os.path.basename(best_stl) if best_stl else 'None'}")
    print(f"  - {len(branch_files)} centerline branches computed")
    print(f"  - {len(branch_files) * args.num_variations} trajectory variations generated")
    print("=" * 70)


if __name__ == "__main__":
    main()
