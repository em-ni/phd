import os
import sys
import subprocess
import argparse
import pyvista as pv
import vtk
import numpy as np

def pick_points(mesh, message):
    """
    Interactively pick points using PyVista.
    Returns a list of coordinates (x, y, z).
    """
    plotter = pv.Plotter()
    plotter.set_background('#080820') # VMTK Dark Blue
    plotter.add_mesh(mesh, color='white', opacity=0.75, pickable=True) 
    plotter.add_text(message, font_size=12, color='white')
    
    picked_points = []
    
    def callback(*args):
        # PyVista/VTK might pass extra arguments (like the picker), so we accept *args
        # The first argument is typically the picked point coordinate
        if not args:
            return
        point = args[0]
        picked_points.append(point)
        plotter.add_mesh(pv.Sphere(radius=1.0, center=point), color='red')
        print(f"  Picked: {point}")

    def trigger_pick():
        # Custom handler for Spacebar
        try:
            # Get cursor position
            if hasattr(plotter, 'iren') and plotter.iren is not None:
                pos = plotter.iren.get_event_position()
                
                # Use vtkCellPicker for accurate surface intersection
                picker = vtk.vtkCellPicker()
                picker.Pick(pos[0], pos[1], 0, plotter.renderer)
                
                # Get the intersection point on the surface
                picked_pos = picker.GetPickPosition()
                
                # If we picked something (z won't be 0 if valid intersection usually, or check picker return)
                if picker.GetCellId() != -1:
                    # Find the closest point index in the mesh to the picked location
                    pid = mesh.find_closest_point(picked_pos)
                    point = mesh.points[pid]
                    callback(point)
        except Exception as e:
            print(f"  Pick Error: {e}")

    plotter.enable_point_picking(callback=callback, show_message=False)
    plotter.add_key_event('space', trigger_pick)
    plotter.show()
    return picked_points

def run_vmtk_branch(input_vtp, output_vtp, source_point, target_point, 
                   flip_normals=0, simplify_voronoi=1, use_tetgen=0, 
                   cap_displacement=0.0, smooth_iter=0):
    """
    Run vmtkcenterlines for a single source-target pair.
    """
    
    # Handle smoothing (temporary file)
    current_input = input_vtp
    temp_files_to_clean = []
    
    if smooth_iter > 0:
        print(f"    Smoothing (iter={smooth_iter})...")
        try:
            mesh = pv.read(input_vtp)
            smoothed = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)
            temp_smooth = input_vtp.replace('.vtp', f'_temp_s{smooth_iter}.vtp')
            smoothed.save(temp_smooth)
            current_input = temp_smooth
            temp_files_to_clean.append(temp_smooth)
        except Exception as e:
            print(f"    [Error] Smoothing failed: {e}")
            return False

    # Format points for command line: x y z
    src_str = f"{source_point[0]} {source_point[1]} {source_point[2]}"
    tgt_str = f"{target_point[0]} {target_point[1]} {target_point[2]}"
    
    current_input_clean = current_input.replace('\\', '/')
    output_vtp_clean = output_vtp.replace('\\', '/')
    
    cmd = [
        sys.executable, "-m", "vmtk.vmtkcenterlines",
        "-ifile", current_input_clean,
        "-ofile", output_vtp_clean,
        "-seedselector", "pointlist",
        "-sourcepoints", *src_str.split(),
        "-targetpoints", *tgt_str.split(),
        "-flipnormals", str(int(flip_normals)),
        "-simplifyvoronoi", str(int(simplify_voronoi)),
        "-usetetgen", str(int(use_tetgen)),
        "-capdisplacement", str(float(cap_displacement))
    ]
    
    success = False
    try:
        # Capture output to avoid spamming console, only show on error? 
        # Or show to let user see progress. Let's show it.
        subprocess.run(cmd, check=True)
        # Check if file exists and is valid
        if os.path.exists(output_vtp) and os.path.getsize(output_vtp) > 100:
            success = True
        else:
            print("    [Error] Output file missing or empty.")
    except subprocess.CalledProcessError as e:
        print(f"    [Error] VMTK execution failed (Exit Code {e.returncode})")
    
    # Cleanup
    for f in temp_files_to_clean:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
                
    return success

def process_file(input_vtp, skip_existing=True):
    scan_dir = os.path.dirname(input_vtp)
    filename = os.path.basename(input_vtp)
    basename = os.path.splitext(filename)[0]
    
    # Final combined output name: ball.vtp
    final_output = os.path.join(scan_dir, "ball.vtp")
    
    print(f"\nProcessing: {filename}")
    print(f"  Directory: {scan_dir}")
    
    # Check for existing branches if requested
    if skip_existing:
        import glob
        existing_branches = glob.glob(os.path.join(scan_dir, "b*.vtp"))
        # Filter strictly for b{digits}.vtp to avoid false positives if any
        existing_branches = [f for f in existing_branches if os.path.basename(f).replace('b','').replace('.vtp','').isdigit()]
        
        if existing_branches:
             print(f"  Found {len(existing_branches)} existing branch files (b*.vtp). Skipping.")
             print(f"  (Use --no-skip to force re-processing)")
             return
    
    if os.path.exists(final_output):
        print(f"  {final_output} already exists. Skipping.")
        return

    mesh = pv.read(input_vtp)
    
    # --- Step 1: Pick Source ---
    print("  Step 1: Pick ONE Source Point (Inlet)")
    source_points = pick_points(mesh, "Pick ONE Inlet Point (Hover & Press Space, then close)")
    if not source_points:
        print("  No source point picked. Skipping.")
        return
    source_point = source_points[-1] # Take the last one if multiple clicked
    print(f"  Selected Source: {source_point}")
    
    # --- Step 2: Pick Targets ---
    print("  Step 2: Pick ANY NUMBER of Target Points (Outlets)")
    target_points = pick_points(mesh, "Pick Outlet Points (Hover & Press Space, then close)")
    if not target_points:
         print("  No target points picked. Skipping.")
         return
    print(f"  Selected {len(target_points)} Targets.")
    
    # --- Step 3: Process Branches ---
    successful_branches = []
    
    for i, target in enumerate(target_points):
        print(f"\n  [Branch {i+1}/{len(target_points)}]")
        
        # Save individual branch as b1.vtp, b2.vtp ...
        branch_output = os.path.join(scan_dir, f"b{i+1}.vtp")
        
        # Default Settings
        flip = 0
        tetgen = 0
        smooth = 0
        cap = 0.0
        
        while True:
            print(f"    Running... (Flip={flip}, TetGen={tetgen}, Smooth={smooth})")
            success = run_vmtk_branch(input_vtp, branch_output, source_point, target, 
                                      flip_normals=flip, use_tetgen=tetgen, 
                                      cap_displacement=cap, smooth_iter=smooth)
            
            if success:
                print("    Success!")
                
                # Verify visualization
                print("    Visualizing branch...")
                
                user_choice = {'resp': None}
                def confirm_keep():
                    print("    [UI] 'y' pressed. Keeping branch.")
                    user_choice['resp'] = 'y'
                
                def confirm_retake():
                    print("    [UI] 'n' pressed. Retaking.")
                    user_choice['resp'] = 'n'
                    
                def confirm_skip():
                    print("    [UI] 's' pressed. Skipping.")
                    user_choice['resp'] = 's'

                p = pv.Plotter(title=f"Branch {i+1}")
                p.set_background('#080820')
                p.add_mesh(mesh, color='white', opacity=0.75)
                p.add_mesh(pv.read(branch_output), color='red', line_width=4, render_lines_as_tubes=True)
                p.add_text("Press 'y' to Keep, 'n' to Retake, 's' to Skip", font_size=12, color='white')
                
                p.add_key_event('y', confirm_keep)
                p.add_key_event('n', confirm_retake)
                p.add_key_event('s', confirm_skip)
                
                # Non-blocking show
                p.show(interactive_update=True)
                
                # Manual event loop
                while user_choice['resp'] is None:
                    # Logic to check if window was closed manually
                    try:
                        # p.update() processes events. Returns True if window is open/active?
                        # Actually p.update() usually handles one frame.
                        p.update()
                        if p.render_window and p.render_window.GetGenericDisplayId() is None:
                             # Window might be closed? PyVista doesn't make this easy to check robustly across backends.
                             # But if user closes it, the loop continues?
                             pass
                    except AttributeError:
                        # Likely window closed
                        break
                    except Exception:
                        break
                    
                    # If user closed window manually (e.g. via X), we might need to break
                    # Often p.last_vt_window check or similar works?
                    # Let's rely on p.update() blocking effectively or running fast.
                    # Add a small sleep to prevent 100% CPU if update is non-blocking
                    # import time; time.sleep(0.01) # pyvista.update() usually waits for VSync or similar?
                    pass
                
                p.close()
                
                resp = user_choice['resp']
                if resp is None:
                    resp = input("    Keep this branch? [Y/n/s] (y: keep, n: retake, s: skip): ").strip().lower()
                if resp == 'n':
                    smooth += 20
                    print("    Retrying with more smoothing...")
                    continue
                elif resp == 's':
                    print("    Skipping branch.")
                    break
                else:
                    successful_branches.append(branch_output)
                    break
            else:
                print("    Failure.")
                print("    Options: [r] Retry (Auto-Smooth), [f] Flip Normals, [t] Toggle TetGen, [c] Toggle Cap, [s] Skip Branch")
                resp = input("    Choice: ").strip().lower()
                
                if resp == 's':
                    print("    Skipping branch.")
                    break
                elif resp == 'f':
                    flip = 1 - flip
                elif resp == 't':
                    tetgen = 1 - tetgen
                elif resp == 'c':
                    cap = 0.1 if cap == 0.0 else 0.0
                else:
                    smooth += 20
    
    # --- Step 4: Combine ---
    if successful_branches:
        print(f"\n  Combining {len(successful_branches)} branches...")
        appender = vtk.vtkAppendPolyData()
        for vtp in successful_branches:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(vtp)
            reader.Update()
            appender.AddInputData(reader.GetOutput())
        
        appender.Update()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(final_output)
        writer.SetInputData(appender.GetOutput())
        writer.Write()
        print(f"  Saved final centerline: {final_output}")
        
        # NOTE: We do NOT delete successful_branches (b1.vtp, etc) as per user request.
        
        # Final Visualization
        p = pv.Plotter(title="Final Result")
        p.set_background('#080820')
        p.add_mesh(mesh, color='white', opacity=0.75)
        p.add_mesh(pv.read(final_output), color='blue', line_width=4, render_lines_as_tubes=True)
        p.show()
        
    else:
        print("  No branches succeeded.")

def main():
    parser = argparse.ArgumentParser(description="Robust Centerline Extraction")
    parser.add_argument("--dir", default="selected_airways", help="Directory containing VTP models")
    parser.add_argument("--skip-existing", action='store_true', default=True, help="Skip folders with existing b*.vtp files (default: True)")
    parser.add_argument("--no-skip", dest='skip_existing', action='store_false', help="Process all folders even if branches exist")
    args = parser.parse_args()
    
    # Recursive search for .vtp files
    files_to_process = []
    
    print(f"Scanning {args.dir}...")
    for root, dirs, files in os.walk(args.dir):
        for f in files:
            if f.endswith(".vtp"):
                # Filters
                if "centerline" in f: continue
                if "ball.vtp" in f: continue
                if f.startswith("b") and f[1].isdigit(): continue # Skip b1.vtp, b2.vtp etc.
                if "temp" in f: continue
                
                full_path = os.path.join(root, f)
                files_to_process.append(full_path)
    
    print(f"Found {len(files_to_process)} VTP models to process.")
    
    for f in files_to_process:
        process_file(f, skip_existing=args.skip_existing)

if __name__ == "__main__":
    main()
