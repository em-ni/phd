import os
import glob
import sys
import subprocess
import argparse
import pyvista as pv

def compute_centerlines(input_dir):
    # Find all VTP files (recursive, exclude outputs)
    print(f"Scanning {input_dir}...")
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".vtp"):
                # Filters
                if "centerline" in f: continue
                if "ball.vtp" in f: continue
                if f.startswith("b") and f[1].isdigit(): continue # Skip b1.vtp, b2.vtp
                if "temp" in f: continue
                
                input_files.append(os.path.join(root, f))
    
    if not input_files:
        print(f"No VTP files found in {input_dir}")
        return

    print(f"Found {len(input_files)} models to process.")

    for i, input_vtp in enumerate(input_files):
        # Scan dir
        scan_dir = os.path.dirname(input_vtp)
        
        # Output filename: ball.vtp (consistent with robust script)
        output_vtp = os.path.join(scan_dir, "ball.vtp")
        
        if os.path.exists(output_vtp):
            print(f"[{i+1}/{len(input_files)}] Skipping {os.path.basename(input_vtp)} (ball.vtp exists)")
            continue

        print(f"[{i+1}/{len(input_files)}] Processing {os.path.basename(input_vtp)}...")
        print(f"  Output: {os.path.basename(output_vtp)}")
        
        flip_normals = 0
        simplify_voronoi = 1 # Enable by default for robustness
        use_tetgen = 0 # Disable TetGen by default (causes crashes)
        cap_displacement = 0.0
        smooth_iter = 0
        
        while True:
            # Handle smoothing
            current_input = input_vtp
            if smooth_iter > 0:
                print(f"  Smoothing surface (iterations={smooth_iter})...")
                mesh = pv.read(input_vtp)
                # Taubin smoothing preserves volume better than Laplacian
                smoothed = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)
                
                temp_smooth_path = input_vtp.replace('.vtp', '_temp_smooth.vtp')
                smoothed.save(temp_smooth_path)
                current_input = temp_smooth_path
            
            # Construct command inside loop to allow parameter updates
            input_vtp_clean = current_input.replace('\\', '/')
            output_vtp_clean = output_vtp.replace('\\', '/')
            
            cmd = [
                sys.executable, "-m", "vmtk.vmtkcenterlines",
                "-ifile", input_vtp_clean,
                "-ofile", output_vtp_clean,
                "-flipnormals", str(flip_normals),
                "-simplifyvoronoi", str(simplify_voronoi),
                "-usetetgen", str(use_tetgen),
                "-capdisplacement", str(cap_displacement)
            ]
            
            success = False
            try:
                # Run interactively (letting vmtk open its window)
                subprocess.run(cmd, check=True)
                print("  Done.")
                success = True
                
                # Visualize the result
                print("  Visualizing result. Close window to continue...")
                plotter = pv.Plotter(title=f"Centerline: {os.path.basename(output_vtp)} (Flip={flip_normals}, Smooth={smooth_iter})")
                
                # Load surface (show smoothed if applied)
                if os.path.exists(current_input):
                    surface = pv.read(current_input)
                    plotter.set_background('black') # Update bg to black to match robust (optional but good)
                    plotter.add_mesh(surface, color='white', opacity=0.75, label='Airway' + (' (Smoothed)' if smooth_iter > 0 else ''))
                    
                # Load centerline
                if os.path.exists(output_vtp):
                    try:
                        centerline = pv.read(output_vtp)
                        if centerline.n_points > 0:
                            plotter.add_mesh(centerline, color='red', line_width=4, render_lines_as_tubes=True, label='Centerline')
                        else:
                            print("  [Warning] Centerline file exists but is empty/invalid.")
                    except Exception as e:
                        print(f"  [Warning] Failed to read output centerline: {e}")
                    
                plotter.add_legend()
                plotter.show()
                
            except subprocess.CalledProcessError as e:
                print(f"  [Error] VMTK execution failed: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return

            # Cleanup temp file
            if smooth_iter > 0 and os.path.exists(current_input) and "_temp_smooth" in current_input:
                try:
                    os.remove(current_input)
                except:
                    pass

            print(f"  Settings: Flip={flip_normals}, TetGen={use_tetgen}, CapDisp={cap_displacement}, Smooth={smooth_iter}")
            
            if success:
                prompt = "  Satisfied? [Y/n] (n: retake, options below): "
            else:
                prompt = "  Extraction Failed. Adjust options and retry? [Y/n] (n: skip, options below): "
            
            print("  Options: [f] Flip Normals, [t] Toggle TetGen, [c] Toggle CapDisp, [s] Add Smoothing, [Enter] Confirm/Retry")
            response = input(prompt).strip().lower()
            
            if response == 'n':
                if success:
                    smooth_iter += 20
                    print(f"  Retaking centerline (Auto-smoothing: {smooth_iter})...")
                    continue
                else:
                    print("  Skipping this model...")
                    break # Break inner loop to go to next file
            elif response == 'f':
                flip_normals = 1 - flip_normals
                print(f"  Toggled FlipNormals to {flip_normals}. Retaking...")
                continue
            elif response == 't':
                use_tetgen = 1 - use_tetgen
                print(f"  Toggled TetGen to {use_tetgen}. Retaking...")
                continue
            elif response == 'c':
                cap_displacement = 0.1 if cap_displacement == 0.0 else 0.0
                print(f"  Toggled CapDisplacement to {cap_displacement}. Retaking...")
                continue
            elif response == 's':
                smooth_iter += 20
                print(f"  Increased smoothing iterations to {smooth_iter}. Retaking...")
                continue
            elif response == 'y' or response == '':
                if success:
                    # Satisfied
                    break
                else:
                    # Retry
                    smooth_iter += 20
                    print(f"  Retrying (Auto-smoothing: {smooth_iter})...")
                    continue
            else:
                # Unknown input, assume retry/continue
                continue

def main():
    parser = argparse.ArgumentParser(description="Compute centerlines for airway models using VMTK.")
    parser.add_argument("--dir", default="selected_airways", help="Directory containing VTP models")
    args = parser.parse_args()
    
    compute_centerlines(args.dir)

if __name__ == "__main__":
    main()
