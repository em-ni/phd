"""
Generate random centerline paths in a 3D mesh using vmtkcenterlines.

This script:
1. Loads a VTP mesh
2. Lets you pick a source point on the mesh (entry point)
3. Generates random target points on the mesh surface
4. Uses vmtkcenterlines to compute centerline paths from source to each target
5. Saves all paths as VTP files

Usage:
    python random_paths.py -ifile mesh.vtp -odir output_paths/ -n 10

Requirements:
    - VMTK installed (conda activate vmtk)
    - VTK
"""

import argparse
import numpy as np
import vtk
import subprocess
import os
import tempfile
import random
from pathlib import Path


def load_mesh(mesh_path: str) -> vtk.vtkPolyData:
    """Load mesh from various formats."""
    extension = mesh_path.lower().split('.')[-1]
    
    if extension == 'vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == 'stl':
        reader = vtk.vtkSTLReader()
    elif extension == 'obj':
        reader = vtk.vtkOBJReader()
    elif extension == 'vtk':
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported file format: {extension}")
    
    reader.SetFileName(mesh_path)
    reader.Update()
    return reader.GetOutput()


def save_mesh_as_vtp(mesh: vtk.vtkPolyData, path: str):
    """Save mesh to VTP format."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(mesh)
    writer.Write()


def get_random_surface_points(mesh: vtk.vtkPolyData, n_points: int, 
                               exclude_point: tuple = None, 
                               min_distance: float = 0.0) -> list:
    """Get random points from mesh surface."""
    num_mesh_points = mesh.GetNumberOfPoints()
    
    if n_points > num_mesh_points:
        print(f"Warning: Requested {n_points} points but mesh only has {num_mesh_points}")
        n_points = num_mesh_points
    
    selected_points = []
    attempts = 0
    max_attempts = n_points * 100
    
    while len(selected_points) < n_points and attempts < max_attempts:
        idx = random.randint(0, num_mesh_points - 1)
        point = mesh.GetPoint(idx)
        
        # Check minimum distance from excluded point
        if exclude_point and min_distance > 0:
            dist = np.linalg.norm(np.array(point) - np.array(exclude_point))
            if dist < min_distance:
                attempts += 1
                continue
        
        # Check minimum distance from already selected points
        too_close = False
        for existing in selected_points:
            dist = np.linalg.norm(np.array(point) - np.array(existing))
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected_points.append(point)
        
        attempts += 1
    
    return selected_points


def run_vmtkcenterlines(input_vtp: str, output_vtp: str, 
                         source_point: tuple, target_point: tuple,
                         vmtk_conda_env: str = "vmtk"):
    """
    Run vmtkcenterlines with specified source and target points.
    
    vmtkcenterlines can accept seed points via:
        -seedselector pointlist 
        -sourcepoints x1 y1 z1 
        -targetpoints x2 y2 z2
    """
    # Format points as space-separated coordinates
    source_str = f"{source_point[0]} {source_point[1]} {source_point[2]}"
    target_str = f"{target_point[0]} {target_point[1]} {target_point[2]}"
    
    # vmtk arguments
    vmtk_args = f'-ifile "{input_vtp}" -ofile "{output_vtp}" -seedselector pointlist -sourcepoints {source_str} -targetpoints {target_str}'
    
    # Try multiple methods to run vmtk on Windows
    methods = [
        # Method 1: Direct vmtkcenterlines (if in PATH)
        f"vmtkcenterlines {vmtk_args}",
        # Method 2: Using conda run with python -c to run vmtk
        f'conda run -n {vmtk_conda_env} python -c "from vmtk import vmtkcenterlines; c = vmtkcenterlines.vmtkCenterlines(); c.InputFileName = r\'{input_vtp}\'; c.OutputFileName = r\'{output_vtp}\'; c.SeedSelectorName = \'pointlist\'; c.SourcePoints = [{source_point[0]}, {source_point[1]}, {source_point[2]}]; c.TargetPoints = [{target_point[0]}, {target_point[1]}, {target_point[2]}]; c.Execute()"',
        # Method 3: Using vmtk pype syntax through conda
        f'conda run -n {vmtk_conda_env} python -c "from vmtk import pypes; pypes.PypeRun(\'vmtkcenterlines {vmtk_args}\')"',
    ]
    
    for i, method in enumerate(methods):
        print(f"Trying method {i+1}...")
        
        try:
            result = subprocess.run(
                method,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per path
            )
            
            if os.path.exists(output_vtp):
                print(f"  ✓ Success with method {i+1}")
                return True
            
            if result.stderr:
                # Only print first 300 chars of error
                err_msg = result.stderr.strip()[:300]
                print(f"  Error: {err_msg}")
                
        except subprocess.TimeoutExpired:
            print("  VMTK timed out")
        except Exception as e:
            print(f"  Exception: {e}")
    
    return False


class InteractiveSourcePicker:
    """Interactive tool to pick source point on mesh."""
    
    def __init__(self, mesh: vtk.vtkPolyData):
        self.mesh = mesh
        self.source_point = None
        self.source_actor = None
        
        self._setup_renderer()
    
    def _setup_renderer(self):
        """Setup VTK renderer."""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)
        
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1200, 900)
        self.render_window.SetWindowName("Pick SOURCE point - click on entry surface, then press 'q' to continue")
        self.render_window.AddRenderer(self.renderer)
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Add mesh
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.9)
        actor.GetProperty().SetOpacity(0.7)
        self.renderer.AddActor(actor)
        
        # Picker
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_click)
        self.renderer.ResetCamera()
    
    def _on_click(self, obj, event):
        """Handle click to pick source point."""
        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        if self.picker.GetCellId() >= 0:
            self.source_point = self.picker.GetPickPosition()
            
            # Remove old marker
            if self.source_actor:
                self.renderer.RemoveActor(self.source_actor)
            
            # Add new marker
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(self.source_point)
            sphere.SetRadius(self._get_radius())
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            self.source_actor = vtk.vtkActor()
            self.source_actor.SetMapper(mapper)
            self.source_actor.GetProperty().SetColor(0.2, 0.8, 0.2)
            
            self.renderer.AddActor(self.source_actor)
            self.render_window.Render()
            
            print(f"Source point: ({self.source_point[0]:.3f}, {self.source_point[1]:.3f}, {self.source_point[2]:.3f})")
        
        self.interactor.GetInteractorStyle().OnLeftButtonDown()
    
    def _get_radius(self):
        bounds = self.mesh.GetBounds()
        diagonal = np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
        return diagonal * 0.015
    
    def run(self) -> tuple:
        """Run interactive picker and return selected point."""
        print("\n" + "="*60)
        print("PICK SOURCE POINT (entry point)")
        print("="*60)
        print("Left-click to select the source/entry point on the mesh")
        print("Press 'q' when done to continue")
        print("="*60 + "\n")
        
        self.render_window.Render()
        self.interactor.Start()
        
        return self.source_point


def visualize_results(mesh: vtk.vtkPolyData, source_point: tuple, 
                      target_points: list, path_files: list):
    """Visualize the generated paths."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.15)
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 900)
    render_window.SetWindowName("Generated Paths")
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Add mesh (transparent)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.9)
    actor.GetProperty().SetOpacity(0.3)
    renderer.AddActor(actor)
    
    # Add source point (green)
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(source_point)
    bounds = mesh.GetBounds()
    radius = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2) * 0.015
    sphere.SetRadius(radius)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    source_actor = vtk.vtkActor()
    source_actor.SetMapper(mapper)
    source_actor.GetProperty().SetColor(0.2, 0.8, 0.2)
    renderer.AddActor(source_actor)
    
    # Add paths
    colors = [
        (0.2, 0.6, 1.0),   # Blue
        (1.0, 0.4, 0.2),   # Orange
        (0.8, 0.2, 0.8),   # Purple
        (0.2, 0.8, 0.8),   # Cyan
        (1.0, 0.8, 0.2),   # Yellow
    ]
    
    for i, path_file in enumerate(path_files):
        if os.path.exists(path_file):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(path_file)
            reader.Update()
            
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(reader.GetOutput())
            tube.SetRadius(radius * 0.3)
            tube.SetNumberOfSides(12)
            tube.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(tube.GetOutput())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*colors[i % len(colors)])
            renderer.AddActor(actor)
    
    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()


def main():
    parser = argparse.ArgumentParser(
        description="Generate random centerline paths using vmtkcenterlines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python random_paths.py -ifile mesh.vtp -odir output_paths/ -n 5
  
This will:
1. Let you pick a source (entry) point on the mesh
2. Generate 5 random target points on the mesh surface
3. Run vmtkcenterlines to compute centerlines from source to each target
4. Save paths as path_0.vtp, path_1.vtp, etc.

Note: Requires vmtk to be installed and accessible from command line.
      Run 'conda activate vmtk' first if using conda.
        """
    )
    
    parser.add_argument(
        "-ifile", "--input",
        required=True,
        help="Input mesh file (VTP format required for vmtk)"
    )
    parser.add_argument(
        "-odir", "--outdir",
        required=True,
        help="Output directory for path files"
    )
    parser.add_argument(
        "-n", "--num-paths",
        type=int,
        default=5,
        help="Number of random paths to generate (default: 5)"
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.0,
        help="Minimum distance between target points (default: 0)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization after generation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load mesh
    print(f"Loading mesh: {args.input}")
    mesh = load_mesh(args.input)
    print(f"Mesh has {mesh.GetNumberOfPoints()} points, {mesh.GetNumberOfCells()} cells")
    
    # Need VTP for vmtk - convert if necessary
    if not args.input.lower().endswith('.vtp'):
        print("Converting mesh to VTP format for vmtk...")
        vtp_path = os.path.join(args.outdir, "mesh_temp.vtp")
        save_mesh_as_vtp(mesh, vtp_path)
        vmtk_input = vtp_path
    else:
        vmtk_input = args.input
    
    # Interactive source point selection
    picker = InteractiveSourcePicker(mesh)
    source_point = picker.run()
    
    if source_point is None:
        print("No source point selected. Exiting.")
        return
    
    # Generate random target points
    print(f"\nGenerating {args.num_paths} random target points...")
    target_points = get_random_surface_points(
        mesh, 
        args.num_paths,
        exclude_point=source_point,
        min_distance=args.min_distance
    )
    
    print(f"Generated {len(target_points)} target points")
    
    # Run vmtkcenterlines for each target
    path_files = []
    successful = 0
    
    print("\n" + "="*60)
    print("COMPUTING CENTERLINES")
    print("="*60)
    
    for i, target in enumerate(target_points):
        print(f"\n[{i+1}/{len(target_points)}] Computing path to target {i}...")
        print(f"  Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
        
        output_file = os.path.join(args.outdir, f"path_{i}.vtp")
        
        success = run_vmtkcenterlines(
            vmtk_input,
            output_file,
            source_point,
            target
        )
        
        if success and os.path.exists(output_file):
            path_files.append(output_file)
            successful += 1
            print(f"  ✓ Saved: {output_file}")
        else:
            print(f"  ✗ Failed to generate path {i}")
    
    print("\n" + "="*60)
    print(f"RESULTS: {successful}/{len(target_points)} paths generated")
    print("="*60)
    
    # Save target points to file
    targets_file = os.path.join(args.outdir, "target_points.txt")
    with open(targets_file, 'w') as f:
        f.write(f"# Source point\n")
        f.write(f"{source_point[0]} {source_point[1]} {source_point[2]}\n\n")
        f.write(f"# Target points\n")
        for i, target in enumerate(target_points):
            f.write(f"{target[0]} {target[1]} {target[2]}\n")
    print(f"Saved points to: {targets_file}")
    
    # Visualize results
    if not args.no_visualize and successful > 0:
        print("\nVisualizing results...")
        visualize_results(mesh, source_point, target_points, path_files)


if __name__ == "__main__":
    main()
