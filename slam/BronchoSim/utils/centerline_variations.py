"""
Generate smooth variations of existing centerline VTP files.

This script:
1. Loads existing centerline VTP files
2. Adds smooth noise while preserving the general path
3. Saves variations as new VTP files

Usage:
    python centerline_variations.py -i centerline.vtp -o variations/ -n 5
    python centerline_variations.py -i centerlines/ -o variations/ -n 3  # batch mode

The noise is filtered to be smooth (low-frequency) so the path flows naturally.
"""

import argparse
import numpy as np
import vtk
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def load_centerline(path: str) -> vtk.vtkPolyData:
    """Load centerline from VTP file."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def save_centerline(polydata: vtk.vtkPolyData, path: str):
    """Save centerline to VTP file."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()


def extract_points(polydata: vtk.vtkPolyData) -> np.ndarray:
    """Extract points from VTK polydata as numpy array."""
    n_points = polydata.GetNumberOfPoints()
    points = np.zeros((n_points, 3))
    for i in range(n_points):
        points[i] = polydata.GetPoint(i)
    return points


def create_polydata_from_points(points: np.ndarray, original: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Create VTK polydata from numpy points, preserving line connectivity from original."""
    new_polydata = vtk.vtkPolyData()
    
    # Create new points
    vtk_points = vtk.vtkPoints()
    for i in range(len(points)):
        vtk_points.InsertNextPoint(points[i])
    new_polydata.SetPoints(vtk_points)
    
    # Copy line connectivity from original
    new_polydata.SetLines(original.GetLines())
    
    # Copy point/cell data if present
    if original.GetPointData().GetNumberOfArrays() > 0:
        for i in range(original.GetPointData().GetNumberOfArrays()):
            new_polydata.GetPointData().AddArray(original.GetPointData().GetArray(i))
    
    return new_polydata


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
    avg_spacing = total_length / n_points
    
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
        # Create taper from 1.0 at start to taper_end_ratio at end
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


def check_points_within_mesh(original_points: np.ndarray, 
                              new_points: np.ndarray, 
                              mesh: vtk.vtkPolyData,
                              safety_margin: float = 0.8) -> tuple:
    """
    Check that new points don't move closer to mesh surface than allowed.
    
    This works by ensuring new points don't exceed a fraction of the original
    distance from the centerline to the mesh surface.
    
    Args:
        original_points: Original centerline points
        new_points: Modified centerline points
        mesh: VTK polydata mesh
        safety_margin: Fraction of original distance to allow (0.8 = stay within 80%)
    
    Returns:
        (valid: bool, num_violations: int)
    """
    # Build cell locator for fast distance queries
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    
    violations = 0
    
    for i in range(len(original_points)):
        orig_pt = original_points[i]
        new_pt = new_points[i]
        
        # Find closest point on mesh to original point
        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        
        cell_locator.FindClosestPoint(orig_pt, closest_point, cell_id, sub_id, dist2)
        orig_dist_to_mesh = np.sqrt(dist2.get())
        
        # Find closest point on mesh to new point
        cell_locator.FindClosestPoint(new_pt, closest_point, cell_id, sub_id, dist2)
        new_dist_to_mesh = np.sqrt(dist2.get())
        
        # Check if new point is too close to mesh surface
        min_allowed_dist = orig_dist_to_mesh * (1 - safety_margin)
        if new_dist_to_mesh < min_allowed_dist:
            violations += 1
    
    valid = (violations == 0)
    return valid, violations


def generate_variations(input_path: str, 
                        output_dir: str, 
                        num_variations: int = 5,
                        amplitude: float = 0.03,
                        smoothness: float = 10.0,
                        preserve_endpoints: bool = True,
                        constraint_mesh: vtk.vtkPolyData = None,
                        max_attempts: int = 50,
                        taper_end_ratio: float = 1.0):
    """
    Generate variations of a centerline file.
    
    Args:
        input_path: Path to input VTP file
        output_dir: Directory to save variations
        num_variations: Number of variations to generate
        amplitude: Noise amplitude (0.01 = subtle, 0.1 = large)
        smoothness: Noise smoothness (higher = smoother)
        preserve_endpoints: Keep start/end points fixed
        constraint_mesh: If provided, ensure all points stay inside this mesh
        max_attempts: Max attempts to find a valid variation
        taper_end_ratio: Amplitude ratio at end vs start (0.3 = 30% at end)
    """
    # Load centerline
    polydata = load_centerline(input_path)
    points = extract_points(polydata)
    
    basename = Path(input_path).stem
    
    print(f"Loaded: {input_path} ({len(points)} points)")
    
    if constraint_mesh:
        print(f"  Constraint: variations must stay inside mesh")
    if taper_end_ratio < 1.0:
        print(f"  Tapering: amplitude reduces to {taper_end_ratio*100:.0f}% at end")
    
    # Generate variations
    successful = 0
    for i in range(num_variations):
        # Try to generate a valid variation
        current_amplitude = amplitude
        
        for attempt in range(max_attempts):
            # Add some randomness to amplitude for variety
            var_amplitude = current_amplitude * (0.8 + 0.4 * np.random.random())
            var_smoothness = smoothness * (0.8 + 0.4 * np.random.random())
            
            new_points = add_smooth_noise(
                points, 
                amplitude=var_amplitude,
                smoothness=var_smoothness,
                preserve_endpoints=preserve_endpoints,
                taper_end_ratio=taper_end_ratio
            )
            
            # Check if within mesh bounds (if constraint provided)
            if constraint_mesh is not None:
                valid, num_violations = check_points_within_mesh(
                    points, new_points, constraint_mesh, safety_margin=0.8
                )
                
                if not valid:
                    if attempt < max_attempts - 1:
                        # Reduce amplitude and try again
                        current_amplitude *= 0.9
                        continue
                    else:
                        print(f"  ✗ Variation {i}: Could not find valid path after {max_attempts} attempts ({num_violations} violations)")
                        break
            
            # Valid variation found
            new_polydata = create_polydata_from_points(new_points, polydata)
            
            output_path = os.path.join(output_dir, f"{basename}_var{i}.vtp")
            save_centerline(new_polydata, output_path)
            
            if constraint_mesh and attempt > 0:
                print(f"  ✓ Saved: {output_path} (found after {attempt+1} attempts, amplitude={var_amplitude:.4f})")
            else:
                print(f"  ✓ Saved: {output_path}")
            
            successful += 1
            break
    
    return successful


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
        raise ValueError(f"Unsupported mesh format: {extension}")
    
    reader.SetFileName(mesh_path)
    reader.Update()
    return reader.GetOutput()


def visualize_variations(original_path: str, variation_paths: list, mesh_path: str = None):
    """Visualize original and variations together, optionally with mesh."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.15)
    
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 900)
    render_window.SetWindowName("Centerline Variations - Left-drag to rotate, Right-drag to zoom, Middle-drag to pan")
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Set trackball camera style for standard rotation controls
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Add mesh if provided
    if mesh_path and os.path.exists(mesh_path):
        mesh = load_mesh(mesh_path)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        
        mesh_actor = vtk.vtkActor()
        mesh_actor.SetMapper(mapper)
        mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.9)
        mesh_actor.GetProperty().SetOpacity(0.3)
        renderer.AddActor(mesh_actor)
        print(f"Loaded mesh: {mesh_path}")
    
    colors = [
        (1.0, 1.0, 1.0),   # White for original
        (0.2, 0.6, 1.0),   # Blue
        (1.0, 0.4, 0.2),   # Orange
        (0.2, 0.8, 0.4),   # Green
        (0.8, 0.2, 0.8),   # Purple
        (1.0, 0.8, 0.2),   # Yellow
    ]
    
    all_paths = [original_path] + variation_paths
    
    for i, path in enumerate(all_paths):
        if not os.path.exists(path):
            continue
            
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(reader.GetOutput())
        tube.SetRadius(0.05 if i == 0 else 0.04)  # Original slightly thicker
        tube.SetNumberOfSides(12)
        tube.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(tube.GetOutput())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colors[i % len(colors)])
        if i == 0:
            actor.GetProperty().SetOpacity(0.8)
        
        renderer.AddActor(actor)
    
    renderer.ResetCamera()
    render_window.Render()
    interactor.Start()


def main():
    parser = argparse.ArgumentParser(
        description="Generate smooth variations of centerline VTP files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file:
  python centerline_variations.py -i centerlines/b1.vtp -o variations/ -n 5
  
  # All VTP files in a directory:
  python centerline_variations.py -i centerlines/ -o variations/ -n 3
  
  # With custom amplitude (larger = more deviation):
  python centerline_variations.py -i b1.vtp -o out/ -n 5 --amplitude 0.05
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input VTP file or directory containing VTP files"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for variations"
    )
    parser.add_argument(
        "-n", "--num-variations",
        type=int,
        default=5,
        help="Number of variations per centerline (default: 5)"
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.03,
        help="Noise amplitude as fraction of path length (default: 0.03)"
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=15.0,
        help="Noise smoothness - higher = smoother (default: 15)"
    )
    parser.add_argument(
        "--no-preserve-endpoints",
        action="store_true",
        help="Allow modification of start/end points"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization after generation"
    )
    parser.add_argument(
        "--mesh", "-m",
        help="Path to mesh file (OBJ, STL, VTP) to show in visualization"
    )
    parser.add_argument(
        "--constraint-mesh", "-c",
        help="Path to mesh file to use as boundary constraint (variations must stay inside)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Max attempts to find valid variation per centerline (default: 50)"
    )
    parser.add_argument(
        "--taper",
        type=float,
        default=1.0,
        help="End amplitude as fraction of start (0.3 = 30%% at end). Useful for bronchi that narrow towards tips (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load constraint mesh if provided
    constraint_mesh = None
    if args.constraint_mesh:
        print(f"Loading constraint mesh: {args.constraint_mesh}")
        constraint_mesh = load_mesh(args.constraint_mesh)
        print(f"  Mesh has {constraint_mesh.GetNumberOfPoints()} points, {constraint_mesh.GetNumberOfCells()} cells")
    
    # Find input files
    if os.path.isdir(args.input):
        input_files = list(Path(args.input).glob("*.vtp"))
        print(f"Found {len(input_files)} VTP files in {args.input}")
    else:
        input_files = [Path(args.input)]
    
    if not input_files:
        print("No VTP files found!")
        return
    
    # Generate variations
    total_generated = 0
    all_variations = []
    
    for input_file in input_files:
        variations = generate_variations(
            str(input_file),
            args.output,
            args.num_variations,
            args.amplitude,
            args.smoothness,
            not args.no_preserve_endpoints,
            constraint_mesh,
            args.max_attempts,
            args.taper
        )
        total_generated += variations
        
        # Track for visualization
        basename = input_file.stem
        for i in range(args.num_variations):
            all_variations.append(os.path.join(args.output, f"{basename}_var{i}.vtp"))
    
    print(f"\nGenerated {total_generated} variations")
    
    # Visualize if requested
    if args.visualize and len(input_files) == 1:
        print("\nVisualizing...")
        visualize_variations(str(input_files[0]), all_variations[:5], args.mesh)


if __name__ == "__main__":
    main()
