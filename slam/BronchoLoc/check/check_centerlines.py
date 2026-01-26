# Visualize all centerlines inside the 3D lung model

import os
import glob
import numpy as np
import pyvista as pv

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patient_dir = os.path.join(base_dir, "patient")
    centerlines_dir = os.path.join(patient_dir, "centerlines")
    mesh_path = os.path.join(patient_dir, "lungs.obj")
    main_centerline = os.path.join(patient_dir, "centerline.vtk")
    
    # Load mesh
    print(f"Loading mesh: {mesh_path}")
    mesh = pv.read(mesh_path)
    
    # Find all centerline files (.vtk and .vtp) in centerlines folder
    vtk_files = glob.glob(os.path.join(centerlines_dir, "*.vtk"))
    vtp_files = glob.glob(os.path.join(centerlines_dir, "*.vtp"))
    all_files = vtk_files + vtp_files
    
    print(f"Found {len(all_files)} centerline files")
    
    # Create plotter
    plotter = pv.Plotter(title="All Centerlines Visualization")
    
    # Add mesh
    plotter.add_mesh(mesh, color='lightblue', opacity=0.15, label='Lungs')
    
    # Generate distinct colors for each centerline
    cmap = plt.cm.get_cmap('tab20', len(all_files))
    
    # Add each centerline
    for i, vtk_path in enumerate(all_files):
        name = os.path.basename(vtk_path)
        label = os.path.splitext(name)[0]  # Remove extension for cleaner label
        try:
            centerline = pv.read(vtk_path)
            color = cmap(i)[:3]
            plotter.add_mesh(
                centerline, 
                color=color, 
                point_size=4,
                render_points_as_spheres=True,
                label=name
            )
            
            # Add label at the end point of the centerline
            if centerline.n_points > 0:
                end_point = centerline.points[-1]
                plotter.add_point_labels(
                    [end_point], 
                    [label],
                    font_size=12,
                    text_color=color,
                    shape_opacity=0.6,
                    always_visible=True
                )
            
            print(f"  Loaded: {name} ({centerline.n_points} points)")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    
    # Add legend and show
    plotter.add_axes()
    plotter.add_text(
        f"Centerlines: {len(all_files)} files\n\n"
        "Rotate: Left Mouse\n"
        "Zoom: Scroll\n"
        "Pan: Middle Mouse",
        position='upper_right', font_size=10
    )
    
    plotter.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
