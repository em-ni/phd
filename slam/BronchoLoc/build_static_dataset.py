import pyvista as pv
import numpy as np  
import os
    
class LungCenterlineBuilder:
    def __init__(self, vtk_path, cad_path):
        self.vtk_path = vtk_path
        self.cad_path = cad_path
        self.sdf_grid = None
        self.voxel_transform = None
        self.voxel_pitch = None
        self.centerline_points = None

    def process(self):
        print(f"[1/3] Loading VTK skeleton: {self.vtk_path}")
        skeleton = pv.read(self.vtk_path)
        self.centerline_points = skeleton.points
        print(f"      Loaded {len(self.centerline_points)} centerline points.")
        
        print(f"[2/3] Loading CAD (for visualization only): {self.cad_path}")
        
        return self.centerline_points

    def export_static_dataset(self, output_dir="dataset/static"):
        """
        Exports files needed for the ANT Network:
        1. centerline.npz (Centerline Points)
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Exporting static dataset to '{output_dir}/'...")

        # --- 1. Export Centerline Points ---
        npz_path = os.path.join(output_dir, "centerline.npz")
        # We save 'centerline_points' and also 'node_pos' as a fallback/alias for compatibility
        np.savez(npz_path, 
                 centerline_points=self.centerline_points,
                 node_pos=self.centerline_points) # Alias for compatibility
        print(f"  - Centerline points saved to {npz_path}")

    def visualize(self, save_screenshot="verification.png"):
        print("Preparing visualization...")
        p = pv.Plotter(off_screen=False) 

        # 1. Plot Original CAD 
        if os.path.exists(self.cad_path):
            mesh = pv.read(self.cad_path)
            p.add_mesh(mesh, color='wheat', opacity=0.25, label='CAD Wall')
        
        # 2. Plot Centerline Points
        if self.centerline_points is not None:
             p.add_mesh(pv.PolyData(self.centerline_points), color='red', point_size=5, 
                        render_points_as_spheres=True, label='Centerline Points')

        p.add_legend()
        p.add_axes()
        
        p.show()

if __name__ == "__main__":
    # Ensure these paths match your files
    vtk_file = "patient/centerline.vtk" 
    cad_file = "patient/lungs.obj" 
    
    builder = LungCenterlineBuilder(vtk_file, cad_file)
    builder.process()
    
    # Creates the 'static' folder with all 3 required files
    builder.export_static_dataset(output_dir="dataset/static")
    
    # Optional check
    builder.visualize()