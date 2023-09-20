import pyvista as pv

# 1. Load the VTK file
mesh = pv.read('data/mesh/output_tetrahedral.vtk')

# 2. Create a plotter object
plotter = pv.Plotter()

# 3. Add the mesh to the plotter
plotter.add_mesh(mesh)

# 4. Display the plot
plotter.show()
