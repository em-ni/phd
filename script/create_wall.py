
import gmsh

gmsh.initialize()

gmsh.model.add("wall")

# Create a wall
gmsh.model.occ.addBox(0, 0, 0, 2, 75, 75)

gmsh.model.occ.synchronize()

# We can activate the calculation of mesh element sizes based on curvature
# (here with a target of 20 elements per 2*Pi radians):

gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

gmsh.model.mesh.generate(3)

gmsh.write("data/mesh/wall.stl")