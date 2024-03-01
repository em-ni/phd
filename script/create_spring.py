# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 19
#
#  Thrusections, fillets, pipes, mesh size from curvature
#
# ------------------------------------------------------------------------------

# The OpenCASCADE geometry kernel supports several useful features for solid
# modelling.

import gmsh
import math

gmsh.initialize()

# Set the number of threads Gmsh should use for computations
num_threads = 10  # Replace with the number of threads you want to use
gmsh.option.setNumber("General.NumThreads", num_threads)
gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay-based tetrahedral mesh


gmsh.model.add("spring")

# OpenCASCADE also allows general extrusions along a smooth path. Let's first
# define a spline curve:
nturns = 20                  # spring_height / spring_pitch
npts = 100                    # number of points to define the spline
r = 1                        # spring_internal_radius
h = 12                        # total height of the spring
disk_radius = 0.1            # radius of the disk to be extruded

p = []
for i in range(0, npts):
    theta = i * 2 * math.pi * nturns / npts
    gmsh.model.occ.addPoint(r * math.cos(theta), r * math.sin(theta),
                            i * h / npts, 1, 1000 + i)
    p.append(1000 + i)
gmsh.model.occ.addSpline(p, 1000)

# A wire is like a curve loop, but open:
gmsh.model.occ.addWire([1000], 1000)

# We define the shape we would like to extrude along the spline (a disk):
gmsh.model.occ.addDisk(1, 0, 0, disk_radius, disk_radius, 1000)
gmsh.model.occ.rotate([(2, 1000)], 0, 0, 0, 1, 0, 0, math.pi / 2)

# We extrude the disk along the spline to create a pipe (other sweeping types
# can be specified; try e.g. 'Frenet' instead of 'DiscreteTrihedron'):
gmsh.model.occ.addPipe([(2, 1000)], 1000, 'DiscreteTrihedron')

# We delete the source surface, and increase the number of sub-edges for a
# nicer display of the geometry:
gmsh.model.occ.remove([(2, 1000)])
gmsh.option.setNumber("Geometry.NumSubEdges", 1000)

gmsh.model.occ.synchronize()

# We can activate the calculation of mesh element sizes based on curvature
# (here with a target of 20 elements per 2*Pi radians):
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)

# We can constraint the min and max element sizes to stay within reasonable
# values (see `t10.py' for more details):
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1) # 0.001
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)

gmsh.model.mesh.generate(3)
gmsh.write("data/mesh/1dof/spring.step")
gmsh.write("data/mesh/1dof/spring.stl")
gmsh.write("data/mesh/1dof/spring.vtk")

# Launch the GUI to see the results:
#if '-nopopup' not in sys.argv:
#    gmsh.fltk.run()

gmsh.finalize()
