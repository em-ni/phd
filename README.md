# navigation
Before proceeding you need to have:
- a solid of the environment in .stl format
- FreeCAD
- gmsh
- vmtk

How to prepare the files needed for the navigation simulation:
- get the CAD of the environment (e.g. lung model.stl)
- convert .stl to .vtk:
    - open FreeCAD and import the STL file.
    - switch to the “Part” Workbench and go to the “Part” menu in the top.
    - select “Create shape from mesh”
    - select the newly created shape and in the same “Part” menu select “Convert to solid”
    - select this newest object in the tree and export it to a .step file.
    - open Gmsh and import the .step file
    - in Gmsh, navigate to the Mesh menu on the left.
    - select 3D. It should produce a volumetric mesh. This result you can now export as a VTK
    - you might need to reduce the mesh size in previous steps if it is too complex. You might need to tinker with the meshing parameters e.g. Tools->Options->Mesh->General->Element Size Factor to arrive at the correct results.
- convert .vtk to .vtp using format_3d.py -i path/to/input.vtk -o path/to/output.vtp
- compute centerline (this is the path followed in the animation):
    - either use centerline.sh or follow this steps
    - conda activate vmtk
    - vmtkcenterlines -ifile path/to/output.vtp -ofile path/to/path.vtp
    - this will open the GUI, where you can set the points for the centerline
    - first select the end point(s) of the centerline pressing space
    - press q
    - then select the start point(s) of the centerline pressing space
    - press q
- get the negative solid for first person view navigation:
    - open the part menu
    - import the .obj file of the model
    - create shape from mesh for the imported model
    - make solid from the created shape
    - create a sphere that can contain the model
    - use boolean difference between the sphere and the solid
    - export the boolean cut as .obj