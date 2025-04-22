# Script to generate .obj and .msh files from an STL file. They are needed for the FEM simulation.
""" 
.obj files: The Wavefront OBJ (Object) file format is a widely used format for representing 3D geometry. 
It stores information about vertices, normals, texture coordinates, and face connectivity. In the context of Sofa, 
.obj files are typically used to define the shape and appearance of objects in a simulation. 
For example, if you want to simulate a sofa, you can create or obtain an .obj file that describes its geometry 
(vertex positions, polygons, etc.) along with any associated material properties such as textures and colors.
The .obj file can then be loaded into Sofa to create the corresponding object within the simulation.

.msh files: The .msh file format is not specific to Sofa but is commonly used in finite element analysis (FEA) and computational physics.
It is a binary or ASCII file format that stores information about the mesh, which is a discretized representation of an
object's geometry used for numerical calculations. A mesh consists of nodes (vertices) and elements (e.g., triangles or tetrahedra) 
that connect these nodes to form a meshed surface or volume. In the context of Sofa, .msh files can be used to import 
pre-defined meshes or generate custom meshes for objects in a simulation. The .msh file contains the necessary data to define
the mesh topology, node positions, and element connectivity.

In both cases, when using Sofa, you would typically load these files into the framework using appropriate components or plugins 
provided by Sofa. These components can interpret the data in the files and create the corresponding simulation objects with the 
desired geometry and mesh. 

.vtk files: The VTK (Visualization Toolkit) file format is another commonly used format for representing 3D geometry and associated data. 
It is widely used in scientific visualization and can store various types of data, including geometry, scalar and vector fields, and metadata. 
VTK files can be used to represent surfaces, volumes, and unstructured grids. They are often used for visualization purposes, 
as they can be easily read and processed by various visualization software and libraries.

.vtp files: The VTP (VTK PolyData) file format is a specific type of VTK file that is used to represent polygonal data. 
It stores information about vertices, polygons, and associated data such as scalar and vector fields. 
VTP files are commonly used to represent surfaces and can be used for visualization, analysis, and simulation purposes.

.vtu files: Part of the VTK (Visualization Toolkit) series, the .vtu file format specifically handles unstructured grid data. 
Unlike regular, structured grids, unstructured grids comprise points and cells in irregular patterns, allowing for complex geometries and 
adaptive meshing in simulations. These files store both the spatial points and the variously shaped cells (like tetrahedra or hexahedra) 
connecting them. .vtu files can include scalar or vector data at each grid point or cell, which is crucial for visualizing and analyzing 
results from fields like finite element analysis (FEA) and computational fluid dynamics (CFD). They are widely used for their flexibility 
and compatibility with diverse scientific visualization and simulation software.

.mdl files: The .mdl file format is not specific to any particular software or library, so its exact specifications may 
vary depending on the context. In general, .mdl files are used to store models or definitions of objects or systems. 
They can contain information about geometry, materials, textures, animations, and other properties. 
The specific structure and content of .mdl files depend on the software or framework they are used with.


"""


import argparse
import gmsh
import trimesh
import vtk
import pyvista as pv
import meshio
import os


def convert_vtk_to_vtp(vtk_path, vtp_path):
    # Using vtkDataSetReader to read legacy VTK files
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllScalarsOn()  # Ensure that all scalar data is read
    reader.Update()

    # Extracting the surface if necessary
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(reader.GetOutputPort())
    surface_filter.Update()

    # Writing the output as a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_path)
    writer.SetInputData(surface_filter.GetOutput())
    writer.Write()


def convert_stl_to_msh(stl_file_path, msh_file_path):
    # Start gmsh
    gmsh.initialize()
    gmsh.model.add("model")

    # Merge the STL file
    gmsh.merge(stl_file_path)

    # Set meshing algorithm (2: 2D, 6: Frontal-Delaunay)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh in MSH format
    gmsh.write(msh_file_path)

    # Finalize gmsh
    gmsh.finalize()


def convert_vtp_to_stl(vtp_path, stl_path):
    # Read VTP using vtk
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()

    # Write to STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stl_path)
    writer.SetInputData(reader.GetOutput())
    writer.Write()


def convert_vtp_to_obj(vtp_path, obj_path):
    temp_stl_path = "temp.stl"
    convert_vtp_to_stl(vtp_path, temp_stl_path)

    if os.path.exists(temp_stl_path) and os.path.getsize(temp_stl_path) > 0:
        convert_stl_to_obj(temp_stl_path, obj_path)
        os.remove(
            temp_stl_path
        )  # Uncomment this line to delete the temp file after conversion
    else:
        print("Error: STL file was not created properly.")


def convert_stl_to_obj(stl_path, obj_path):
    try:
        # Load the STL file
        mesh = trimesh.load(stl_path)
        # Save as OBJ
        mesh.export(obj_path)
    except Exception as e:
        print(f"Error occurred while converting STL to OBJ: {e}")


# Better using gmsh software (open terminal type "gmsh")
def convert_stl_to_vtk(stl_filename, vtk_filename):

    # Using FreeCad and gmsh
    """
    1) Using FreeCAD, import the STL file.
    * Switch to the “Part” Workbench and go to the “Part” menu in the top.
    * Select “Create shape from mesh”
    * Select the newly created shape and in the same “Part” menu select “Convert to solid”
    * Select this newest object in the tree and export it to a .step file.
    2) Using Gmsh, import the .step-file
    * In Gmsh, navigate to the Mesh menu on the left.
    * Select 3D. It should produce a volumetric mesh. This result you can now export as a VTK
    3) You might need to reduce the mesh size in step
        1) if it is too complex.
        In 2) you might need to tinker with the meshing parameters
        e.g. Tools->Options->Mesh->General->Element Size Factor to arrive at the correct results.
    """

    ## Using vtk
    # Read STL
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_filename)

    # Write VTK
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_filename)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()

    ## Using pyvista
    # Read STL
    # mesh = pv.read(stl_filename)

    # Write VTK
    # mesh.save(vtk_filename)

    ## Using meshio
    # Read STL
    # mesh = meshio.read(stl_filename)

    # Write VTK
    # meshio.write(vtk_filename, mesh)


def main():
    # Define the command line arguments using argparse
    parser = argparse.ArgumentParser(description="Convert 3D file formats.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file.")
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output file."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extract the file extensions to determine the conversion type
    input_extension = args.input.split(".")[-1].lower()
    output_extension = args.output.split(".")[-1].lower()

    # Depending on the extensions, call the appropriate conversion function
    if input_extension == "stl" and output_extension == "msh":
        convert_stl_to_msh(args.input, args.output)
    elif input_extension == "stl" and output_extension == "obj":
        convert_stl_to_obj(args.input, args.output)
    elif input_extension == "stl" and output_extension == "vtk":
        convert_stl_to_vtk(args.input, args.output)
    elif input_extension == "vtp" and output_extension == "obj":
        convert_vtp_to_obj(args.input, args.output)
    elif input_extension == "vtp" and output_extension == "stl":
        convert_vtp_to_stl(args.input, args.output)
    elif input_extension == "vtk" and output_extension == "vtp":
        convert_vtk_to_vtp(args.input, args.output)
    else:
        print(
            f"Conversion from {input_extension} to {output_extension} is not supported."
        )


# Uncomment the following line to run the program
if __name__ == "__main__":
    main()
