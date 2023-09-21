import gmsh

def create_cavity(x, y, z, radius, height, mesh_size):
    # Initialize the gmsh model
    gmsh.initialize()
    gmsh.model.add("Cylinder with Cavity")

    # Create the cavity
    ov = gmsh.model.occ.addCylinder(x, y, z,  0, 0, height,  radius)

    # Set the mesh size for all points in the model
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    
    # Synchronize the model
    gmsh.model.occ.synchronize()

def save_gmsh_model(filename):
    extension = filename.split('.')[-1].lower()
    
    if extension in ["stl", "step", "vtk"]:
        gmsh.model.mesh.generate(3)
        gmsh.write(filename)
    else:
        raise ValueError("Unsupported file format.")

if __name__ == "__main__":
    x = 0.2
    y = 0
    z = 2
    radius = 0.15
    height = 6

    # Set mesh_size to a desired value. Smaller values will produce a finer mesh.
    mesh_size = 0.1
    create_cavity(x, y, z, radius, height, mesh_size)

    save_gmsh_model("data/mesh/cavity.stl")

    gmsh.finalize()
    