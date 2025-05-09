import gmsh

def create_cylinder_with_cavity(outer_radius, height, cavity_height, cavity_radius, cavity_position, cavity_offset, mesh_size):
    """
    CAD model of a cylinder with a cavity. Used for 1 dof catheter.
    """
    # Initialize the gmsh model
    gmsh.initialize()
    gmsh.model.add("Cylinder with Cavity")

    # Create the outer cylinder
    ov = gmsh.model.occ.addCylinder(0, 0, 0,  0, 0, height,  outer_radius)
    
    # Create the cavity
    cv = gmsh.model.occ.addCylinder(cavity_offset, 0, cavity_position,  0, 0, cavity_height,  cavity_radius)
    
    cavity_volume = []
    cavity_volume.append((3, cv))

    # Subtract the cavity from the outer cylinder
    gmsh.model.occ.cut([(3, ov)], cavity_volume)    

    # Set the mesh size for all points in the model
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    
    # Synchronize the model
    gmsh.model.occ.synchronize()


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
    gmsh.model.mesh.generate(3)
    filename = "data/mesh/1dof/cavity.stl"
    gmsh.write(filename)
    print(f"Cavity saved as {filename}")

    # Remove the model
    gmsh.model.remove()



def save_gmsh_model(filename):
    extension = filename.split('.')[-1].lower()
    
    if extension in ["stl", "step", "vtk"]:
        gmsh.model.mesh.generate(3)
        gmsh.write(filename)
    else:
        raise ValueError("Unsupported file format.")

if __name__ == "__main__":

    ## 1 dof robot
    # Data are in mm
    outer_radius = 0.4
    height = 7
    cavity_height = 5
    cavity_radius = 0.15
    cavity_position = 0.1
    cavity_offset = 0.2

    # Set mesh_size to a desired value. Smaller values will produce a finer mesh.
    mesh_size = 0.1

    # Create body files
    create_cylinder_with_cavity(outer_radius, height, cavity_height, cavity_radius, cavity_position, cavity_offset, mesh_size)
    
    # Cavity data
    x = cavity_offset
    y = 0
    z = cavity_position
    radius = cavity_radius
    height = cavity_height

    # Create cavity file
    create_cavity(x, y, z, radius, height, mesh_size)

    save_gmsh_model("data/mesh/1dof/cylinder_with_cavity.stl")
    save_gmsh_model("data/mesh/1dof/cylinder_with_cavity.step")
    save_gmsh_model("data/mesh/1dof/cylinder_with_cavity.vtk")

    gmsh.finalize()



    