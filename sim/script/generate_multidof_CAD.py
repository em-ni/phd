import gmsh
from math import cos, sin, radians


def create_cylinder_with_four_cavities(outer_radius, height, cavity_height, cavity_radius, cavity_distance_from_center, tip_offset, mesh_size):
    """
    CAD model of a cylinder with 4 cavities, placed at 90° from each other equally spaced from the cylinder center.
    """
    # Initialize the gmsh model
    gmsh.initialize()
    gmsh.model.add("Cylinder with Four Cavities")

    # Create the outer cylinder
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)

    # Initialize a list to store the cavity volumes
    cavity_volumes = []

    # Create the cavities
    for i in range(4):
        angle = i * 90  # Angle for the cavity placement
        x = cavity_distance_from_center * cos(radians(angle))
        y = cavity_distance_from_center * sin(radians(angle))

        # Create cavity cylinder
        cavity = gmsh.model.occ.addCylinder(x, y, tip_offset, 0, 0, cavity_height, cavity_radius)
        cavity_volumes.append((3, cavity))

    # Subtract the cavities from the outer cylinder
    gmsh.model.occ.cut([(3, outer_cylinder)], cavity_volumes)

    # Set the mesh size for all points in the model
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Synchronize the model
    gmsh.model.occ.synchronize()


def create_cavities(outer_radius, height, cavity_height, cavity_radius, cavity_distance_from_center, tip_offset, mesh_size):
    """
    Create and save individual cavities as separate files.
    Assumes Gmsh is already initialized.
    """
    for i in range(4):
        gmsh.model.add(f"Cavity_{i}")
        angle = i * 90  # Angle for the cavity placement
        x = cavity_distance_from_center * cos(radians(angle))
        y = cavity_distance_from_center * sin(radians(angle))

        # Create cavity cylinder
        gmsh.model.occ.addCylinder(x, y, tip_offset, 0, 0, cavity_height, cavity_radius)

        # Set the mesh size and synchronize
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

        # Save the model to file
        filename = f"data/mesh/multidof/cavity_{i}.stl"
        gmsh.write(filename)
        print(f"Cavity {i} saved as {filename}")

        # Remove the cavity model to prepare for the next cavity
        gmsh.model.remove()


def save_gmsh_model(filename):
    extension = filename.split('.')[-1].lower()
    
    if extension in ["stl", "step", "vtk"]:
        gmsh.model.mesh.generate(3)
        gmsh.write(filename)
    else:
        raise ValueError("Unsupported file format.")
    

if __name__ == "__main__":

    ## multi dof robot
    # Data are in mm
    outer_radius = 0.45
    height = 7
    cavity_height = 5
    cavity_radius = 0.1
    cavity_distance_from_center = 0.3
    tip_offset = 0.25

    # Set mesh_size to a desired value. Smaller values will produce a finer mesh.
    mesh_size = 0.1
    create_cylinder_with_four_cavities(outer_radius, height, cavity_height, cavity_radius, cavity_distance_from_center, tip_offset, mesh_size)
    
    # Create and save cavities
    create_cavities(outer_radius, height, cavity_height, cavity_radius, cavity_distance_from_center, tip_offset, mesh_size)
    
    save_gmsh_model("data/mesh/multidof/cylinder_with_four_cavities.stl")
    save_gmsh_model("data/mesh/multidof/cylinder_with_four_cavities.step")
    save_gmsh_model("data/mesh/multidof/cylinder_with_four_cavities.vtk")

    gmsh.finalize()

    