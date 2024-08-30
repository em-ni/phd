import argparse
import pyvista as pv # type: ignore

def main():
    # Define the command line arguments using argparse
    parser = argparse.ArgumentParser(description="Visualize 3D files.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file.")

    # Parse the arguments
    args = parser.parse_args()

    # Load the VTK file
    mesh = pv.read(args.input)

    # Create a plotter object
    plotter = pv.Plotter()

    # Add the mesh to the plotter
    plotter.add_mesh(mesh)

    # Display the plot
    plotter.show()

if __name__ == "__main__":
    main()

