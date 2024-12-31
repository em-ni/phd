import argparse
import os
import pyvista as pv
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


def smooth_vectors(vectors, window_size=10, passes=1):
    """Apply a simple smoothing filter to vectors multiple times."""
    smoothed_vectors = vectors.copy()
    for _ in range(passes):
        # Pad the vectors at the start and end
        pad_width = window_size // 2
        padded_vectors = np.pad(
            smoothed_vectors, ((pad_width, pad_width), (0, 0)), mode="edge"
        )

        # Apply a uniform filter
        for i in range(len(smoothed_vectors)):
            smoothed_vectors[i] = np.mean(padded_vectors[i : i + window_size], axis=0)

        # Normalize the smoothed vectors
        norms = np.linalg.norm(smoothed_vectors, axis=1)
        smoothed_vectors = smoothed_vectors / norms[:, np.newaxis]

    return smoothed_vectors


def preprocess_points(points, epsilon=1e-5):
    """
    Preprocess the points by removing duplicates or points that are too close to each other.
    :param points: np.array of points
    :param epsilon: Minimum distance between points
    :return: np.array of preprocessed points
    """
    # Check if points array is at least one-dimensional
    if points.ndim < 1:
        print("Error: Points data must be at least one-dimensional.")
        exit()

    # Check for a minimum number of points
    if len(points) < 2:
        print("Error: Not enough points to compute differences.")
        exit()

    unique_points = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - unique_points[-1]) > epsilon:
            unique_points.append(point)
    return np.array(unique_points)


def interpolate_line(points, num_points=None):
    """
    Robustly interpolate a line through 3D points, handling duplicates and non-monotonic data.
    :param points: np.array of 3D points
    :param num_points: Number of points in the interpolated line
    :return: np.array of interpolated points
    """
    try:
        points = preprocess_points(np.array(points))
        if num_points is None:
            num_points = len(points) * 2  # Default to double the input points

        # Compute the cumulative distance along the line
        distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])

        # Handle duplicates in distances
        distances, unique_indices = np.unique(distances, return_index=True)
        points = points[unique_indices]

        # Ensure that distances are strictly increasing
        if len(distances) < 2 or np.any(np.diff(distances) <= 0):
            print("Error: Not enough unique points to interpolate.")
            return None

        # Normalize the distances
        distance_normalized = distances / distances[-1]

        # Interpolate for each dimension
        interpolated_points = np.zeros((num_points, 3))
        for i in range(3):
            interpolator = interp1d(
                distance_normalized,
                points[:, i],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated_points[:, i] = interpolator(np.linspace(0, 1, num_points))

        return interpolated_points

    except Exception as e:
        print(f"An error occurred during interpolation: {e}")
        return None


def compute_tangent_vectors(interpolated_points):
    # Compute tangent vectors using finite differences
    tangents = np.diff(interpolated_points, axis=0)
    tangents = np.vstack(
        (tangents, tangents[-1])
    )  # Ensure same number of tangents as points
    # Normalize the tangent vectors
    norms = np.linalg.norm(tangents, axis=1)
    tangents = tangents / norms[:, np.newaxis]
    tangents = smooth_vectors(tangents)  # Smooth the tangent vectors
    return tangents


def compute_MRF(tangents):
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    # Initialize the normal for the first point
    # Make sure this is orthogonal to the first tangent
    normals[0] = np.array([tangents[0][1], -tangents[0][0], 0])
    normals[0] /= np.linalg.norm(normals[0])

    # Compute the binormal for the first point
    binormals[0] = np.cross(tangents[0], normals[0])

    # Propagate the frame along the curve
    for i in range(1, len(tangents)):
        # Project the previous normal onto the plane orthogonal to the current tangent
        normals[i] = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        normals[i] /= np.linalg.norm(normals[i])

        # Compute the binormal as the cross product of tangent and normal
        binormals[i] = np.cross(tangents[i], normals[i])

    normals = smooth_vectors(normals)
    binormals = smooth_vectors(binormals)

    return normals, binormals


def compute_normal_vectors(tangents):
    # Use finite differences to approximate normal vectors
    normals = np.diff(tangents, axis=0)
    normals = np.vstack(
        (normals, normals[-1])
    )  # Ensure same number of normals as tangents

    # Replace any zero vectors with NaN to avoid division by zero
    norms = np.linalg.norm(normals, axis=1)
    zeros = norms < 1e-6
    normals[zeros] = np.nan

    # Normalize the normal vectors, skipping any NaN entries
    normals = normals / norms[:, None]

    # Handle NaNs if there were any zero vectors
    normals[zeros] = [0, 0, 0]  # or any other placeholder for zero vectors

    normals = smooth_vectors(normals)  # Smooth the normal vectors
    return normals


def compute_binormal_vectors(tangents, normals):
    # Compute binormal vectors as the cross product of tangent and normal vectors
    binormals = np.cross(tangents, normals)
    # Normalize the binormal vectors
    norms = np.linalg.norm(binormals, axis=1)
    zeros = norms < 1e-6
    binormals[zeros] = np.nan
    binormals = binormals / norms[:, np.newaxis]
    binormals[zeros] = [0, 0, 0]  # or any other placeholder for zero vectors
    binormals = smooth_vectors(binormals)  # Smooth the binormal vectors
    return binormals


def draw_FS_frames(
    num_points=10, draw_tangent=True, draw_normal=True, draw_binormal=True, path=None
):
    # Load the .vtp file
    # line_model = pv.read("data/mesh/vascularmodel/0023_H_AO_MFS/sim/path.vtp")
    # line_model = pv.read("data/mesh/vascularmodel/0063_H_PULMGLN_SVD/sim/path.vtp")
    # line_model = pv.read("data/mesh/easier_slam_test/centerline.vtp")
    line_model = pv.read(path)
    # line_model = pv.read("data/mesh/easier_slam_test/path.vtp")
    n_cells = line_model.n_cells
    print(f"points: {line_model.points.shape}")
    print(f"Number of branches (cells): {n_cells}")

    # Create a plotter
    plotter = pv.Plotter()

    # Draw the origin of the world frame
    plotter.add_arrows(np.zeros((1, 3)), np.array([[10, 0, 0]]), color="red", mag=1)
    plotter.add_arrows(np.zeros((1, 3)), np.array([[0, 10, 0]]), color="green", mag=1)
    plotter.add_arrows(np.zeros((1, 3)), np.array([[0, 0, 10]]), color="blue", mag=1)

    for i in range(n_cells):
        # Extract the i-th cell (branch)
        single_line = line_model.extract_cells(i)
        points = single_line.points
        print(f"Processing branch {i} with {len(points)} points")

        if len(points) < 2:
            print(f"Skipping branch {i} due to insufficient points.")
            continue

        # Interpolate the line for smoothing
        interpolated_points = interpolate_line(points)

        if interpolated_points is None or len(interpolated_points) == 0:
            print(f"Skipping branch {i} due to interpolation failure.")
            continue

        # Compute tangent vectors for the interpolated points
        tangents = compute_tangent_vectors(interpolated_points)

        # Compute the Frenet-Serret frame using the MRF algorithm
        normals, binormals = compute_MRF(tangents)

        # Define a list of distinct colors for branches
        branch_colors = ["red", "green", "blue", "orange", "purple", "cyan", "yellow"]
        branch_color = branch_colors[i % len(branch_colors)]  # Cycle through colors

        # Draw the very first point of the branch
        plotter.add_points(interpolated_points[0], color=branch_color, point_size=10)

        # Add the branch to the plotter
        plotter.add_mesh(single_line, color="blue", line_width=2)

        # Select random points from the interpolated line to display frames
        random_indices = np.random.choice(
            len(interpolated_points),
            min(num_points, len(interpolated_points)),
            replace=False,
        )

        # Add Frenet-Serret frames to the plotter for the random points
        for idx in random_indices:
            point = interpolated_points[idx]
            tangent = tangents[idx]
            normal = normals[idx]
            binormal = binormals[idx]

            mag = 1.0
            # Draw the tangent vector
            if draw_tangent:
                plotter.add_arrows(
                    point[np.newaxis], tangent[np.newaxis], color="green", mag=mag
                )
            # Draw the normal vector
            if draw_normal:
                plotter.add_arrows(
                    point[np.newaxis], normal[np.newaxis], color="red", mag=mag
                )
            # Draw the binormal vector
            if draw_binormal:
                plotter.add_arrows(
                    point[np.newaxis], binormal[np.newaxis], color="blue", mag=mag
                )

    # Show the plotter
    plotter.show()
    return


def save_frames_single_branch(input_path):

    output_path = input_path.replace(".vtp", "_frames.txt")

    # Load the .vtp file
    line_model = pv.read(input_path)
    n_cells = line_model.n_cells
    print(f"Number of branches (cells): {n_cells}")

    # Open the output file once
    with open(output_path, "w") as file:
        for i in range(n_cells):
            # Extract the i-th cell (branch)
            single_line = line_model.extract_cells(i)
            points = single_line.points
            print(f"Processing branch {i} with {len(points)} points")

            # Convert from mm to m
            points = points / 1000

            if len(points) < 2:
                print(f"Skipping branch {i} due to insufficient points.")
                continue

            # Interpolate the line for smoothing
            interpolated_points = interpolate_line(points)

            if interpolated_points is None or len(interpolated_points) == 0:
                print(f"Skipping branch {i} due to interpolation failure.")
                continue

            # Compute tangent vectors for the interpolated points
            tangents = compute_tangent_vectors(interpolated_points)

            # Compute the Frenet-Serret frame using the MRF algorithm
            normals, binormals = compute_MRF(tangents)

            # For each point, save the coordinates and the respective FS frame
            for idx in range(len(interpolated_points)):
                point = interpolated_points[idx]
                tangent = tangents[idx]
                normal = normals[idx]
                binormal = binormals[idx]

                # Write to file
                file.write(
                    f"{point[0]}, {point[1]}, {point[2]}, "
                    f"{tangent[0]}, {tangent[1]}, {tangent[2]}, "
                    f"{normal[0]}, {normal[1]}, {normal[2]}, "
                    f"{binormal[0]}, {binormal[1]}, {binormal[2]}\n"
                )


def save_frames_all_branches(input_paths):

    output_path = os.path.join(
        os.path.dirname(input_paths[0]), "centerline_frames_all_branches.txt"
    )
    all_frames = []

    # Open the output file once
    with open(output_path, "w") as file:
        for input_path in input_paths:
            # Load the .vtp file
            line_model = pv.read(input_path)
            n_cells = line_model.n_cells
            print(f"Number of branches (cells): {n_cells}")
            branch_frames = []

            for i in range(n_cells):
                # Extract the i-th cell (branch)
                single_line = line_model.extract_cells(i)
                points = single_line.points
                print(f"Processing branch {i} with {len(points)} points")

                # Convert from mm to m
                points = points / 1000

                if len(points) < 2:
                    print(f"Skipping branch {i} due to insufficient points.")
                    continue

                # Interpolate the line for smoothing
                interpolated_points = interpolate_line(points)

                if interpolated_points is None or len(interpolated_points) == 0:
                    print(f"Skipping branch {i} due to interpolation failure.")
                    continue

                # Compute tangent vectors for the interpolated points
                tangents = compute_tangent_vectors(interpolated_points)

                # Compute the Frenet-Serret frame using the MRF algorithm
                normals, binormals = compute_MRF(tangents)

                # For each point, save the coordinates and the respective FS frame
                for idx in range(len(interpolated_points)):
                    point = interpolated_points[idx]
                    tangent = tangents[idx]
                    normal = normals[idx]
                    binormal = binormals[idx]

                    # Add to branch frames
                    branch_frames.append(
                        [
                            point[0],
                            point[1],
                            point[2],
                            tangent[0],
                            tangent[1],
                            tangent[2],
                            normal[0],
                            normal[1],
                            normal[2],
                            binormal[0],
                            binormal[1],
                            binormal[2],
                        ]
                    )

            # Add branch frames to all frames
            all_frames.append(branch_frames)

        # Process and write each branch separately
        for branch_frames in all_frames:

            # Write frames from this branch
            for frame in branch_frames:
                file.write(
                    f"{frame[0]}, {frame[1]}, {frame[2]}, "
                    f"{frame[3]}, {frame[4]}, {frame[5]}, "
                    f"{frame[6]}, {frame[7]}, {frame[8]}, "
                    f"{frame[9]}, {frame[10]}, {frame[11]}\n"
                )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process centerline file and compute Frenet-Serret frames."
    )
    parser.add_argument(
        "i", type=str, help="Path to the input centerline .vtp file or folder"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Check if input is a file or directory
    if os.path.isfile(args.i) and args.i.endswith(".vtp"):
        # Process single .vtp file
        save_frames_single_branch(args.i)
    elif os.path.isdir(args.i):
        # Process all centerline_b*.vtp files in directory
        vtp_files = [
            os.path.join(args.i, f)
            for f in os.listdir(args.i)
            if f.startswith("centerline_b") and f.endswith(".vtp")
        ]

        save_frames_all_branches(vtp_files)

    else:
        print(
            "Error: Input must be either a .vtp file or a directory containing .vtp files"
        )
