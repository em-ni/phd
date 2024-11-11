import argparse
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
    Robustly interpolate a line through 3D points.
    :param points: np.array of 3D points
    :param num_points: Number of points in the interpolated line
    :return: np.array of interpolated points
    """
    try:
        points = preprocess_points(np.array(points))
        if num_points is None:
            num_points = len(points) * 2  # Default to double the input points

        # Compute the cumulative distance along the line
        distance = np.cumsum(np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1)])
        distance_normalized = distance / distance[-1]

        # Interpolate for each dimension
        interpolated_points = np.zeros((num_points, 3))
        for i in range(3):
            interpolator = interp1d(distance_normalized, points[:, i], kind="cubic")
            interpolated_points[:, i] = interpolator(np.linspace(0, 1, num_points))

        return interpolated_points

    except Exception as e:
        print(f"An error occurred: {e}")
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
    num_points=10, draw_tangent=True, draw_normal=True, draw_binormal=True
):
    # Load the .vtp file
    # line_model = pv.read("data/mesh/vascularmodel/0023_H_AO_MFS/sim/path.vtp")
    # line_model = pv.read("data/mesh/vascularmodel/0063_H_PULMGLN_SVD/sim/path.vtp")
    line_model = pv.read("data/mesh/easier_slam_test/centerline.vtp")
    print(f"points: {line_model.points.shape}")

    # Interpolate the line for smoothing
    interpolated_points = interpolate_line(line_model.points)

    # Compute tangent vectors for the interpolated points
    tangents = compute_tangent_vectors(interpolated_points)

    # Compute normal and binormal vectors for the interpolated points
    # normals = compute_normal_vectors(tangents)
    # binormals = compute_binormal_vectors(tangents, normals)

    # Compute the Frenet-Serret frame using the MRF algorithm
    normals, binormals = compute_MRF(tangents)

    # Create a plotter
    plotter = pv.Plotter()
    # Add the original line to the plotter
    plotter.add_mesh(line_model, color="blue", line_width=2)

    # Select random points from the interpolated line to display tangents
    random_indices = np.random.choice(
        len(interpolated_points), num_points, replace=False
    )
    
    # Draw the origin of the world frame
    plotter.add_arrows(
        np.zeros((1, 3)), np.array([[100, 0, 0]]), color="black", mag=1
    )
    plotter.add_arrows(
        np.zeros((1, 3)), np.array([[0, 100, 0]]), color="black", mag=1
    )
    plotter.add_arrows(
        np.zeros((1, 3)), np.array([[0, 0, 100]]), color="black", mag=1
    )

    # Add Frenet-Serret frames to the plotter for the random points
    for idx in random_indices:
        point = interpolated_points[idx]
        tangent = tangents[idx]
        normal = normals[idx]
        binormal = binormals[idx]

        # Draw the tangent vector
        if draw_tangent:
            plotter.add_arrows(
                point[np.newaxis], tangent[np.newaxis], color="green", mag=0.1
            )
        # Draw the normal vector
        if draw_normal:
            plotter.add_arrows(
                point[np.newaxis], normal[np.newaxis], color="red", mag=0.1
            )
        # Draw the binormal vector
        if draw_binormal:
            plotter.add_arrows(
                point[np.newaxis], binormal[np.newaxis], color="blue", mag=0.1
            )

    # Show the plotter
    plotter.show()
    return

def save_frames(input_path, output_path):
    # Load the .vtp file
    line_model = pv.read(input_path)
    print(f"points: {line_model.points.shape}")

    # Interpolate the line for smoothing
    interpolated_points = interpolate_line(line_model.points)

    # Compute tangent vectors for the interpolated points
    tangents = compute_tangent_vectors(interpolated_points)

    # Compute the Frenet-Serret frame using the MRF algorithm
    normals, binormals = compute_MRF(tangents)

    # For each point save the coordinates and the respective FS frame
    with open(output_path, "w") as file:
        for idx in range(len(interpolated_points)):
            point = interpolated_points[idx]
            # point = interpolated_points[idx] / 1000.0 # Convert to meters from mm
            tangent = tangents[idx]
            normal = normals[idx]
            binormal = binormals[idx]

            file.write(f"{point[0]}, {point[1]}, {point[2]}, {tangent[0]}, {tangent[1]}, {tangent[2]}, {normal[0]}, {normal[1]}, {normal[2]}, {binormal[0]}, {binormal[1]}, {binormal[2]}\n")


# Add at the beginning of the file, after the existing imports
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process centerline file and compute Frenet-Serret frames.')
    parser.add_argument('i', type=str, help='Path to the input centerline .vtp file')
    parser.add_argument('o', type=str, help='Path for the output frames .txt file')
    return parser.parse_args()


if __name__ == "__main__":
    # draw_FS_frames(
    #     num_points=689, draw_tangent=True, draw_normal=True, draw_binormal=True
    # )
    args = parse_arguments()
    save_frames(args.i, args.o)
