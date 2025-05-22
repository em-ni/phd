import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import os


def filter_trajectory_positions(frames, sigma):
    """
    Applies a 1D Gaussian filter to the x, y, z coordinates of trajectory positions.
    Preserves original orientations.
    """
    # Ensure sigma is a float for comparison and for gaussian_filter1d
    try:
        float_sigma = float(sigma)
    except ValueError:
        print(
            f"[ERROR] Invalid sigma value for filtering: {sigma}. Cannot convert to float. Skipping filtering."
        )
        return frames

    if not frames or float_sigma <= 0.0:
        return frames

    positions = np.array([frame[:3, 3] for frame in frames])

    # Need at least a few points for meaningful filtering, gaussian_filter1d might also have minimums
    # depending on sigma and internal truncation. A simple check:
    if (
        positions.shape[0] < 3
    ):  # Arbitrary small number, can be tuned or made more robust
        print(
            "[WARNING] Not enough points to filter trajectory effectively, returning original."
        )
        return frames

    filtered_positions = np.empty_like(positions)
    for i in range(positions.shape[1]):  # Iterate over x, y, z (columns)
        # Using mode='nearest' to handle boundaries by extending with the edge value.
        # 'reflect' is another good option.
        filtered_positions[:, i] = gaussian_filter1d(
            positions[:, i], sigma=float_sigma, mode="nearest"
        )

    new_frames = []
    for i, original_frame in enumerate(frames):
        new_frame = np.copy(original_frame)
        new_frame[:3, 3] = filtered_positions[i]
        new_frames.append(new_frame)
    return new_frames


def get_depth_image(depthTex):
    """
    Returns the current depth image as a NumPy array (float32, values between 0.0 and 1.0).
    """
    # Retrieve the raw depth data from the texture.
    data = depthTex.getRamImage()
    if data is None or len(data) == 0:
        print("Depth image not ready yet!")
        return None
    # Convert the raw data to a NumPy array.
    depth_image = np.frombuffer(data, dtype=np.float32)
    # Reshape according to the texture's dimensions.
    depth_image.shape = (
        depthTex.getYSize(),
        depthTex.getXSize(),
        depthTex.getNumComponents(),
    )
    # Flip vertically (Panda3D's origin is bottom-left).
    depth_image = np.flipud(depth_image)

    return depth_image


def get_rotation_from_index(index, tangents, normals, binormals):
    """
    Returns a 3x3 rotation matrix from the FS frame at the given index.
    The rotation is built from the tangent, normal, and binormal vectors.
    """
    R = np.eye(3)
    R[:, 0] = tangents[index]
    R[:, 1] = normals[index]
    R[:, 2] = binormals[index]
    return R


def save_fs_frames_multibranch(
    data_folder,
    interpolated_points,
    tangents,
    normals,
    binormals,
    file_name="ball_fs.txt",
):
    """
    Save the FS frames in a format that can be read by the app.
    """
    # Save the FS frames in a format that can be read by the app.

    fs_frames_path = os.path.join(data_folder, "centerlines", file_name)
    # For each point, save the coordinates and the respective FS frame
    with open(fs_frames_path, "w") as file:
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

    print(f"[INFO] FS frames saved as {fs_frames_path}")
    return fs_frames_path


def trajectory_snapping(slam_frames, centerline_frames, threshold_radius):
    """
    Snaps the SLAM trajectory points to be within a certain radius of the centerline.
    If a SLAM point is further than threshold_radius from its closest centerline point,
    it's moved along the vector connecting them to be exactly at threshold_radius distance.
    Orietations are preserved from the original SLAM frames.
    """
    if not slam_frames or not centerline_frames:
        print("[WARNING] SLAM or centerline frames are empty, cannot perform snapping.")
        return []

    slam_positions = np.array([frame[:3, 3] for frame in slam_frames])
    centerline_positions = np.array([frame[:3, 3] for frame in centerline_frames])

    if centerline_positions.shape[0] == 0:
        print("[WARNING] Centerline positions are empty, cannot perform snapping.")
        return []

    # Build KDTree for efficient closest point search on the centerline
    centerline_tree = cKDTree(centerline_positions)

    snapped_slam_frames = []
    for i, slam_frame in enumerate(slam_frames):
        slam_pos = slam_positions[i]

        # Find the closest centerline point
        distance, closest_centerline_idx = centerline_tree.query(slam_pos)
        closest_centerline_pos = centerline_positions[closest_centerline_idx]

        snapped_pos = np.copy(slam_pos)
        if distance > threshold_radius:
            # Vector from centerline point to SLAM point
            vec_c_to_s = slam_pos - closest_centerline_pos
            # Normalize this vector (avoid division by zero if distance is already small, though covered by `if` )
            if distance > 1e-6:  # Check for non-zero distance
                unit_vec_c_to_s = vec_c_to_s / distance
                # New snapped position is on the sphere of threshold_radius around the centerline point
                snapped_pos = (
                    closest_centerline_pos + unit_vec_c_to_s * threshold_radius
                )

        # Create new frame with snapped position but original orientation
        new_frame = np.copy(slam_frame)
        new_frame[:3, 3] = snapped_pos
        snapped_slam_frames.append(new_frame)

    return snapped_slam_frames
