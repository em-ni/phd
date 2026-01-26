import heapq
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import os
import pyvista as pv
from utils.set_FS_frame import (  # Adjust this import path based on your project structure
    interpolate_line,
    compute_tangent_vectors,
    compute_MRF,
    smooth_vectors,
    interpolate_fs_frames,
)


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


def build_all_branches_path(app_config, data_folder):
    """
    Reads multiple branch VTP files, computes the FS frame for every point,
    and stacks them together (forward and then reverse for each branch).
    Returns a list of 4x4 FS frame matrices and a list of all points.
    """
    centerline_folder_name = app_config["PATHS"]["all_branches_folder"]
    centerline_folder_path = os.path.join(data_folder, centerline_folder_name)
    print(f"[INFO] Looking for .vtp files in: {centerline_folder_path}")

    branch_files = []
    if not os.path.isdir(centerline_folder_path):
        print(f"[ERROR] Directory not found: {centerline_folder_path}")
        return [], []

    for file in os.listdir(centerline_folder_path):
        if file.endswith(".vtp"):
            # Storing relative path from data_folder to match original logic
            full_path = os.path.join(centerline_folder_name, file)
            branch_files.append(full_path)

    if not branch_files:
        print(f"[ERROR] No .vtp files found in {centerline_folder_path}")
        return [], []

    print(f"[INFO] Found {len(branch_files)} branch files: {branch_files}")

    final_frames = []
    all_points_collected = (
        []
    )  # Renamed to avoid conflict with any 'points' variable if this were a class

    for branch_file_relative in branch_files:
        branch_file_relative = branch_file_relative.strip()
        # Construct absolute path for pv.read
        branch_path_abs = os.path.join(data_folder, branch_file_relative)
        print(f"[INFO] Processing branch file: {branch_path_abs}")
        try:
            branch_model = pv.read(branch_path_abs)
        except Exception as e:
            print(f"[ERROR] Could not read VTP file {branch_path_abs}: {e}")
            continue

        n_d = 0  # Discard the first n_d points
        if len(branch_model.points) > n_d:
            branch_points_list = [tuple(point) for point in branch_model.points[n_d:]]
        else:
            print(
                f"[WARNING] Branch {branch_file_relative} has fewer than {n_d} points ({len(branch_model.points)}). Skipping branch."
            )
            continue

        if not branch_points_list:
            print(
                f"[WARNING] Branch {branch_file_relative} resulted in an empty list of points after discarding. Skipping."
            )
            continue

        interp_points = interpolate_line(branch_points_list, num_points=1000)
        if interp_points is None or len(interp_points) == 0:
            print(
                f"[WARNING] Interpolation failed or yielded no points for branch {branch_file_relative}. Skipping."
            )
            continue

        branch_tangents = compute_tangent_vectors(interp_points)
        branch_tangents = smooth_vectors(
            branch_tangents, 10, 10
        )  # Sigma = 10, Window size = 10
        branch_normals, branch_binormals = compute_MRF(branch_tangents)

        # FORWARD TRAVEL
        for i, pt in enumerate(interp_points):
            fs_frame = np.eye(4)
            fs_frame[:3, 0] = branch_tangents[i]
            fs_frame[:3, 1] = branch_normals[i]
            fs_frame[:3, 2] = branch_binormals[i]
            fs_frame[:3, 3] = pt

            if len(final_frames) > 0 and i == 0:
                extra_frames = interpolate_fs_frames(
                    final_frames[-1], fs_frame, num_points=10
                )
                final_frames.extend(extra_frames)
                all_points_collected.extend([f[:3, 3] for f in extra_frames])
            else:
                final_frames.append(fs_frame)
                all_points_collected.append(pt)

        # RETURN TRAVEL (skip the last point of forward)
        if len(interp_points) > 1:  # Need at least two points to have a return path
            for i, pt in enumerate(
                interp_points[-2::-1]
            ):  # From second to last, backwards
                idx = len(interp_points) - 2 - i
                fs_frame = np.eye(4)
                # For return travel, tangent might be inverted depending on desired camera orientation.
                # Original code uses the same tangent: fs_frame[:3, 0] = branch_tangents[idx]
                # If camera should "look back", invert: fs_frame[:3, 0] = -branch_tangents[idx]
                # Keeping original behavior:
                fs_frame[:3, 0] = branch_tangents[idx]
                fs_frame[:3, 1] = branch_normals[
                    idx
                ]  # Normal might also need adjustment if tangent is flipped
                fs_frame[:3, 2] = branch_binormals[idx]
                fs_frame[:3, 3] = pt
                final_frames.append(fs_frame)
                all_points_collected.append(pt)

    if not final_frames:
        print("[WARNING] No frames were generated from any branch.")
        return [], []

    print(f"[INFO] Initial number of points: {len(final_frames)}")
    divide_factor = 4  # TODO: Make this configurable if needed
    final_frames_reduced = final_frames[::divide_factor]
    all_points_reduced = all_points_collected[::divide_factor]
    print(f"[INFO] Reduced number of points: {len(final_frames_reduced)}")

    return final_frames_reduced, all_points_reduced


def build_random_branches_path(branch_files, data_folder):
    """
    Given a list of relative branch .vtp files, computes the FS frame for every point,
    and stacks them together (forward and then reverse for each branch).
    Returns a list of 4x4 FS frame matrices and a list of all points.
    """
    import pyvista as pv
    from utils.set_FS_frame import (
        interpolate_line,
        compute_tangent_vectors,
        compute_MRF,
        smooth_vectors,
        interpolate_fs_frames,
    )
    import numpy as np
    import os

    final_frames = []
    all_points_collected = []
    for branch_file_relative in branch_files:
        branch_file_relative = branch_file_relative.strip()
        branch_path_abs = os.path.join(data_folder, branch_file_relative)
        print(f"[INFO] Processing branch file: {branch_path_abs}")
        try:
            branch_model = pv.read(branch_path_abs)
        except Exception as e:
            print(f"[ERROR] Could not read VTP file {branch_path_abs}: {e}")
            continue
        n_d = 0
        if len(branch_model.points) > n_d:
            branch_points_list = [tuple(point) for point in branch_model.points[n_d:]]
        else:
            print(
                f"[WARNING] Branch {branch_file_relative} has fewer than {n_d} points ({len(branch_model.points)}). Skipping branch."
            )
            continue
        if not branch_points_list:
            print(
                f"[WARNING] Branch {branch_file_relative} resulted in an empty list of points after discarding. Skipping."
            )
            continue
        interp_points = interpolate_line(branch_points_list, num_points=1000)
        if interp_points is None or len(interp_points) == 0:
            print(
                f"[WARNING] Interpolation failed or yielded no points for branch {branch_file_relative}. Skipping."
            )
            continue
        branch_tangents = compute_tangent_vectors(interp_points)
        branch_tangents = smooth_vectors(branch_tangents, 10, 10)
        branch_normals, branch_binormals = compute_MRF(branch_tangents)
        # FORWARD TRAVEL
        for i, pt in enumerate(interp_points):
            fs_frame = np.eye(4)
            fs_frame[:3, 0] = branch_tangents[i]
            fs_frame[:3, 1] = branch_normals[i]
            fs_frame[:3, 2] = branch_binormals[i]
            fs_frame[:3, 3] = pt
            if len(final_frames) > 0 and i == 0:
                extra_frames = interpolate_fs_frames(
                    final_frames[-1], fs_frame, num_points=10
                )
                final_frames.extend(extra_frames)
                all_points_collected.extend([f[:3, 3] for f in extra_frames])
            else:
                final_frames.append(fs_frame)
                all_points_collected.append(pt)
        # RETURN TRAVEL (skip the last point of forward)
        if len(interp_points) > 1:
            for i, pt in enumerate(interp_points[-2::-1]):
                idx = len(interp_points) - 2 - i
                fs_frame = np.eye(4)
                fs_frame[:3, 0] = branch_tangents[idx]
                fs_frame[:3, 1] = branch_normals[idx]
                fs_frame[:3, 2] = branch_binormals[idx]
                fs_frame[:3, 3] = pt
                final_frames.append(fs_frame)
                all_points_collected.append(pt)
    if not final_frames:
        print("[WARNING] No frames were generated from any branch.")
        return [], []
    print(f"[INFO] Initial number of points: {len(final_frames)}")
    divide_factor = 4
    final_frames_reduced = final_frames[::divide_factor]
    all_points_reduced = all_points_collected[::divide_factor]
    print(f"[INFO] Reduced number of points: {len(final_frames_reduced)}")
    return final_frames_reduced, all_points_reduced


def curvilinear_abscissa(
    current_point_arr, interpolated_points_arr, all_branches_bool_str, record_mode_bool
):
    """
    Computes the curvilinear abscissa (distance along the path).
    Uses a graph-based approach for multi-branch paths if all_branches_bool_str is "1" and record_mode_bool is True.
    Otherwise, uses a simpler cumulative distance for single branches.
    """
    current_point = np.asarray(current_point_arr)
    interpolated_points = np.asarray(interpolated_points_arr)

    if interpolated_points is None or len(interpolated_points) == 0:
        return 0.0

    if all_branches_bool_str == "1" and record_mode_bool:
        # Graph-based approach for multi-branch
        n = len(interpolated_points)
        if n < 2:
            return 0.0

        diffs = interpolated_points[1:] - interpolated_points[:-1]
        if len(diffs) == 0:  # Only one point in interpolated_points
            return np.linalg.norm(interpolated_points[0] - current_point)

        mean_distance = np.mean(np.linalg.norm(diffs, axis=1))
        threshold = 2 * mean_distance

        tree = cKDTree(interpolated_points)
        graph = {i: [] for i in range(n)}
        for i in range(n):
            neighbors_indices = tree.query_ball_point(
                interpolated_points[i], r=threshold
            )
            for j in neighbors_indices:
                if i == j:
                    continue
                dist_ij = np.linalg.norm(
                    interpolated_points[i] - interpolated_points[j]
                )
                graph[i].append((j, dist_ij))

        current_idx_on_path = np.argmin(
            np.linalg.norm(interpolated_points - current_point, axis=1)
        )
        target_idx_on_path = 0  # Assuming distance to the start of the path

        distances_from_current = {i: float("inf") for i in range(n)}
        distances_from_current[current_idx_on_path] = 0.0
        pq = [(0.0, current_idx_on_path)]  # (distance, node_index)

        path_found = False
        while pq:
            d, u_idx = heapq.heappop(pq)

            if u_idx == target_idx_on_path:
                path_found = True
                break

            if d > distances_from_current[u_idx]:
                continue

            for v_idx, weight_uv in graph.get(u_idx, []):
                if (
                    distances_from_current[u_idx] + weight_uv
                    < distances_from_current[v_idx]
                ):
                    distances_from_current[v_idx] = (
                        distances_from_current[u_idx] + weight_uv
                    )
                    heapq.heappush(pq, (distances_from_current[v_idx], v_idx))

        return (
            distances_from_current[target_idx_on_path]
            if path_found and distances_from_current[target_idx_on_path] != float("inf")
            else 0.0
        )

    else:
        # Simpler cumulative distance for single branches or non-record mode
        if len(interpolated_points) == 0:
            return 0.0
        distances_to_current = np.linalg.norm(
            interpolated_points - current_point, axis=1
        )
        closest_index = np.argmin(distances_to_current)

        total_distance = 0.0
        for i in range(closest_index):
            segment = interpolated_points[i + 1] - interpolated_points[i]
            total_distance += np.linalg.norm(segment)

        # Add distance from the closest point on path to the actual current_point
        # This makes it more accurate if current_point is slightly off the path.
        # total_distance += distances_to_current[closest_index] # Optional: depends on desired definition
        return total_distance


def get_vtp_line_points(path_path_str):
    """
    Loads a .vtp file and returns its points as a list of tuples.
    """
    try:
        line_model = pv.read(path_path_str)
        points = [tuple(point) for point in line_model.points]
        return points
    except Exception as e:
        print(f"[ERROR] Could not read VTP file {path_path_str}: {e}")
        return []
