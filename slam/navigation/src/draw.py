import numpy as np
from panda3d.core import (  # type: ignore
    LineSegs,
    LVector3f,
    TransparencyAttrib,
    NodePath,
    GeomVertexFormat,
    GeomVertexData,
    Geom,
    GeomTriangles,
    GeomNode,
)
from utils.set_FS_frame import interpolate_line


def draw_elements(app):
    if app.draw_frames_bool == "1":
        # Draw some frames
        draw_FS_frames(
            app,
            draw_tangent=True,
            draw_normal=True,
            draw_binormal=True,
        )

    if app.draw_circles_bool == "1":
        # Drawing circles
        draw_circles_around_points(app, radius=0.2, num_segments=50)

    if app.view_mode == "tp" and app.draw_reference_frames_bool == "1":
        # Draw the origin frame
        draw_origin_frame(app)

        # Draw the base frame
        draw_base_frame(app)

    # Draw the robot tip
    draw_robot_tip(app)


def draw_circles_around_points(app, radius=1, num_segments=10):
    for i, center in enumerate(app.interpolated_points):
        if i >= len(app.normals) or i >= len(app.binormals):
            print(
                f"[WARNING] draw_circles_around_points: Index {i} out of bounds for normals/binormals."
            )
            continue
        normal = app.normals[i]
        binormal = app.binormals[i]

        # Debug: Print normal and binormal
        # print(f"Point {i}: Normal = {normal}, Binormal = {binormal}")

        # Generate circle points
        circle_points = []
        for j in range(num_segments):
            angle = 2 * np.pi * j / num_segments
            # Ensure normal and binormal are numpy arrays for vectorized operations
            dx = np.cos(angle) * np.array(normal)
            dy = np.sin(angle) * np.array(binormal)
            point = np.array(center) + radius * (dx + dy)
            circle_points.append(point)

        # Debug: Print first few points of each circle
        # print(f"Circle {i} points: {circle_points[:3]}")

        # Draw the circle
        draw_circle(app, circle_points)


def draw_circle(app, points):  # points are list of np.array world coordinates
    parent_node, offset_vec = app.get_scene_graph_parent_and_offset()

    circle = LineSegs()  # type: ignore
    circle.setThickness(5.0)  # Increased thickness
    circle.setColor(1, 1, 0, 1)  # Changed color to yellow for better visibility

    if not points:
        return

    # Convert points to LVecBase3f, make them local to parent_node, and draw the circle
    for i, point_np_world in enumerate(points):
        point_world = LVector3f(point_np_world[0], point_np_world[1], point_np_world[2])  # type: ignore
        point_local = point_world - offset_vec
        if i == 0:
            circle.moveTo(point_local)
        else:
            circle.drawTo(point_local)
    # Connect back to the first point
    first_point_np_world = points[0]
    first_point_world = LVector3f(first_point_np_world[0], first_point_np_world[1], first_point_np_world[2])  # type: ignore
    first_point_local = first_point_world - offset_vec
    circle.drawTo(first_point_local)

    # Add the circle to the scene
    circle_node_geom = circle.create()
    parent_node.attachNewNode(circle_node_geom)


def draw_FS_frames(
    app,
    draw_tangent=True,
    draw_normal=True,
    draw_binormal=True,
):
    # Draw the frames
    for i, point in enumerate(app.interpolated_points):
        if draw_tangent:
            draw_vector(app, point, app.tangents[i], (1, 0, 0, 1))  # Red for tangent
        if draw_normal:
            draw_vector(app, point, app.normals[i], (0, 1, 0, 1))  # Green for normal
        if draw_binormal:
            draw_vector(app, point, app.binormals[i], (0, 0, 1, 1))  # Blue for binormal
        pass


def draw_vector(app, point_world_np, direction_world_np, color):
    """Draw a vector from a point in a specified direction with a given color."""
    parent_node, offset_vec = app.get_scene_graph_parent_and_offset()

    line = LineSegs()  # type: ignore
    line.setThickness(5.0)  # Increased thickness
    line.setColor(*color)  # Set the color

    start_world = LVector3f(point_world_np[0], point_world_np[1], point_world_np[2])  # type: ignore
    direction_vec_world = LVector3f(direction_world_np[0], direction_world_np[1], direction_world_np[2])  # type: ignore

    start_local = start_world - offset_vec
    # Direction vectors are not affected by the parent's translation for calculating the end point relative to start_local
    end_local = start_local + direction_vec_world

    line.moveTo(start_local)
    line.drawTo(end_local)

    # Add the line to the scene
    line_node_geom = line.create()
    parent_node.attachNewNode(line_node_geom)


def draw_origin_frame(app):
    # Create a LineSegs object to draw the frame
    frame = LineSegs()  # type: ignore
    frame.setThickness(5.0)  # Set a reasonable thickness

    # Draw the X axis in red
    frame.setColor(1, 0, 0, 1)  # Red color
    frame.moveTo(0, 0, 0)
    frame.drawTo(1 * 0.5, 0, 0)

    # Draw the Y axis in green
    frame.setColor(0, 1, 0, 1)  # Green color
    frame.moveTo(0, 0, 0)
    frame.drawTo(0, 1 * 0.5, 0)

    # Draw the Z axis in blue
    frame.setColor(0, 0, 1, 1)  # Blue color
    frame.moveTo(0, 0, 0)
    frame.drawTo(0, 0, 1 * 0.5)

    # Create a node to attach the frame to
    frame_node_path = app.render.attachNewNode("OriginFrame")

    # Create the frame geometry and attach it directly
    frame_geom = frame.create()
    frame_node_path.attachNewNode(frame_geom)

    # Set the scale of the frame
    frame_node_path.setScale(1)  # Scale to a reasonable size

    # Return the node for later reference
    return frame_node_path


def draw_base_frame(app):
    if not hasattr(app, "o_T_w") or app.o_T_w is None:
        print("[WARNING] o_T_w not initialized. Cannot draw base frame.")
        return None

    w_T_o = np.linalg.inv(app.o_T_w)

    # Draw w_T_o frame
    frame = LineSegs()  # type: ignore
    frame.setThickness(5.0)  # Set thickness

    # Extract position and axes from w_T_o matrix
    position = w_T_o[:3, 3]
    x_axis = w_T_o[:3, 0] * 0.5
    y_axis = w_T_o[:3, 1] * 0.5
    z_axis = w_T_o[:3, 2] * 0.5

    # Draw the X axis in red
    frame.setColor(1, 0, 0, 1)  # Red
    frame.moveTo(*position)
    frame.drawTo(*(position + x_axis))

    # Draw the Y axis in green
    frame.setColor(0, 1, 0, 1)  # Green
    frame.moveTo(*position)
    frame.drawTo(*(position + y_axis))

    # Draw the Z axis in blue
    frame.setColor(0, 0, 1, 1)  # Blue
    frame.moveTo(*position)
    frame.drawTo(*(position + z_axis))

    # Create a node to attach the frame
    frame_node_path = app.render.attachNewNode("BaseFrame")

    # Create the frame geometry and attach it
    frame_geom = frame.create()
    frame_node_path.attachNewNode(frame_geom)

    # Set scale of the frame
    frame_node_path.setScale(1)

    # Return the node for later reference
    return frame_node_path


def draw_robot_tip(app):
    if app.view_mode == "fp":  # Prevent drawing in first-person view
        # If there was an old node, ensure it's removed
        if hasattr(app, "robot_tip_node") and app.robot_tip_node:
            app.robot_tip_node.removeNode()
            app.robot_tip_node = None
        if hasattr(app, "light_cone_geom_np") and app.light_cone_geom_np:
            app.light_cone_geom_np.removeNode()
            app.light_cone_geom_np = None
        return

    if app.results_mode:
        return

    parent_node, offset_vec = app.get_scene_graph_parent_and_offset()

    if hasattr(app, "robot_tip_node") and app.robot_tip_node:
        app.robot_tip_node.removeNode()  # Remove the old node if it exists
        app.robot_tip_node = None  # Clear reference

    try:
        robot_tip_visual = app.loader.loadModel("models/misc/sphere")
    except Exception as e:
        print(f"[ERROR] Could not load robot tip model: models/smiley. {e}")
        # Fallback to a simple sphere or just a NodePath if model fails
        vdata = GeomVertexData("fallback_tip_geom", GeomVertexFormat.getV3n3cpt2(), Geom.UHStatic)  # type: ignore
        geom = Geom(vdata)  # type: ignore
        tris = GeomTriangles(Geom.UHStatic)  # type: ignore
        node = GeomNode("fallback_tip_node")  # type: ignore
        node.addGeom(geom)
        robot_tip_visual = NodePath(node)

    robot_tip_visual.setScale(1)  # Scale to appropriate size
    robot_tip_visual.setColor(0, 1, 0, 1)  # Set color to green

    robot_tip_pos_world = LVector3f(0, 0, 0)  # type: ignore Default if tip not set
    if hasattr(app, "robot_tip") and app.robot_tip is not None:
        robot_tip_pos_world = LVector3f(*app.robot_tip)  # type: ignore

    # Create a new node parented to parent_node (render or pivot)
    app.robot_tip_node = parent_node.attachNewNode("RobotTipNode")
    robot_tip_visual.reparentTo(
        app.robot_tip_node
    )  # smiley is now child of RobotTipNode

    # Position robot_tip_visual locally so its world position is robot_tip_pos_world
    robot_tip_visual_local_pos = robot_tip_pos_world - offset_vec
    robot_tip_visual.setPos(robot_tip_visual_local_pos)

    # Draw the light cone geometry, passing the parent and offset
    draw_light_cone_geom(app, parent_node, offset_vec)


def draw_light_cone_geom(app, parent_node, offset_vec):
    """
    Draws a translucent cone geometry that starts at the robot tip
    and extends in the tangent direction.
    """
    # Remove old cone if it exists
    if hasattr(app, "light_cone_geom_np") and app.light_cone_geom_np:
        app.light_cone_geom_np.removeNode()

    # Load cone model
    cone_model = app.loader.loadModel("data/icons/cone.obj")
    cone_model.setScale(0.5)
    cone_model.setColor(1, 1, 0, 0.3)  # Slightly yellow, partial alpha
    cone_model.setTransparency(TransparencyAttrib.MAlpha)  # type: ignore

    # Find the index of the closest point to the robot tip on the centerline
    # This logic for tangent_vector is preserved from your snippet
    if (
        hasattr(app, "interpolated_points")
        and len(app.interpolated_points) > 0
        and hasattr(app, "robot_tip")
        and app.robot_tip is not None
    ):  # Added None check for app.robot_tip for safety before use
        distances = np.linalg.norm(app.interpolated_points - app.robot_tip, axis=1)
        closest_index = np.argmin(distances)

        # Get the tangent vector at this point
        if closest_index < len(app.tangents):
            direction_vector = app.tangents[closest_index]
            tangent_vector = LVector3f(*direction_vector)  # type: ignore
        else:
            # Fallback to a default direction if something is wrong with indexing
            tangent_vector = LVector3f(1, 0, 0)  # type: ignore
    else:
        # Fallback if centerline data is not available
        tangent_vector = LVector3f(1, 0, 0)  # type: ignore

    # Create a transformation node to handle the orientation, parented to parent_node
    cone_np = parent_node.attachNewNode("cone_transform")

    # Position the cone at the robot tip, local to parent_node
    robot_tip_world_lvec = LVector3f(*app.robot_tip)  # type: ignore
    cone_pos_local = robot_tip_world_lvec - offset_vec
    cone_np.setPos(cone_pos_local)

    # Calculate focal point (world) as per your snippet's structure
    focal_point_world_np = app.robot_tip  # This is a numpy array

    # The point to lookAt, converted to LVector3f and made local to parent_node
    lookat_point_local = LVector3f(*focal_point_world_np) - offset_vec

    # lookAt uses the point (local to parent) and up-vector (local to parent)
    # tangent_vector is already an LVector3f (world direction)
    # For a parent that mainly scales/translates, world direction can often be used directly for 'up'
    cone_np.lookAt(lookat_point_local, tangent_vector)

    # Parent the cone model to the transformation node
    cone_model.reparentTo(cone_np)

    # Store for later (if we want to remove it next frame)
    app.light_cone_geom_np = cone_np


def draw_path(app, points, up_to_index):  # points are world coordinates
    parent_node, offset_vec = app.get_scene_graph_parent_and_offset()
    # Ensure the up_to_index is within bounds
    if up_to_index >= len(points):
        up_to_index = len(points) - 1

    # Clean everything before drawing again
    if hasattr(app, "path_line_node") and app.path_line_node:
        app.path_line_node.removeNode()
        app.path_line_node = None

    # Create the line
    line = LineSegs()  # type: ignore
    line.setThickness(5.0)  # Set a reasonable thickness
    line.setColor(0, 0.8, 0, 1)  # A nice green color

    if len(points) == 0 or up_to_index < 0:
        app.path_line_node = parent_node.attachNewNode(
            line.create()
        )  # Attach empty node if no points
        return

    # Start drawing the line from the first point (local to parent_node)
    first_point_world_np = points[0]
    first_point_world = LVector3f(first_point_world_np[0], first_point_world_np[1], first_point_world_np[2])  # type: ignore
    current_point_local = first_point_world - offset_vec
    line.moveTo(current_point_local)

    # Draw to the rest of the points up to the specified index
    for i in range(1, up_to_index + 1):
        next_point_world_np = points[i]
        next_point_world = LVector3f(next_point_world_np[0], next_point_world_np[1], next_point_world_np[2])  # type: ignore

        # Check for large jumps in the points (world space check before localization)
        # This check might be less relevant if points are already smoothed centerline
        # if (next_point_world - (current_point_local + offset_vec)).length() < 1.0: # Adjust this threshold as needed
        next_point_local = next_point_world - offset_vec
        line.drawTo(next_point_local)
        current_point_local = (
            next_point_local  # Update for next potential jump check (if re-enabled)
        )

    # Add the line to the scene
    line_node_geom = line.create()
    app.path_line_node = parent_node.attachNewNode(line_node_geom)


def draw_trajectory(app):
    parent_node, offset_vec = app.get_scene_graph_parent_and_offset()
    # Check if the trajectory line node already exists and remove it
    if hasattr(app, "trajectory_line_node") and app.trajectory_line_node:
        app.trajectory_line_node.removeNode()
        app.trajectory_line_node = None  # Clear the reference

    # Create the line
    line = LineSegs()  # type: ignore
    line.setThickness(5.0)
    line.setColor(8 / 255, 232 / 255, 222 / 255, 1)  # Same color as the arrow button

    points_to_draw_world = []
    if app.live_mode == False:
        # Smooth a lot the line
        temp_points = app.points  # These are world coordinates
        temp_points = interpolate_line(
            temp_points, num_points=1000
        )  # World coordinates

        # Start drawing the line from the robot tip
        robot_tip_world_np = app.robot_tip  # World coordinate
        points_to_draw_world.append(robot_tip_world_np)
        points_to_draw_world.extend(temp_points)

    elif (
        app.live_mode
        and hasattr(app, "trajectory_history_position")
        and len(app.trajectory_history_position) > 0
    ):
        # In live mode, draw the trajectory from history (these are world coordinates)
        points_to_draw_world = app.trajectory_history_position

    if not points_to_draw_world:
        app.trajectory_line_node = parent_node.attachNewNode(
            line.create()
        )  # Attach empty if no points
        return

    # Draw the line using points_to_draw_world, converting to local space
    first_point_world_np = points_to_draw_world[0]
    first_point_world = LVector3f(first_point_world_np[0], first_point_world_np[1], first_point_world_np[2])  # type: ignore
    current_point_local = first_point_world - offset_vec
    line.moveTo(current_point_local)

    for i in range(1, len(points_to_draw_world)):
        next_point_world_np = points_to_draw_world[i]
        next_point_world = LVector3f(next_point_world_np[0], next_point_world_np[1], next_point_world_np[2])  # type: ignore

        # Optional: jump check in world space
        # if (next_point_world - (current_point_local + offset_vec)).length() < 1.0: # Adjust threshold
        next_point_local = next_point_world - offset_vec
        line.drawTo(next_point_local)
        current_point_local = next_point_local

    # Create the line node and attach it to the scene
    line_node_geom = line.create()
    app.trajectory_line_node = parent_node.attachNewNode(line_node_geom)


def draw_trajectory_from_frames(
    app, frames_list, color_tuple, thickness=3.0, node_name="trajectory_node"
):
    """Helper function to draw a single trajectory from a list of 4x4 frames."""
    if not frames_list:
        print(f"[INFO] No frames to draw for {node_name}")
        return None

    positions = [frame[:3, 3] for frame in frames_list]

    if len(positions) < 2:
        print(
            f"[INFO] Not enough points to draw trajectory for {node_name} (needs at least 2, got {len(positions)})"
        )
        return None

    line_segs = LineSegs()  # type: ignore
    line_segs.setThickness(thickness)
    line_segs.setColor(
        color_tuple[0], color_tuple[1], color_tuple[2], color_tuple[3]
    )  # R, G, B, A

    line_segs.moveTo(LVector3f(*positions[0]))  # type: ignore
    for i in range(1, len(positions)):
        line_segs.drawTo(LVector3f(*positions[i]))  # type: ignore

    node = line_segs.create()
    path_node = app.render.attachNewNode(node)
    path_node.setName(node_name)
    return path_node


def draw_results_trajectories(app):
    """Draws the centerline, aligned GT, and aligned SLAM trajectories for results mode."""
    print("[INFO] Drawing results trajectories...")

    # Colors: (R, G, B, A)
    color_centerline = (0.8, 0.8, 0.8, 1)  # Light Gray/White for centerline
    color_gt = (0, 1, 0, 1)  # Green for Ground Truth
    color_slam = (0, 0, 1, 1)  # Blue for SLAM
    color_slam_snapped = (1, 0.5, 0, 1)  # Orange for Snapped SLAM

    # Store nodes to app if they need to be accessed/removed later
    if app.draw_centerline_bool == "1":
        app.results_centerline_node = draw_trajectory_from_frames(
            app,
            app.res_centerline_frames,
            color_centerline,
            thickness=4.0,
            node_name="results_centerline_traj",
        )
    if app.draw_gt_bool == "1":
        app.results_gt_node = draw_trajectory_from_frames(
            app,
            app.res_gt_aligned_frames,
            color_gt,
            thickness=4.0,
            node_name="results_gt_aligned_traj",
        )
    if app.draw_original_slam_bool == "1":
        app.results_slam_node = draw_trajectory_from_frames(
            app,
            app.res_slam_aligned_frames,
            color_slam,
            thickness=4.0,
            node_name="results_slam_aligned_traj",
        )

    if app.draw_snapped_slam_bool == "1":
        app.results_slam_snapped_node = draw_trajectory_from_frames(
            app,
            app.res_slam_snapped_frames,
            color_slam_snapped,
            thickness=4.0,
            node_name="results_slam_snapped_traj",
        )


def highlight_arrow(app, arrow):
    rgb_color = (8 / 255, 232 / 255, 222 / 255, 1)
    if arrow == "up":
        app.up_arrow["image_color"] = rgb_color
    elif arrow == "down":
        app.down_arrow["image_color"] = rgb_color


def unhighlight_arrow(app, arrow):
    if arrow == "up":
        app.up_arrow["image_color"] = (1, 1, 1, 1)  # Change back to normal color
    elif arrow == "down":
        app.down_arrow["image_color"] = (1, 1, 1, 1)
