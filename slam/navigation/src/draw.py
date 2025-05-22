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
    GeomVertexWriter,
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


def draw_circle(app, points):
    circle = LineSegs()  # type: ignore
    circle.setThickness(5.0)  # Increased thickness
    circle.setColor(1, 1, 0, 1)  # Changed color to yellow for better visibility

    if not points:
        return

    # Convert points to LVecBase3f and draw the circle
    for i, point_np in enumerate(points):
        point = LVector3f(point_np[0], point_np[1], point_np[2])  # type: ignore
        if i == 0:
            circle.moveTo(point)
        else:
            circle.drawTo(point)
    # Connect back to the first point
    first_point_np = points[0]
    circle.drawTo(LVector3f(first_point_np[0], first_point_np[1], first_point_np[2]))  # type: ignore

    # Add the circle to the scene
    circle_node = circle.create()
    app.render.attachNewNode(circle_node)


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


def draw_vector(app, point, direction, color):
    """Draw a vector from a point in a specified direction with a given color."""
    line = LineSegs()  # type: ignore
    line.setThickness(5.0)  # Increased thickness
    line.setColor(*color)  # Set the color

    start = LVector3f(point[0], point[1], point[2])  # type: ignore
    end = LVector3f(point[0] + direction[0], point[1] + direction[1], point[2] + direction[2])  # type: ignore

    line.moveTo(start)
    line.drawTo(end)

    # Add the line to the scene
    line_node = line.create()
    app.render.attachNewNode(line_node)


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
    if app.results_mode:
        return
    if hasattr(app, "robot_tip_node") and app.robot_tip_node:
        app.robot_tip_node.removeNode()  # Remove the old node if it exists
        app.robot_tip_node = None  # Clear reference

    try:
        robot_tip_visual = app.loader.loadModel("models/smiley")
    except Exception as e:
        print(f"[ERROR] Could not load robot tip model: models/smiley. {e}")
        # Fallback to a simple sphere or just a NodePath if model fails
        vdata = GeomVertexData("fallback_tip_geom", GeomVertexFormat.getV3n3cpt2(), Geom.UHStatic)  # type: ignore
        # Placeholder for actual geometry if needed, or just use an empty GeomNode
        geom = Geom(vdata)  # type: ignore
        tris = GeomTriangles(Geom.UHStatic)  # type: ignore
        # geom.addPrimitive(tris) # Add primitives if you define vertices
        node = GeomNode("fallback_tip_node")  # type: ignore
        node.addGeom(geom)
        robot_tip_visual = NodePath(node)

    robot_tip_visual.setScale(1)  # Scale to appropriate size
    robot_tip_visual.setColor(0, 1, 0, 1)  # Set color to green
    if hasattr(app, "robot_tip") and app.robot_tip is not None:
        robot_tip_visual.setPos(LVector3f(*app.robot_tip))  # type: ignore
    else:
        robot_tip_visual.setPos(LVector3f(0, 0, 0))  # type: ignore Default if tip not set

    # Create a new node and parent the visual to it
    app.robot_tip_node = app.render.attachNewNode("RobotTipNode")
    robot_tip_visual.reparentTo(app.robot_tip_node)

    # Draw the light cone geometry
    draw_light_cone_geom(app)


def draw_light_cone_geom(app):
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
    if hasattr(app, "interpolated_points") and len(app.interpolated_points) > 0:
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

    # Create a transformation node to handle the orientation
    cone_np = app.render.attachNewNode("cone_transform")

    # Position the cone at the robot tip
    cone_np.setPos(LVector3f(*app.robot_tip))  # type: ignore

    # Calculate focal point
    focal_point = app.robot_tip

    # cone_np.lookAt(LVector3f(*focal_point), -normal_vector if normal_vector else LVector3f(0, 0, 1))  # type: ignore
    cone_np.lookAt(LVector3f(*focal_point), tangent_vector)  # type: ignore

    # Parent the cone model to the transformation node
    cone_model.reparentTo(cone_np)

    # Store for later (if we want to remove it next frame)
    app.light_cone_geom_np = cone_np


def draw_path(app, points, up_to_index):
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
    line.setColor(1, 0, 0, 1)  # Red color

    # Start drawing the line from the first point
    first_point = LVector3f(points[0][0], points[0][1], points[0][2])  # type: ignore
    line.moveTo(first_point)

    # Draw to the rest of the points up to the specified index
    for i in range(1, up_to_index + 1):
        next_point = LVector3f(points[i][0], points[i][1], points[i][2])  # type: ignore
        # Check for large jumps in the points and skip if necessary
        if (next_point - first_point).length() < 1.0:  # Adjust this threshold as needed
            line.drawTo(next_point)
            first_point = next_point

    # Add the line to the scene
    line_node = line.create()
    app.path_line_node = app.render.attachNewNode(line_node)


def draw_trajectory(app):
    # Check if the trajectory line node already exists and remove it
    if hasattr(app, "trajectory_line_node") and app.trajectory_line_node:
        app.trajectory_line_node.removeNode()
        app.trajectory_line_node = None  # Clear the reference

    # Create the line
    line = LineSegs()  # type: ignore
    line.setThickness(5.0)
    line.setColor(8 / 255, 232 / 255, 222 / 255, 1)  # Same color as the arrow button

    if app.live_mode == False:
        # Smooth a lot the line
        points = app.points
        points = interpolate_line(points, num_points=1000)

        # Start drawing the line from the robot tip
        robot_tip = app.robot_tip
        first_point = LVector3f(robot_tip[0], robot_tip[1], robot_tip[2])  # type: ignore
        line.moveTo(first_point)

        # Draw to the rest of the points
        for i in range(1, len(points)):
            next_point = LVector3f(points[i][0], points[i][1], points[i][2])  # type: ignore
            if (next_point - first_point).length() < 1.0:
                line.drawTo(next_point)
                first_point = next_point

    elif (
        app.live_mode
        and hasattr(app, "trajectory_history_position")
        and len(app.trajectory_history_position) > 0
    ):
        # In live mode, draw the trajectory from history
        first_point = LVector3f(*app.trajectory_history_position[0])  # type: ignore
        line.moveTo(first_point)

        for i in range(1, len(app.trajectory_history_position)):
            next_point = LVector3f(*app.trajectory_history_position[i])  # type: ignore
            line.drawTo(next_point)

    # Create the line node and attach it to the scene
    line_node = line.create()
    app.trajectory_line_node = app.render.attachNewNode(line_node)


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
