import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from direct.showbase.ShowBase import ShowBase  # type: ignore
from panda3d.core import *  # type: ignore
from direct.task import Task  # type: ignore
from direct.gui.DirectGui import DirectLabel  # type: ignore
import pyvista as pv  # type: ignore
import numpy as np  # type: ignore
import socket
import math
import configparser
from set_FS_frame import (
    interpolate_line,
    compute_tangent_vectors,
    compute_MRF,
    smooth_vectors,
)

# TODO: Check all the measurements units and make sure they are consistent

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Path Navigation Tool")
parser.add_argument(
    "-view",
    type=str,
    default="fp",
    choices=["fp", "tp"],
    help="Set view mode: fp (first person) or tp (third person)",
)
parser.add_argument(
    "-live",
    type=bool,
    default=False,
    help="Set live mode: True (live) or False (offline)",
)
parser.add_argument(
    "-record",
    type=bool,
    default=False,
    help="Set record mode: True (record) or False (no record)",
)

args = parser.parse_args()


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Read the configuration file
        self.app_config = configparser.ConfigParser()
        self.app_config.read("config.ini")
        self.data_folder = self.app_config["PATHS"]["data_folder"]
        self.path_name = self.app_config["PATHS"]["path_name"]
        self.model_name = self.app_config["PATHS"]["model_name"]
        self.negative_model_name = self.app_config["PATHS"]["negative_model_name"]

        self.draw_circles_bool = self.app_config["DRAW"]["draw_circles_bool"]
        self.draw_centerline_bool = self.app_config["DRAW"]["draw_centerline_bool"]
        self.draw_frames_bool = self.app_config["DRAW"]["draw_frames_bool"]
        self.draw_reference_frames_bool = self.app_config["DRAW"][
            "draw_reference_frames_bool"
        ]

        self.sim_server_bool = self.app_config["SLAM"]["sim_server_bool"]

        # Define path of the .vtp file
        self.path_path = self.data_folder + self.path_name

        # Read parameters from command line
        self.view_mode = args.view
        self.live_mode = args.live
        self.record_mode = args.record
        print("\nCommand line arguments:")
        print(f"-View mode: {self.view_mode}")
        print(f"-Live mode: {self.live_mode}")
        print(f"-Record mode: {self.record_mode}\n")

        # Quit app on "q"
        self.accept("q", self.quit_app)

        # Set up camera parameters
        self.setup_camera_params()

        # Set background color
        self.setBackgroundColor(0, 0.168627, 0.211765, 1.0)

        # Load arrow key icons with transparency
        self.up_arrow = DirectLabel(
            image="data/icons/up_white.png",
            pos=(1.7, 0, -0.70),
            scale=0.05,
            relief=None,
        )
        self.down_arrow = DirectLabel(
            image="data/icons/down_white.png",
            pos=(1.7, 0, -0.85),
            scale=0.05,
            relief=None,
        )

        # Enable transparency for these icons
        self.up_arrow.setTransparency(TransparencyAttrib.MDual)  # type: ignore
        self.down_arrow.setTransparency(TransparencyAttrib.MDual)  # type: ignore

        # Set antialiasing
        self.render.setAntialias(AntialiasAttrib.MAuto)  # type: ignore

        # Initialize timer for blinking
        self.blink_timer = 0
        self.blink_interval = 1  # in seconds
        self.robot_tip_visible = True  # Initial visibility status

        # Load basic environment
        # self.scene = self.loader.loadModel("models/environment")
        # self.scene.reparentTo(self.render)

        # Set up key controls
        if self.live_mode == False:
            self.setup_key_controls()

        # Task for updating the scene
        self.taskMgr.add(self.update_scene, "updateScene")

        # Get centerline points from the .vtp
        points = self.get_vtp_line_points()

        # print number of points
        print("[INFO] Number of points in the centerline: ", len(points))

        # Temporarily draw the horizontal line
        # points = [(0, 0, 3), (1, 0, 3), (2, 0, 3), (3, 0, 3), (4, 0, 3), (5, 0, 3), (6, 0, 3), (7, 0, 3), (8, 0, 3), (9, 0, 3), (10, 0, 3)]

        # Setup
        self.setup_line(points)
        self.points = points

        # Init transformation matrices
        self.c_T_w = np.eye(4)
        self.w_T_o = self.setup_w_T_o()

        # Load the model
        if self.view_mode == "fp":

            # Load the negative model to visualize the internal part
            """
            To create the negative solid using FreeCAD:
            - open the part menu
            - import the .obj file of the model
            - create shape from mesh for the imported model
            - make solid from the created shape
            - create a sphere that can contain the model
            - use boolean difference between the sphere and the solid
            - export the boolean cut as .obj
            """

            self.model = self.data_folder + self.negative_model_name

            print("[INFO] Initializing First Person View Mode...")
            # Load the phantom model
            self.scene = self.loader.loadModel(self.model)
            self.scene.reparentTo(self.render)
            self.scene.setTransparency(TransparencyAttrib.MDual)  # type: ignore
            self.scene.setColorScale(1, 1, 1, 1)  # Set transparency level
            self.scene.setTwoSided(True)

            # Adjust material properties
            myMaterial = Material()  # type: ignore
            myMaterial.setShininess(80)  # Higher shininess for more specular highlight
            myMaterial.setSpecular((0.9, 0.9, 0.9, 1))  # Brighter specular highlights
            myMaterial.setAmbient((0.3, 0.3, 0.3, 1))  # Slightly brighter ambient color
            myMaterial.setDiffuse(
                (0.7, 0.2, 0.2, 1)
            )  # Reddish diffuse color, adjust as needed
            self.scene.setMaterial(myMaterial, 1)

            # Add directional light (consider also using ambient light)
            directionalLight = DirectionalLight("directionalLight")  # type: ignore
            directionalLight.setColor((1, 0.9, 0.8, 1))  # Warm light color
            directionalLightNP = self.render.attachNewNode(directionalLight)
            directionalLightNP.setHpr(
                45, -45, 0
            )  # Adjust the light direction as needed
            self.render.setLight(directionalLightNP)

            # Add ambient light
            ambientLight = AmbientLight("ambientLight")  # type: ignore
            ambientLight.setColor((0.2, 0.2, 0.2, 1))
            ambientLightNP = self.render.attachNewNode(ambientLight)
            self.render.setLight(ambientLightNP)

            # Store the directional light node as an instance variable for later updates
            self.directionalLightNP = directionalLightNP

            # Adjust clipping planes
            self.camLens.setNearFar(0.1, 100)

        elif self.view_mode == "tp":
            print("[INFO] Initializinig Third Person View Mode...")

            # Load the standard model to visualize the external part
            self.model = self.data_folder + self.model_name

            # Load the phantom model
            self.scene = self.loader.loadModel(self.model)
            self.scene.reparentTo(self.render)

            # Set transparency level (0.5 for 50% transparency) to see the robot moving inside
            self.scene.setTransparency(TransparencyAttrib.MDual)  # type: ignore
            self.scene.setColorScale(1, 1, 1, 0.5)

            # Initially draw the path up to the first point
            self.draw_path(self.interpolated_points, 0)

        print("[INFO] Initialization done\n")

        if self.live_mode == True:

            # Start the server in a thread
            self.listen_thread = threading.Thread(target=self.start_server, daemon=True)
            self.listen_thread.start()

            if self.sim_server_bool == "1":
                # Start the simulation server
                self.sim_server_thread = threading.Thread(
                    target=self.sim_server, daemon=True
                )
                self.sim_server_thread.start()

        if self.record_mode == True:
            self.setup_video_recorder()

        # Draw elements
        self.draw_elements(points)

    ## SETUP METHODS
    def start_server(self, host="127.0.0.1", port=12345):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            print(f"Server listening on {host}:{port}")

            while True:
                conn, addr = s.accept()
                with conn:
                    # print(f"Connected by {addr}")
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        # Parse the received data and update the transformation matrix
                        self.c_T_w = data.decode()
                    print("Connection closed")

    def sim_server(self, host="127.0.0.1", port=12345):
        time.sleep(1)  # Give the server time to start
        try:
            # Create a socket and connect to the server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))

            while True:
                # Create a mock transformation matrix
                mock_c_T_o = np.eye(4)
                mock_c_T_w = np.eye(4)

                # Take the last frame of the centerline as the translation
                mock_c_T_o[:3, 3] = self.interpolated_points[-1]
                mock_c_T_o[:3, 0] = self.tangents[-1]
                mock_c_T_o[:3, 1] = self.normals[-1]
                mock_c_T_o[:3, 2] = self.binormals[-1]

                # Get the transformation matrix from the origin to the first point
                w_T_o_matrix = self.w_T_o
                o_T_w_matrix = np.linalg.inv(w_T_o_matrix)

                # Compute the transformation matrix from the camera to the world
                mock_c_T_w = np.dot(mock_c_T_o, o_T_w_matrix)

                # Add some random noise to the translation
                mock_c_T_w[:3, 3] += np.random.normal(0, 0.05, 3)

                # Convert to string and send
                data = str(mock_c_T_w.tolist())
                s.sendall(data.encode())
                time.sleep(0.1)  # Add small delay between sends

        except ConnectionRefusedError:
            print("Could not connect to server. Is it running?")
        except Exception as e:
            print(f"Error in sim_server: {e}")
        finally:
            s.close()

    def draw_elements(self, points):
        if self.draw_frames_bool == "1":
            # Draw some frames
            self.draw_FS_frames(
                points,
                draw_tangent=True,
                draw_normal=True,
                draw_binormal=True,
            )

        if self.draw_circles_bool == "1":
            # Drawing circles
            self.draw_circles_around_points(radius=0.2, num_segments=50)

        if self.view_mode == "tp" and self.draw_reference_frames_bool == "1":
            # Draw the origin frame
            self.draw_origin_frame()

            # Draw the base frame
            self.draw_base_frame()

        # Draw the robot tip
        self.draw_robot_tip()

    def save_calibration_file(
        self, width, height, fx, fy, cx, cy, filename="calibration_sim.yaml"
    ):
        """
        Writes a YAML file in the same format as shown,
        but replaces the camera parameters with the ones in code.
        """
        # If you want *distortion* to remain zero, just keep them as 0.0
        # If you want to provide your own, fill them in accordingly.
        k1 = 0.0
        k2 = 0.0
        k3 = 0.0
        p1 = 0.0
        p2 = 0.0

        # Construct the file content as a multi-line string
        yaml_content = f"""%YAML:1.0
Camera.RGB: 1
Camera.ThDepth: 40.0
Camera.bf: 40.0
Camera.fps: 15
Camera.height: {height}
Camera.type: PinHole
Camera.width: {width}
Camera1.cx: {cx}
Camera1.cy: {cy}
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.k3: {k3}
Camera1.p1: {p1}
Camera1.p2: {p2}
File.version: '1.0'
ORBextractor.iniThFAST: 1
ORBextractor.minThFAST: 1
ORBextractor.nFeatures: 2500
ORBextractor.nLevels: 10
ORBextractor.scaleFactor: 1.2
Viewer.CameraLineWidth: 3.0
Viewer.CameraSize: 0.08
Viewer.GraphLineWidth: 0.9
Viewer.KeyFrameLineWidth: 1.0
Viewer.KeyFrameSize: 0.05
Viewer.PointSize: 2.0
Viewer.ViewpointF: 500.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
"""

        # Write to file
        with open(os.path.join(self.data_folder, filename), "w") as f:
            f.write(yaml_content)

        print(f"[INFO] Saved camera calibration to {filename}")

    def setup_camera_params(self):
        # 1) Camera parameters
        # Read camera parameters from config.ini
        width = int(self.app_config["CAMERA"]["width"])
        height = int(self.app_config["CAMERA"]["height"])
        fx = float(self.app_config["CAMERA"]["fx"])
        fy = float(self.app_config["CAMERA"]["fy"])
        cx = float(self.app_config["CAMERA"]["cx"])
        cy = float(self.app_config["CAMERA"]["cy"])

        # 2) Force window size and basic config
        loadPrcFileData("", f"win-size {width} {height}")  # type: ignore
        loadPrcFileData("", "window-title Panda3D Full Camera Control")  # type: ignore
        loadPrcFileData("", "load-file-type p3assimp")  # type: ignore

        # 3) Grab PerspectiveLens
        self.camLens = self.cam.node().getLens()

        # 4) Compute FoV from fx, fy
        fov_x = 2.0 * math.degrees(math.atan(0.5 * width / fx))
        fov_y = 2.0 * math.degrees(math.atan(0.5 * height / fy))
        self.camLens.setFov(fov_x, fov_y)

        # 5) Set film size + offset for the principal point
        self.camLens.setFilmSize(width, height)
        offset_x = cx - (width / 2.0)
        offset_y = (height / 2.0) - cy  # note: Panda has +y pointing "up"
        self.camLens.setFilmOffset(offset_x, offset_y)

        # 6) Near/far planes
        self.camLens.setNearFar(0.1, 100.0)

        # 7) Save out the calibration_sim.yaml.
        self.save_calibration_file(width, height, fx, fy, cx, cy)

    def setup_key_controls(self):
        self.keyMap = {"robot_tip_forward": False, "robot_tip_backward": False}

        # Bind arrow keys for moving the robot tip
        self.accept("arrow_up", self.update_key_map, ["robot_tip_forward", True])
        self.accept("arrow_up-up", self.update_key_map, ["robot_tip_forward", False])
        self.accept("arrow_down", self.update_key_map, ["robot_tip_backward", True])
        self.accept("arrow_down-up", self.update_key_map, ["robot_tip_backward", False])

    def setup_line(self, points):
        # Load the .vtp file and interpolate the line
        self.interpolated_points = interpolate_line(points, num_points=1000)
        self.tangents = compute_tangent_vectors(self.interpolated_points)
        self.tangents = smooth_vectors(self.tangents, 10, 10)

        # Compute the Frenet-Serret frame using the MRF algorithm
        self.normals, self.binormals = compute_MRF(self.tangents)

        # Set first and end point
        self.start_point = self.interpolated_points[0]
        self.end_point = self.interpolated_points[-1]

        # Initialize the robot tip
        self.robot_tip = self.interpolated_points[
            0
        ]  # Setting the first point as the start
        self.robot_tip_node = None

        # Compute line length
        self.line_length = self.curvilinear_abscissa(self.end_point)
        print("[INFO] Centerline length: ", self.line_length, "mm")

    def setup_w_T_o(self):
        """Load the transformation matrix from centerline_frames.txt"""
        w_T_o = np.eye(4)

        # Set the transformation matrix from the origin to the first point using self.tangents, self.normals, and self.binormals
        w_T_o[:3, 0] = self.tangents[0]
        w_T_o[:3, 1] = self.normals[0]
        w_T_o[:3, 2] = self.binormals[0]
        w_T_o[:3, 3] = self.interpolated_points[0]

        # print(f"Transformation matrix w_T_o: \n{w_T_o}")

        return w_T_o

    ## LINE UTILS
    def curvilinear_abscissa(self, point):
        # Find the closest point to the given point
        distances = np.linalg.norm(self.interpolated_points - point, axis=1)
        closest_index = np.argmin(distances)

        # Calculate cumulative distance from start to the closest point
        total_distance = 0
        for i in range(closest_index):
            segment = self.interpolated_points[i + 1] - self.interpolated_points[i]
            total_distance += np.linalg.norm(segment)

        return total_distance

    def get_vtp_line_points(self):
        # Load the .vtp file
        line_model = pv.read(self.path_path)

        # Convert the points to a list of tuples
        points = [tuple(point) for point in line_model.points]

        # Return the points so as to be able to use them in create_line
        return points

    ## UPDATE METHODS
    def update_camera_to_robot_tip(self):
        # Find the index of the closest point to the robot tip
        distances = np.linalg.norm(self.interpolated_points - self.robot_tip, axis=1)
        closest_index = np.argmin(distances)

        # Get the corresponding tangent, normal, and binormal vectors
        tangent = LVector3f(*self.tangents[closest_index])  # type: ignore
        normal = LVector3f(*self.normals[closest_index])  # type: ignore
        binormal = LVector3f(*self.binormals[closest_index])  # type: ignore

        # Set the camera position at the robot tip
        self.camera.setPos(LVector3f(*self.robot_tip))  # type: ignore

        # Calculate the focal point using the tangent vector
        focal_point = self.robot_tip + tangent

        # Set the camera to look at the focal point with the binormal as the up vector
        self.camera.lookAt(LVector3f(*focal_point), -normal)  # type: ignore

        # Update the directional light's orientation to match the camera's orientation
        # if in first-person view mode
        if self.view_mode == "fp":
            cameraHpr = self.camera.getHpr()
            self.directionalLightNP.setHpr(cameraHpr)

        # Adjust lighting to follow the camera
        self.directionalLightNP.setPos(
            self.camera.getX(), self.camera.getY(), self.camera.getZ()
        )

    def update_robot_tip_position(self, dt, forward=True):
        if self.live_mode == False:
            # Define the speed of movement along the line
            movement_speed = 5  # Adjust as needed

            # Calculate distances from self.robot_tip to each point in self.interpolated_points
            distances = np.linalg.norm(
                self.interpolated_points - self.robot_tip, axis=1
            )
            current_index = np.argmin(distances)

            if forward:
                # Check if the robot tip is at the last point
                if current_index >= len(self.interpolated_points) - 1:
                    return  # Stop moving forward
                next_index = current_index + 1
            else:
                # Check if the robot tip is at the first point
                if current_index == 0:
                    return  # Stop moving backward
                next_index = current_index - 1

            # Calculate the direction and distance to the next point
            direction = (
                self.interpolated_points[next_index]
                - self.interpolated_points[current_index]
            )
            distance_to_next_point = np.linalg.norm(direction)
            direction = (
                direction / distance_to_next_point
            )  # Normalize the direction vector

            # Calculate the movement step
            step_size = movement_speed * dt
            if step_size > distance_to_next_point:
                step_size = (
                    distance_to_next_point  # Limit step to not overshoot the next point
                )

            # Update the position
            new_position = self.robot_tip + direction * step_size
            self.robot_tip = new_position

            # Find the index of the closest point to the robot tip
            distances = np.linalg.norm(
                self.interpolated_points - self.robot_tip, axis=1
            )
            closest_index = np.argmin(distances)

            if self.view_mode == "tp":
                # Redraw the path up to the robot tip
                self.draw_path(self.interpolated_points, closest_index)

            # Get the current point on the centerline using the closest_index
            current_point = self.interpolated_points[closest_index]

            # Compute the curvilinear abscissa
            self.current_ca = self.curvilinear_abscissa(current_point)

        else:
            # Update the robot tip position based on the transformation matrix
            try:
                # Convert string to numpy array and multiply matrices
                c_T_w_matrix = np.array(eval(self.c_T_w))
                w_T_o_matrix = self.w_T_o

                w_T_c_matrix = np.linalg.inv(c_T_w_matrix)
                o_T_w_matrix = np.linalg.inv(w_T_o_matrix)

                c_T_o_matrix = np.dot(c_T_w_matrix, w_T_o_matrix)
                o_T_c_matrix = np.dot(o_T_w_matrix, w_T_c_matrix)

                translation = c_T_o_matrix[:3, 3]

                # Update the robot tip position
                self.robot_tip = translation

            except (SyntaxError, AttributeError) as e:
                # Skip update if data is not valid
                pass

        # Update the visual representation
        self.draw_robot_tip()

    def update_key_map(self, controlName, controlState):
        self.keyMap[controlName] = controlState

        if controlName == "robot_tip_forward":
            if controlState:
                self.highlight_arrow("up")
            else:
                self.unhighlight_arrow("up")
        elif controlName == "robot_tip_backward":
            if controlState:
                self.highlight_arrow("down")
            else:
                self.unhighlight_arrow("down")

    def update_scene(self, task):
        dt = globalClock.getDt()  # type: ignore

        # Update the robot tip position
        if self.live_mode == False:
            if self.keyMap["robot_tip_forward"]:
                self.update_robot_tip_position(dt, forward=True)
            if self.keyMap["robot_tip_backward"]:
                self.update_robot_tip_position(dt, forward=False)
        else:
            self.update_robot_tip_position(dt)

        # Update the camera position and orientation
        if self.view_mode == "fp":
            self.update_camera_to_robot_tip()

            if self.draw_centerline_bool == "1":
                # Update the trajectory
                self.update_trajectory()

        # Blinking logic
        self.blink_timer += dt
        if self.blink_timer >= self.blink_interval:
            self.blink_timer = 0  # Reset timer
            self.robot_tip_visible = not self.robot_tip_visible  # Toggle visibility
            if self.robot_tip_node:
                self.robot_tip_node.setTransparency(
                    TransparencyAttrib.MDual  # type: ignore
                )  # Enable transparency
                self.robot_tip_node.setAlphaScale(
                    1 if self.robot_tip_visible else 0.5
                )  # Set visibility

        # If recording mode is enabled, capture the frame
        if self.record_mode:
            self.record_frame()

        # Start the terminal update thread if it hasn't been started yet
        if not hasattr(self, "terminal_thread"):
            self.terminal_thread = threading.Thread(
                target=self.update_terminal, daemon=True
            )
            self.terminal_thread.start()

        return Task.cont

    def update_terminal(self):

        # Delay the start of the thread to allow the main thread to start
        time.sleep(2)
        while True:
            if self.live_mode == True:
                if hasattr(self, "c_T_w"):
                    print(f"\rReceived: {self.c_T_w}\033[F", end="", flush=True)
            else:
                if hasattr(self, "current_ca"):
                    print(
                        f"\rCurrent curvilinear abscissa: {self.current_ca:.2f} mm",
                        end="",
                        flush=True,
                    )
            # Add small delay to prevent high CPU usage
            time.sleep(0.25)

    def update_trajectory(self):
        # Draw the trajectory from the current robot tip position
        self.draw_trajectory()

    ## DRAW METHODS
    def draw_circles_around_points(self, radius=1, num_segments=10):
        for i, center in enumerate(self.interpolated_points):
            normal = self.normals[i]
            binormal = self.binormals[i]

            # Debug: Print normal and binormal
            # print(f"Point {i}: Normal = {normal}, Binormal = {binormal}")

            # Generate circle points
            circle_points = []
            for j in range(num_segments):
                angle = 2 * np.pi * j / num_segments
                dx = np.cos(angle) * normal
                dy = np.sin(angle) * binormal
                point = center + radius * (dx + dy)
                circle_points.append(point)

            # Debug: Print first few points of each circle
            # print(f"Circle {i} points: {circle_points[:3]}")

            # Draw the circle
            self.draw_circle(circle_points)

    def draw_circle(self, points):
        circle = LineSegs()  # type: ignore
        circle.setThickness(5.0)  # Increased thickness
        circle.setColor(1, 1, 0, 1)  # Changed color to yellow for better visibility

        # Convert points to LVecBase3f and draw the circle
        for i, point in enumerate(points):
            panda_point = LVector3f(point[0], point[1], point[2])  # type: ignore
            if i == 0:
                circle.moveTo(panda_point)
            else:
                circle.drawTo(panda_point)
        # Connect back to the first point
        circle.drawTo(LVector3f(points[0][0], points[0][1], points[0][2]))  # type: ignore

        # Add the circle to the scene
        circle_node = circle.create()
        self.render.attachNewNode(circle_node)

    def draw_FS_frames(
        self,
        points,
        draw_tangent=True,
        draw_normal=True,
        draw_binormal=True,
    ):
        # Draw the frames
        for i, point in enumerate(self.interpolated_points):
            if draw_tangent:
                self.draw_vector(
                    point, self.tangents[i], (1, 0, 0, 1)
                )  # Red for tangent
            if draw_normal:
                self.draw_vector(
                    point, self.normals[i], (0, 1, 0, 1)
                )  # Green for normal
            if draw_binormal:
                self.draw_vector(
                    point, self.binormals[i], (0, 0, 1, 1)
                )  # Blue for binormal

    def draw_light_cone_geom(self):
        """
        Draws a translucent cone geometry that starts at the robot tip
        and extends in the tangent direction.
        """
        # Remove old cone if it exists
        if hasattr(self, "light_cone_geom_np") and self.light_cone_geom_np:
            self.light_cone_geom_np.removeNode()

        # Load cone model
        cone_model = self.loader.loadModel(
            "/home/emanuele/Desktop/github/navigation/data/icons/cone.obj"
        )
        cone_model.reparentTo(self.robot_tip_node)

        # Set size and color/transparency
        cone_model.setScale(0.1)
        cone_model.setColor(1, 1, 0, 0.3)  # Slightly yellow, partial alpha
        cone_model.setTransparency(TransparencyAttrib.MAlpha)  # type: ignore

        # Find the index of the closest point to the robot tip
        distances = np.linalg.norm(self.interpolated_points - self.robot_tip, axis=1)
        closest_index = np.argmin(distances)

        # Get the corresponding tangent, normal, and binormal vectors
        tangent_vector = LVector3f(*self.tangents[closest_index])  # type: ignore
        normal_vector = LVector3f(*self.normals[closest_index])  # type: ignore

        # Orient the cone so that +Z aligns with the tangent and the origin of the cone is at the robot tip.
        # Create a transformation node to handle the orientation
        cone_np = self.render.attachNewNode("cone_transform")

        # Position the cone at the robot tip
        cone_np.setPos(LVector3f(*self.robot_tip))  # type: ignore

        # Calculate focal point
        focal_point = self.robot_tip + tangent_vector

        # Set the HPR of the cone to align with the tangent vector
        cone_np.lookAt(LVector3f(*focal_point), normal_vector)  # type: ignore

        # Parent the cone model to the transformation node
        cone_model.reparentTo(cone_np)

        # Store for later (if we want to remove it next frame)
        self.light_cone_geom_np = cone_np

    def draw_origin_frame(self):
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
        frame_node = self.render.attachNewNode("OriginFrame")

        # Create the frame geometry and attach it directly
        frame_geom = frame.create()
        frame_node.attachNewNode(frame_geom)

        # Set the scale of the frame
        frame_node.setScale(1)  # Scale to a reasonable size

        # Return the node for later reference
        return frame_node

    def draw_base_frame(self):

        w_T_o = self.w_T_o

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
        frame_node = self.render.attachNewNode("BaseFrame")

        # Create the frame geometry and attach it
        frame_geom = frame.create()
        frame_node.attachNewNode(frame_geom)

        # Set scale of the frame
        frame_node.setScale(1)

        # Return the node for later reference
        return frame_node

    def draw_robot_tip(self):
        if self.robot_tip_node:
            self.robot_tip_node.removeNode()  # Remove the old node if it exists

        # Create the robot tip visual
        robot_tip_visual = self.loader.loadModel(
            "models/smiley"
        )  # Ensure this is a valid model path

        robot_tip_visual.setScale(0.1)  # Scale to appropriate size
        robot_tip_visual.setColor(0, 1, 0, 1)  # Set color to green
        robot_tip_visual.setPos(LVector3f(*self.robot_tip))  # type: ignore

        # Create a new node and parent the visual to it
        robot_tip_node = self.render.attachNewNode("RobotTipNode")
        robot_tip_visual.reparentTo(robot_tip_node)
        self.robot_tip_node = robot_tip_node

        # Draw the light cone geometry (TODO: fix this)
        # self.draw_light_cone_geom()

    def draw_path(self, points, up_to_index):
        # Ensure the up_to_index is within bounds
        if up_to_index >= len(points):
            up_to_index = len(points) - 1

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
            if (
                next_point - first_point
            ).length() < 1.0:  # Adjust this threshold as needed
                line.drawTo(next_point)
                first_point = next_point

        # Add the line to the scene
        line_node = line.create()
        self.render.attachNewNode(line_node)

    def draw_trajectory(self):
        # Check if the trajectory line node already exists and remove it
        if hasattr(self, "trajectory_line_node") and self.trajectory_line_node:
            self.trajectory_line_node.removeNode()

        # Smooth a lot the line
        points = self.points
        points = interpolate_line(points, num_points=1000)

        # Create the line
        line = LineSegs()  # type: ignore
        line.setThickness(5.0)
        line.setColor(
            8 / 255, 232 / 255, 222 / 255, 1
        )  # Same color as the arrow button

        # Start drawing the line from the robot tip
        robot_tip = self.robot_tip
        first_point = LVector3f(robot_tip[0], robot_tip[1], robot_tip[2])  # type: ignore
        line.moveTo(first_point)

        # Draw to the rest of the points
        for i in range(1, len(points)):
            next_point = LVector3f(points[i][0], points[i][1], points[i][2])  # type: ignore
            if (next_point - first_point).length() < 1.0:
                line.drawTo(next_point)
                first_point = next_point

        # Create the line node and attach it to the scene
        line_node = line.create()
        self.trajectory_line_node = self.render.attachNewNode(line_node)

    def draw_vector(self, start_point, direction, color):
        # Convert NumPy array to LVecBase3f
        start_point = LVector3f(start_point[0], start_point[1], start_point[2])  # type: ignore
        end_point = (
            start_point + LVector3f(direction[0], direction[1], direction[2]) * 0.2  # type: ignore
        )

        line = LineSegs()  # type: ignore
        line.setThickness(2.0)
        line.setColor(color)
        line.moveTo(start_point)
        line.drawTo(end_point)
        line_node = line.create()
        self.render.attachNewNode(line_node)

    def highlight_arrow(self, arrow):
        rgb_color = (8 / 255, 232 / 255, 222 / 255, 1)
        if arrow == "up":
            self.up_arrow["image_color"] = rgb_color
        elif arrow == "down":
            self.down_arrow["image_color"] = rgb_color

    def unhighlight_arrow(self, arrow):
        if arrow == "up":
            self.up_arrow["image_color"] = (1, 1, 1, 1)  # Change back to normal color
        elif arrow == "down":
            self.down_arrow["image_color"] = (1, 1, 1, 1)

    # RECORD METHODS
    def setup_video_recorder(self):
        """
        If recording is True, create a directory for frames,
        and initialize the counter to 0.
        """
        self.record_frame_idx = 0
        if self.record_mode:
            # Create (or reuse) a directory to store frames
            self.record_dir = "recorded_frames"
            os.makedirs(self.record_dir, exist_ok=True)
            print(f"[INFO] Recording frames to ./{self.record_dir}/")
        else:
            self.record_dir = None

    def record_frame(self):
        """
        Capture the current window as an image and save it to a numbered PNG.
        """
        if not self.record_dir:
            return  # Not recording

        # Grab a screenshot from the main window.
        # getScreenshot() returns a PNMImage, which we can write to file.
        screenshot = self.win.getScreenshot()

        # Create a filename like: frame_00000_ca_XXX
        if hasattr(self, "current_ca"):
            filename = os.path.join(
                self.record_dir,
                f"frame_{self.record_frame_idx:05d}_ca_{self.current_ca:.2f}_mm.png",
            )
        else:
            filename = os.path.join(
                self.record_dir,
                f"frame_{self.record_frame_idx:05d}_ca_0.0_mm.png",
            )
        screenshot.write(filename)

        self.record_frame_idx += 1

    def quit_app(self):
        """
        Called when the user presses 'q'.
        Stops the main loop, optionally runs ffmpeg to encode video, and exits.
        """
        print("[INFO] Quitting the app now.")

        # Stop the Panda3D main loop
        self.taskMgr.stop()

        # If we recorded frames, let's encode them
        if self.record_mode and hasattr(self, "record_dir"):
            # Extract centerline name from path
            centerline_name = os.path.splitext(os.path.basename(self.path_name))[0]

            # Set the video directory
            video_dir = os.path.join(self.data_folder, "videos")
            os.makedirs(video_dir, exist_ok=True)
            timestamp = time.time()
            video = os.path.join(video_dir, f"output_{centerline_name}_{timestamp}.mp4")

            # Extract timestamps and CA values from filenames
            frames = sorted(os.listdir(self.record_dir))
            frame_numbers = []
            timestamps = []
            ca_values = []

            for frame in frames:
                if frame.endswith(".png"):
                    # Extract frame number
                    frame_num = int(frame.split("_")[1])
                    frame_numbers.append(frame_num)
                    # Extract CA value from filename
                    ca_str = frame.split("_ca_")[1].split("_mm")[0]
                    ca_values.append(float(ca_str))
                    # Calculate timestamp based on frame number (at 15 fps)
                    timestamps.append(frame_num / 15.0)  # 15 fps

            # Save timestamps and CA values to CSV
            csv_file = os.path.join(
                video_dir, f"ca_data_{centerline_name}_{timestamp}.csv"
            )
            with open(csv_file, "w") as f:
                f.write("frame,timestamp,curvilinear_abscissa\n")
                for frame, t, ca in zip(frame_numbers, timestamps, ca_values):
                    f.write(f"{frame},{t:.3f},{ca:.3f}\n")

            print("[INFO] Converting images to video with ffmpeg...")
            cmd = [
                "ffmpeg",
                "-y",  # overwrite output if exists
                "-framerate",
                "15",
                "-pattern_type",
                "glob",
                "-i",
                os.path.join(self.record_dir, "frame_*_ca_*_mm.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video,
            ]
            subprocess.run(cmd, check=True)
            print(f"[INFO] ffmpeg video saved as {os.path.basename(video)}")
            print(
                f"[INFO] Curvilinear abscissa data saved to {os.path.basename(csv_file)}"
            )

            # Delete all the frames
            shutil.rmtree(self.record_dir)

        # Finally, exit the Python process
        self.userExit()


if __name__ == "__main__":
    app = MyApp()
    app.run()
