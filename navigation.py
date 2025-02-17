import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from direct.showbase.ShowBase import ShowBase  # type: ignore
import configparser

# Read width and height from config.ini
cfg = configparser.ConfigParser()
cfg.read("config.ini")
width = int(cfg["CAMERA"]["width"])
height = int(cfg["CAMERA"]["height"])

from panda3d.core import *  # type: ignore

# Set the window size and title before anything else
loadPrcFileData("", f"win-size {width} {height}")  # type: ignore
loadPrcFileData("", "window-title Bronchoscopy Simulation")  # type: ignore
loadPrcFileData("", "load-file-type p3assimp")  # type: ignore

from direct.task import Task  # type: ignore
from direct.gui.DirectGui import DirectLabel  # type: ignore
import pyvista as pv  # type: ignore
import numpy as np  # type: ignore
import socket
import math
import cv2
from set_FS_frame import (
    interpolate_line,
    compute_tangent_vectors,
    compute_MRF,
    smooth_vectors,
)

# TODO: Check all the measurements units and make sure they are consistent
# TODO: videos are not saved with the desired resolution

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
        self.videos_dir = self.app_config["PATHS"]["record_dir"]

        self.draw_circles_bool = self.app_config["DRAW"]["draw_circles_bool"]
        self.draw_centerline_bool = self.app_config["DRAW"]["draw_centerline_bool"]
        self.draw_frames_bool = self.app_config["DRAW"]["draw_frames_bool"]
        self.draw_reference_frames_bool = self.app_config["DRAW"][
            "draw_reference_frames_bool"
        ]

        self.sim_server_bool = self.app_config["SLAM"]["sim_server_bool"]

        self.depth_bool = self.app_config["CAMERA"]["depth_bool"]

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

        # --- Set up Depth Buffer & Camera for Depth Image ---
        # Create window properties matching the main window's size.
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        # Define framebuffer properties and request a depth channel.
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)
        # Create an offscreen buffer for the depth image.
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe,
            "depth buffer",
            -2,
            fbprops,
            winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(),
            self.win,
        )
        # Create a texture to store depth values.
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        # Attach the depth texture to the depth buffer.
        self.depthBuffer.addRenderTexture(
            self.depthTex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth
        )
        # Use the same lens as your main camera.
        lens = self.cam.node().getLens()
        # Create a camera that renders the scene into the depth buffer.
        self.depthCam = self.makeCamera(self.depthBuffer, lens=lens, scene=self.render)
        # Parent the depth camera to the main camera to follow its movement.
        self.depthCam.reparentTo(self.cam)
        # --- End of Depth Setup ---

        # Set background color
        self.setBackgroundColor(0, 0.168627, 0.211765, 1.0)

        if self.record_mode == False:
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

        # # Load basic environment
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

        # Setup
        self.setup_line(points)
        self.points = points

        # Init transformation matrices
        self.c_T_w = np.eye(4)
        self.w_T_o = self.setup_w_T_o()

        # Load the model
        if self.view_mode == "fp":
            self.setup_fp()

        elif self.view_mode == "tp":
            self.setup_tp()

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

    def save_calibration_file(self, width, height, fx, fy, cx, cy, filename):
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
        if self.depth_bool == "0":
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
        else:
            yaml_content = f"""%YAML:1.0
Camera.RGB: 1
Stereo.ThDepth: 40.0
Stereo.b: 0.07732
RGBD.DepthMapFactor: 5000.0
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
        if self.depth_bool == "1":
            calibration_filename = "calibration_sim_rgbd.yaml"
        else:
            calibration_filename = "calibration_sim_mono.yaml"

        self.save_calibration_file(width, height, fx, fy, cx, cy, calibration_filename)

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

    def setup_fp(self):
        print("[INFO] Initializing First Person View Mode...")
        self.model = self.data_folder + self.negative_model_name

        # Load the negative model
        self.scene = self.loader.loadModel(self.model)
        self.scene.reparentTo(self.render)
        self.scene.setTransparency(TransparencyAttrib.MDual)  # type: ignore
        self.scene.setColorScale(1, 1, 1, 1)
        self.scene.setTwoSided(True)

        # Adjust material
        myMaterial = Material()  # type: ignore
        myMaterial.setShininess(80)
        myMaterial.setSpecular((0.9, 0.9, 0.9, 1))
        myMaterial.setAmbient((0.3, 0.3, 0.3, 1))
        myMaterial.setDiffuse((0.7, 0.2, 0.2, 1))
        self.scene.setMaterial(myMaterial, 1)

        # (Optional) enable auto-shader
        self.render.setShaderAuto()

        # Add a brighter ambient light
        ambientLight = AmbientLight("ambientLight")  # type: ignore
        ambientLight.setColor((0.5, 0.5, 0.5, 1))  # Brighter
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        # Add a directional light
        directionalLight = DirectionalLight("directionalLight")  # type: ignore
        directionalLight.setColor((1, 1, 1, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setHpr(45, -45, 0)
        self.render.setLight(directionalLightNP)
        self.directionalLightNP = directionalLightNP

        # Add a point light that moves with the camera
        pointLight = PointLight("pointLight")  # type: ignore
        pointLight.setColor((1, 1, 1, 1))
        # Tweak attenuation: constant=1, linear=0, quadratic=0.02
        pointLight.setAttenuation((1, 0, 0.02))

        self.pointLightNP = self.camera.attachNewNode(pointLight)
        self.pointLightNP.setPos(0, 0, 0)  # Right at the camera
        self.render.setLight(self.pointLightNP)

        # Optionally store them so we can reference or tweak later
        self.ambientLightNP = ambientLightNP

    def setup_tp(self):
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
            if self.record_mode == False:
                if controlState:
                    self.highlight_arrow("up")
                else:
                    self.unhighlight_arrow("up")
        elif controlName == "robot_tip_backward":
            if self.record_mode == False:
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
        Create a directory for recording and subdirectories for RGB and depth images.
        Also prepare files for associations and CA data.
        Everything will be saved under this record_dir.
        """
        self.record_frame_idx = 0
        if self.record_mode:
            # Use a persistent record directory.
            self.record_dir = os.path.join(os.getcwd(), "recorded_frames")
            os.makedirs(self.record_dir, exist_ok=True)
            # Subdirectories for rgb and depth images
            self.rgb_dir = os.path.join(self.record_dir, "rgb")
            self.depth_dir = os.path.join(self.record_dir, "depth")
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)
            # Association file in TUM format (timestamps, rgb file, timestamps, depth file)
            self.assoc_file = os.path.join(self.record_dir, "associations.txt")
            with open(self.assoc_file, "w") as f:
                f.write("")
            # CSV file to store CA data (frame number, timestamp, CA)
            self.ca_csv_file = os.path.join(self.record_dir, "ca_data.csv")
            with open(self.ca_csv_file, "w") as f:
                f.write("frame,timestamp,curvilinear_abscissa\n")
            print(f"[INFO] Recording frames to {self.record_dir}")
        else:
            self.record_dir = None

    def record_frame(self):
        """
        Capture the current window as two images (RGB and, if enabled, depth) and save them.
        Update an association file (TUM format) and a CSV file with the current curvilinear abscissa (CA).
        """
        if not self.record_dir:
            return  # Not recording

        # Get a high-precision timestamp (adjust as needed)
        timestamp = time.time()
        timestamp_str = f"{timestamp:.6f}"

        # --- Save RGB Image ---
        screenshot = self.win.getScreenshot()
        rgb_data = screenshot.getRamImage()
        if rgb_data is None:
            print("RGB screenshot not ready!")
            return
        rgb = np.frombuffer(rgb_data, np.uint8)
        rgb.shape = (
            screenshot.getYSize(),
            screenshot.getXSize(),
            screenshot.getNumComponents(),
        )
        rgb = np.flipud(rgb)
        # Remove alpha channel if present.
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        rgb_filename = (
            f"{timestamp_str}_ca_{self.current_ca:.2f}_mm.png"
            if hasattr(self, "current_ca")
            else f"{timestamp_str}.png"
        )
        rgb_filepath = os.path.join(self.rgb_dir, rgb_filename)
        cv2.imwrite(rgb_filepath, rgb)

        # --- Save Depth Image (if enabled) ---
        depth_filename = ""
        if self.depth_bool:
            depth = self.get_depth_image()
            if depth is not None:
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_norm = depth_norm.astype(np.uint8)
                if depth_norm.ndim == 2 or depth_norm.shape[2] == 1:
                    depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
                else:
                    depth_bgr = depth_norm
                depth_filename = (
                    f"{timestamp_str}_ca_{self.current_ca:.2f}_mm.png"
                    if hasattr(self, "current_ca")
                    else f"{timestamp_str}.png"
                )
                depth_filepath = os.path.join(self.depth_dir, depth_filename)
                cv2.imwrite(depth_filepath, depth_bgr)
            else:
                print("Depth image not ready; skipping depth for this frame.")

        # --- Update Association File ---
        with open(self.assoc_file, "a") as f:
            if self.depth_bool and depth_filename:
                f.write(
                    f"{timestamp_str} rgb/{rgb_filename} {timestamp_str} depth/{depth_filename}\n"
                )
            else:
                f.write(f"{timestamp_str} rgb/{rgb_filename}\n")

        # --- Update CA CSV File ---
        if hasattr(self, "current_ca"):
            with open(self.ca_csv_file, "a") as f:
                f.write(
                    f"{self.record_frame_idx},{timestamp_str},{self.current_ca:.2f}\n"
                )

        self.record_frame_idx += 1

    def quit_app(self):
        """
        Called when the user presses 'q'.
        Stops the main loop, runs ffmpeg to encode video (from the RGB images),
        and then exits.
        All output (RGB, depth, associations, CA data, and final video)
        is saved under record_dir.
        """
        print("[INFO] Quitting the app now.")
        # Stop the Panda3D main loop
        self.taskMgr.stop()

        # If we recorded frames, encode them and generate CSV data.
        if self.record_mode and hasattr(self, "record_dir"):
            # Extract centerline name from path (for naming purposes)
            centerline_name = os.path.splitext(os.path.basename(self.path_name))[0]

            # Final video will be saved in record_dir.
            video_name = f"record_{centerline_name}_{time.time()}"
            video = os.path.join(self.record_dir, f"{video_name}.mp4")

            # Build ffmpeg command to generate a video from the RGB images.
            # We assume a frame rate of 15 fps.
            if sys.platform.startswith("linux"):
                print("[INFO] Converting images to video with ffmpeg (Linux)...")
                cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite if exists.
                    "-framerate",
                    "15",
                    "-pattern_type",
                    "glob",
                    "-i",
                    os.path.join(self.rgb_dir, "*.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    video,
                ]
            elif sys.platform.startswith("win"):
                print("[INFO] Converting images to video with ffmpeg (Windows)...")
                file_list_path = os.path.join(self.record_dir, "frames.txt")
                rgb_frames = sorted(os.listdir(self.rgb_dir))
                with open(file_list_path, "w") as f:
                    for frame in rgb_frames:
                        full_path = os.path.join(self.rgb_dir, frame).replace("\\", "/")
                        f.write(f"file '{full_path}'\n")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-r",
                    "15",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    file_list_path,
                    "-r",
                    "15",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    video,
                ]
            subprocess.run(cmd, check=True)
            print(f"[INFO] ffmpeg video saved as {os.path.basename(video)}")

            # Copy the content of record_dir into the data_folder + video without the .mp4
            final_path = os.path.join(self.data_folder, self.videos_dir, video_name)
            shutil.copytree(self.record_dir, final_path)

            # Remove the temp folder
            shutil.rmtree(self.record_dir)

        # Finally, exit the Python process.
        self.userExit()

    # UTILS
    def get_depth_image(self):
        """
        Returns the current depth image as a NumPy array (float32, values between 0.0 and 1.0).
        """
        # Retrieve the raw depth data from the texture.
        data = self.depthTex.getRamImage()
        if data is None or len(data) == 0:
            print("Depth image not ready yet!")
            return None
        # Convert the raw data to a NumPy array.
        depth_image = np.frombuffer(data, dtype=np.float32)
        # Reshape according to the texture's dimensions.
        depth_image.shape = (
            self.depthTex.getYSize(),
            self.depthTex.getXSize(),
            self.depthTex.getNumComponents(),
        )
        # Flip vertically (Panda3D's origin is bottom-left).
        depth_image = np.flipud(depth_image)
        return depth_image


if __name__ == "__main__":
    app = MyApp()
    app.run()
