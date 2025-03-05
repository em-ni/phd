import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from direct.showbase.ShowBase import ShowBase  # type: ignore
import configparser
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from scipy.spatial import cKDTree
import heapq

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
    save_frames_single_branch,
    convert_fs_to_tum,
    interpolate_fs_frames,
)

# TODO: Check all the measurements units and make sure they are consistent
# TODO: Minimize the rotation and translation from frames of different branches

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
parser.add_argument(
    "-autopilot",
    action="store_true",
    help="Enable continuous forward motion (autopilot)",
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
        self.all_branches_bool = self.app_config["PATHS"]["all_branches_bool"]

        self.draw_circles_bool = self.app_config["DRAW"]["draw_circles_bool"]
        self.draw_centerline_bool = self.app_config["DRAW"]["draw_centerline_bool"]
        self.draw_frames_bool = self.app_config["DRAW"]["draw_frames_bool"]
        self.draw_reference_frames_bool = self.app_config["DRAW"][
            "draw_reference_frames_bool"
        ]

        self.sim_server_bool = self.app_config["SLAM"]["sim_server_bool"]

        self.depth_bool = self.app_config["CAMERA"]["depth_bool"]
        self.save_vis_depth = self.app_config["CAMERA"]["save_vis_depth_bool"]

        # Define path of the .vtp file
        self.path_path = self.data_folder + self.path_name

        # Read parameters from command line
        self.view_mode = args.view
        self.live_mode = args.live
        self.record_mode = args.record
        self.autopilot = args.autopilot
        print("\nCommand line arguments:")
        print(f"-View: {self.view_mode}")
        print(f"-Live: {self.live_mode}")
        print(f"-Record: {self.record_mode}")
        print(f"-Autopilot: {self.autopilot}\n")

        if self.live_mode and self.autopilot:
            print(
                "[WARNING] Autopilot cannot be used in live mode. Disabling autopilot."
            )
            self.autopilot = False

        # Quit app on "q"
        self.accept("q", self.quit_app)

        # Set up camera parameters
        if self.view_mode == "fp":
            self.setup_camera_params()

        self.connected = False

        # --- Set up Depth Buffer & Camera for Depth Image ---
        # Create window properties matching the main window's size.
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())  # type: ignore
        # Define framebuffer properties and request a depth channel.
        fbprops = FrameBufferProperties()  # type: ignore
        fbprops.setDepthBits(1)
        # Create an offscreen buffer for the depth image.
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe,
            "depth buffer",
            -2,
            fbprops,
            winprops,
            GraphicsPipe.BFRefuseWindow,  # type: ignore
            self.win.getGsg(),
            self.win,
        )
        # Create a texture to store depth values.
        self.depthTex = Texture()  # type: ignore
        self.depthTex.setFormat(Texture.FDepthComponent)  # type: ignore
        # Attach the depth texture to the depth buffer.
        self.depthBuffer.addRenderTexture(
            self.depthTex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth  # type: ignore
        )
        # Use the same lens as your main camera.
        lens = self.cam.node().getLens()
        # Create a camera that renders the scene into the depth buffer.
        self.depthCam = self.makeCamera(self.depthBuffer, lens=lens, scene=self.render)
        # Parent the depth camera to the main camera to follow its movement.
        self.depthCam.reparentTo(self.cam)

        print(f"[INFO] Save depth map: {self.depth_bool}")
        print(f"[INFO] Save visual depth map: {self.save_vis_depth}")
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

        if self.all_branches_bool == "1":
            # Crate a trajectory traversing all the branches in the folder forward and backward
            print("[INFO] Building final path combining all branches...")
            fs_frames, points = self.build_all_branches_path()

            if not fs_frames:
                print("[ERROR] No branches found. Exiting...")
                sys.exit(1)

            self.setup_line_all_branches(fs_frames)
            print("[INFO] Final path built successfully")
            self.points = points

        else:
            # Get centerline points from the .vtp
            points = self.get_vtp_line_points()

            # print number of points
            print("[INFO] Number of points in the centerline: ", len(points))

            # Setup
            self.setup_line(points)
            self.points = points

        # Init transformation matrices
        self.w_T_c = np.eye(4)
        self.o_T_fs = np.eye(4)

        R_n = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        self.fsi_T_ci = np.eye(4)
        self.fsi_T_ci[:3, :3] = R_n

        self.o_T_w = self.setup_o_T_w()

        # Load the model
        if self.view_mode == "fp":
            self.setup_fp()

        elif self.view_mode == "tp":
            self.setup_tp()

        print("[INFO] Initialization done")

        if self.autopilot:
            print(
                "[INFO] Autopilot enabled. Starting continuous forward motion for full lungs inspection..."
            )

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
                        w_T_c_string = data.decode()
                        lines = w_T_c_string.splitlines()

                        # Convert each line into a list of floats
                        matrix = [list(map(float, line.split())) for line in lines]
                        # Convert the list of lists into a NumPy array
                        self.w_T_c = np.array(matrix)
                        self.connected = True
                        print(f"Received: {self.w_T_c}")
                        time.sleep(1)
                    print("Connection closed")

    def sim_server(self, host="127.0.0.1", port=12345):
        time.sleep(1)  # Give the server time to start
        print("Starting sim_server...")
        try:
            # Create a socket and connect to the server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))

            while True:

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

        # Set camera baseline
        baseline = 0.07732
        bf = fx * baseline
        depth_map_factor = float(self.app_config["CAMERA"]["depth_map_factor"])

        # Construct the file content as a multi-line string
        if self.depth_bool == "0":
            yaml_content = f"""%YAML:1.0
Camera.RGB: 1
Camera.ThDepth: 40.0
Camera.bf: {bf}
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
Stereo.b: {baseline}
RGBD.DepthMapFactor: {depth_map_factor}
Camera.bf: {bf}
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
        print("[INFO] Setting up camera parameters for first-person view mode")
        # 1) Camera parameters
        # Read camera parameters from config.ini
        width = int(self.app_config["CAMERA"]["width"])
        height = int(self.app_config["CAMERA"]["height"])
        fx = float(self.app_config["CAMERA"]["fx"])
        fy = float(self.app_config["CAMERA"]["fy"])
        cx = float(self.app_config["CAMERA"]["cx"])
        cy = float(self.app_config["CAMERA"]["cy"])
        np = float(self.app_config["CAMERA"]["np"])
        fp = float(self.app_config["CAMERA"]["fp"])

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
        self.camLens.setNearFar(np, fp)

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

    def setup_line_all_branches(self, fs_frames):
        # Extract translation and axes from each FS frame:
        self.interpolated_points = np.array([fs[:3, 3] for fs in fs_frames])
        self.tangents = np.array([fs[:3, 0] for fs in fs_frames])
        self.normals = np.array([fs[:3, 1] for fs in fs_frames])
        self.binormals = np.array([fs[:3, 2] for fs in fs_frames])

        # Set first and end point
        self.start_point = self.interpolated_points[0]
        self.end_point = self.interpolated_points[-1]

        # Initialize the robot tip
        self.robot_tip = self.interpolated_points[
            0
        ]  # Setting the first point as the start
        self.robot_tip_node = None

        # Compute total line length without the curvilinear abscissa function but adding all the distances between consecutive points from the first to the last
        self.line_length = 0
        for i in range(len(self.interpolated_points) - 1):
            segment = self.interpolated_points[i + 1] - self.interpolated_points[i]
            self.line_length += np.linalg.norm(segment)

        print("[INFO] Centerline length: ", self.line_length, "mm")

        # Set current index to 0
        self.current_index = 0
        self.next_index = 1

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

    def setup_o_T_w(self):
        """The first point of the centerline (i = 0) corresponds to the transformation from the world frame to the origin frame
        o_T_w = o_T_fs0 * fs0_T_c0"""
        o_T_fs0 = np.eye(4)

        # Set the transformation matrix from the origin to the first point using self.tangents, self.normals, and self.binormals
        o_T_fs0[:3, 0] = self.tangents[0]
        o_T_fs0[:3, 1] = self.normals[0]
        o_T_fs0[:3, 2] = self.binormals[0]
        o_T_fs0[:3, 3] = self.interpolated_points[0]

        o_T_c0 = np.dot(o_T_fs0, self.fsi_T_ci)

        return o_T_c0

    ## LINE UTILS
    def build_all_branches_path(self):
        """
        Reads multiple branch VTP files (listed in the config entry 'branch_files' under [BRANCHES]),
        computes the FS frame for every point in each branch, and stacks them together
        by first appending the branch (forward) and then appending the same branch in reverse (back).
        Returns a list of 4x4 FS frame matrices.
        """
        # Get the full path to the centerline folder
        centerline_folder_name = self.app_config["PATHS"]["all_branches_folder"]
        centerline_folder_path = os.path.join(self.data_folder, centerline_folder_name)
        print(f"[INFO] Looking for .vtp files in: {centerline_folder_path}")

        # Find all .vtp files in the folder
        branch_files = []
        for file in os.listdir(centerline_folder_path):
            if file.endswith(".vtp"):
                full_path = os.path.join(centerline_folder_name, file)
                branch_files.append(full_path)

        if not branch_files:
            print("[ERROR] No .vtp files found in the specified folder")
            return []

        print(f"[INFO] Found {len(branch_files)} branch files: {branch_files}")

        final_frames = []
        all_points = []  # Initialize a list to collect all points

        for branch_file in branch_files:
            branch_file = branch_file.strip()
            branch_path = os.path.join(self.data_folder, branch_file)
            print(f"[INFO] Processing branch file: {branch_path}")
            branch_model = pv.read(branch_path)
            # Discard the first n_d points of each branch
            n_d = 0
            if len(branch_model.points) > n_d:
                branch_points = [tuple(point) for point in branch_model.points[n_d:]]
            else:
                print(
                    f"[WARNING] Branch has fewer than {n_d} points ({len(branch_model.points)}). Skipping branch."
                )
                continue

            # Interpolate the branch points
            interp_points = interpolate_line(branch_points, num_points=1000)
            branch_tangents = compute_tangent_vectors(interp_points)
            branch_tangents = smooth_vectors(branch_tangents, 10, 10)
            branch_normals, branch_binormals = compute_MRF(branch_tangents)

            # FORWARD TRAVEL: Build FS frames for each point along the branch
            for i, pt in enumerate(interp_points):
                fs_frame = np.eye(4)
                fs_frame[:3, 0] = branch_tangents[i]
                fs_frame[:3, 1] = branch_normals[i]
                fs_frame[:3, 2] = branch_binormals[i]
                fs_frame[:3, 3] = pt

                if len(final_frames) > 0 and i == 0:
                    # Interpolate the first frame with the last frame added to the final_frames list
                    # This is to glue the branches together smoothly
                    extra_frames = interpolate_fs_frames(
                        final_frames[-1], fs_frame, num_points=10
                    )
                    final_frames.extend(extra_frames)
                    all_points.extend([f[:3, 3] for f in extra_frames])
                else:
                    final_frames.append(fs_frame)
                    all_points.append(pt)

            # RETURN TRAVEL: Append the same points in reverse order.
            # Skip the last point of the forward travel when creating the return path
            for i, pt in enumerate(interp_points[-2::-1]):
                fs_frame = np.eye(4)
                # Adjust the index calculation since we're starting from second-to-last point
                idx = len(interp_points) - 2 - i
                fs_frame[:3, 0] = branch_tangents[idx]  # Invert tangent for return
                fs_frame[:3, 1] = branch_normals[idx]
                fs_frame[:3, 2] = branch_binormals[idx]
                fs_frame[:3, 3] = pt
                final_frames.append(fs_frame)
                all_points.append(pt)  # Add point to all_points list

        # Reduce the total number of points
        print(f"[INFO] Initial number of points: {len(final_frames)}")
        divide_factor = 4
        final_frames = final_frames[::divide_factor]
        all_points = all_points[::divide_factor]
        print(f"[INFO] Reduced number of points: {len(final_frames)}")
        return final_frames, all_points

    def curvilinear_abscissa(self, current_point):
        if self.all_branches_bool == "1" and self.record_mode == True:
            """
            Compute the curvilinear abscissa (distance) from the current_point to the first point
            by finding the shortest path along the graph of points. The graph is built by connecting
            points that are within a threshold distance of each other (to avoid jumping between branches).
            It's very slow so it's only used in record mode when data is being saved.

            Parameters:
            current_point: (3,) array_like representing the current robot tip position.

            Returns:
            The shortest-path distance from the current_point to the first point in self.interpolated_points.
            """
            # return 0.0
            points = self.interpolated_points  # assumed shape (N, 3)
            n = len(points)
            if n < 2:
                return 0.0

            # Compute a threshold distance based on the average distance between consecutive points.
            diffs = points[1:] - points[:-1]
            mean_distance = np.mean(np.linalg.norm(diffs, axis=1))
            threshold = (
                2 * mean_distance
            )  # factor can be tuned; it prevents jumps between distant branches

            # Build a KD-tree for efficient neighbor search.
            tree = cKDTree(points)

            # Build a graph (as an adjacency list) connecting each point to its neighbors within threshold.
            graph = {i: [] for i in range(n)}
            for i in range(n):
                # Find all neighbor indices within the threshold radius.
                neighbors = tree.query_ball_point(points[i], r=threshold)
                for j in neighbors:
                    if j == i:
                        continue
                    # Use the Euclidean distance as the edge weight.
                    dist_ij = np.linalg.norm(points[i] - points[j])
                    graph[i].append((j, dist_ij))

            # Identify the node corresponding to the current point.
            current_index = np.argmin(np.linalg.norm(points - current_point, axis=1))
            target_index = 0  # we want the distance to the first point

            # Run Dijkstra's algorithm from current_index to target_index.
            distances = {i: float("inf") for i in range(n)}
            distances[current_index] = 0.0
            pq = [(0.0, current_index)]
            while pq:
                d, i = heapq.heappop(pq)
                if i == target_index:
                    break
                if d > distances[i]:
                    continue
                for neighbor, weight in graph[i]:
                    new_dist = d + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))

            return distances[target_index]

        else:
            """
            Simpler and faster method to compute the curvilinear abscissa (distance) from the current_point to the first point.
            This works only for single branches and does not consider the possibility of multiple branches with forward and return paths.
            """
            # Find the closest point to the given point
            distances = np.linalg.norm(self.interpolated_points - current_point, axis=1)
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
                    print("[INFO] Robot tip reached the end of the path")
                    return  # Stop moving forward
                next_index = current_index + 1
            else:
                # Check if the robot tip is at the first point
                if current_index == 0:
                    print("[INFO] Robot tip reached the start of the path")
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
            if self.connected == True:
                try:
                    """ "
                    To draw get o_T_fs as
                    o_T_fs = o_T_w * w_T_c * c_T_fs
                    """
                    # Convert string to numpy array and multiply matrices
                    w_T_c_matrix = self.w_T_c
                    o_T_w_matrix = self.o_T_w
                    fs_T_c_matrix = self.fsi_T_ci
                    c_T_fs_matrix = np.linalg.inv(fs_T_c_matrix)

                    # Compute the robot tip position
                    o_T_c_matrix = np.dot(o_T_w_matrix, w_T_c_matrix)
                    o_T_fs_matrix = np.dot(o_T_c_matrix, c_T_fs_matrix)
                    self.o_T_fs = o_T_fs_matrix
                    translation = o_T_fs_matrix[:3, 3]

                    # Update the robot tip position
                    self.robot_tip = translation
                    print(f"Robot tip position: {self.robot_tip}")

                except (SyntaxError, AttributeError) as e:
                    print(f"Error in update_robot_tip_position: {e}")
                    pass

        # Update the visual representation
        self.draw_robot_tip()

    def update_tip_position_all_branches(self, dt, forward):
        """
        Automatically updates the robot tip position along the centerline for the
        all_branches_bool mode. If the translation difference is near zero (i.e. the
        two FS frames share the same position but differ in orientation), then
        perform an orientation interpolation via SLERP.
        TODO: it can probably be deleted and merged inside update_robot_tip_position
        """

        # Determine the next index based on the direction.
        if forward:
            if self.current_index >= len(self.interpolated_points) - 1:
                print("[INFO] Robot tip reached the end of the path")
                if self.autopilot:
                    # Deactivate autopilot mode
                    self.autopilot = False
                    print("[INFO] Autopilot mode deactivated")
                    print("[INFO] Exiting application...")
                    self.quit_app()
                return  # Stop moving forward
            next_index = self.current_index + 1
        else:
            if self.current_index <= 0:
                print("[INFO] Robot tip reached the start of the path")
                return  # Stop moving backward
            next_index = self.current_index - 1

        # Compute translation difference between current and next FS frames.
        current_pos = self.interpolated_points[self.current_index]
        next_pos = self.interpolated_points[next_index]
        delta_pos = next_pos - current_pos
        dist = np.linalg.norm(delta_pos)

        # If translation difference is significant, use normal update.
        if dist > 1e-5:
            # Normal update: move toward next point along the translation.
            direction = delta_pos / dist  # Safe normalization
            # Define a movement step (you can adjust movement_speed as needed)
            movement_speed = 5000  # units per second
            step = movement_speed * dt
            # Don't overshoot the next point.
            if step > dist:
                step = dist
            new_pos = self.robot_tip + direction * step
            self.robot_tip = new_pos

            # When we've nearly reached the next point, snap to it and reset interpolation.
            if np.linalg.norm(new_pos - next_pos) < 1e-3:
                self.robot_tip = next_pos
                self.interp_alpha = 0.0  # reset orientation interpolation
                self.current_index = next_index
        else:
            print(
                "[INFO] Translation difference is near zero. Update only tip orientation."
            )
            # Orientation interpolation via SLERP
            R_current = self.get_rotation_from_index(self.current_index)
            R_next = self.get_rotation_from_index(next_index)

            # Create Slerp instance with key times and rotations
            key_times = [0, 1]  # Start and end times
            rotations = Rotation.from_matrix([R_current, R_next])
            slerp = Slerp(key_times, rotations)

            # Use the instance to interpolate at a specific time
            R_interp = slerp([self.interp_alpha])[0].as_matrix()

            self.interp_alpha += 0.1
            if self.interp_alpha >= 1.0:
                self.interp_alpha = 0.0
                self.current_index = next_index

            # Update the robot tip position
            self.robot_tip = self.interpolated_points[self.current_index]

        if self.view_mode == "tp":
            # Redraw the path up to the robot tip
            self.draw_path(self.interpolated_points, self.current_index)

        # Compute the curvilinear abscissa
        current_point = self.interpolated_points[self.current_index]
        self.current_ca = self.curvilinear_abscissa(current_point)

        # Update next index based on the new current_index.
        if forward:
            self.next_index = min(
                self.current_index + 1, len(self.interpolated_points) - 1
            )
        else:
            self.next_index = max(self.current_index - 1, 0)

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
            if self.autopilot:
                if self.all_branches_bool == "1":
                    self.update_tip_position_all_branches(dt, forward=True)
                else:
                    self.update_robot_tip_position(dt, forward=True)
            else:
                if self.keyMap["robot_tip_forward"]:
                    if self.all_branches_bool == "1":
                        self.update_tip_position_all_branches(dt, forward=True)
                    else:
                        self.update_robot_tip_position(dt, forward=True)
                if self.keyMap["robot_tip_backward"]:
                    if self.all_branches_bool == "1":
                        self.update_tip_position_all_branches(dt, forward=True)
                    else:
                        self.update_robot_tip_position(dt, forward=True)
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
                if hasattr(self, "c_T_w") and not np.array_equal(self.w_T_c, np.eye(4)):
                    print(f"\rReceived: {self.w_T_c}\033[F", end="", flush=True)
            else:
                if self.autopilot:
                    print(
                        f"\rCurrent index: {self.current_index} / {len(self.interpolated_points)}",
                        end="",
                        flush=True,
                    )
                else:
                    if hasattr(self, "current_ca") and self.all_branches_bool == "0":
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

        if self.live_mode == False:
            # Load cone model
            cone_model = self.loader.loadModel(
                "/home/emanuele/Desktop/github/navigation/data/icons/cone.obj"
            )
            cone_model.reparentTo(self.robot_tip_node)

            # Set size and color/transparency
            cone_model.setScale(10)
            cone_model.setColor(1, 1, 0, 0.3)  # Slightly yellow, partial alpha
            cone_model.setTransparency(TransparencyAttrib.MAlpha)  # type: ignore

            # Find the index of the closest point to the robot tip
            distances = np.linalg.norm(
                self.interpolated_points - self.robot_tip, axis=1
            )
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

        else:
            # Load cone model
            cone_model = self.loader.loadModel(
                "/home/emanuele/Desktop/github/navigation/data/icons/cone.obj"
            )
            cone_model.reparentTo(self.robot_tip_node)
            if not cone_model:
                print("Error loading cone model")

            # Set size and color/transparency
            cone_model.setScale(0.5)
            cone_model.setColor(1, 1, 0, 0.3)  # Slightly yellow, partial alpha
            cone_model.setTransparency(TransparencyAttrib.MAlpha)  # type: ignore

            # Use the self.o_T_fs matrix to draw the cone
            # Compute the tangent vector from o_T_fs (if available)
            if hasattr(self, "o_T_fs"):
                tangent_vector = LVector3f(*self.o_T_fs[:3, 0])  # type: ignore
                normal_vector = LVector3f(*self.o_T_fs[:3, 1])  # type: ignore
            else:
                tangent_vector = LVector3f(1, 0, 0)  # type: ignore # Fallback
                normal_vector = LVector3f(0, 1, 0)  # type: ignore # Fallback

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

        w_T_o = np.linalg.inv(self.o_T_w)

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

        robot_tip_visual.setScale(1)  # Scale to appropriate size
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
        if self.depth_bool == "1":
            depth = self.get_depth_image()  # normalized, from 0 to 1
            if depth is not None:
                near_plane = float(
                    self.app_config["CAMERA"]["np"]
                )  # in meters, e.g., 0.1
                far_plane = float(
                    self.app_config["CAMERA"]["fp"]
                )  # in meters, e.g., 100.0

                # Linearize the depth (result in meters)
                depth_linear = (2.0 * near_plane * far_plane) / (
                    far_plane
                    + near_plane
                    - (2.0 * depth - 1.0) * (far_plane - near_plane)
                )
                # (Do not divide by 1000.0 if near and far are in meters.)

                # For ORB-SLAM raw data: if you want to save as uint16 with 1m -> 5000:
                depth_map_factor = float(self.app_config["CAMERA"]["depth_map_factor"])
                raw_depth = (depth_linear * depth_map_factor).astype(np.uint16)
                # Save the raw depth image (you might choose a different filename convention)
                depth_filename = (
                    f"{timestamp_str}_raw_ca_{self.current_ca:.2f}_mm.png"
                    if hasattr(self, "current_ca")
                    else f"{timestamp_str}_raw.png"
                )
                depth_filepath = os.path.join(self.depth_dir, depth_filename)
                cv2.imwrite(depth_filepath, raw_depth)

                # For visualization: create a color version with grid overlay.
                if self.save_vis_depth == "1":
                    vis_depth = cv2.normalize(
                        depth_linear, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    vis_depth_color = cv2.cvtColor(vis_depth, cv2.COLOR_GRAY2BGR)
                    block_size = 50
                    h, w = vis_depth_color.shape[:2]
                    for y in range(0, h, block_size):
                        for x in range(0, w, block_size):
                            block = depth_linear[
                                y : min(y + block_size, h), x : min(x + block_size, w)
                            ]
                            avg_depth = np.mean(block)
                            text = f"{avg_depth:.2f}"
                            cv2.putText(
                                vis_depth_color,
                                text,
                                (x + 5, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.rectangle(
                                vis_depth_color,
                                (x, y),
                                (min(x + block_size, w), min(y + block_size, h)),
                                (255, 0, 0),
                                1,
                            )
                    # Save visualization (if desired)
                    vis_filename = (
                        f"{timestamp_str}_vis_ca_{self.current_ca:.2f}_mm.png"
                        if hasattr(self, "current_ca")
                        else f"{timestamp_str}_vis.png"
                    )
                    vis_filepath = os.path.join(self.depth_dir, vis_filename)
                    cv2.imwrite(vis_filepath, vis_depth_color)

                # Update association file using the raw depth image filename.

        else:
            print("Depth image not ready; skipping depth for this frame.")

        # --- Update Association File ---
        with open(self.assoc_file, "a") as f:
            if self.depth_bool == "1" and depth_filename:
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
        Stops the main loop, runs ffmpeg to encode video from the RGB images and (if enabled) from the depth visualization images,
        copies all output (RGB, depth, associations, CA data, and final videos) under record_dir to a permanent folder,
        and then exits.
        """
        print("[INFO] Quitting the app now.")
        # Stop the Panda3D main loop.
        self.taskMgr.stop()

        if self.record_mode and hasattr(self, "record_dir"):
            # Extract centerline name from path (for naming purposes).
            if self.all_branches_bool == "1":
                centerline_name = "ball"
            else:
                centerline_name = os.path.splitext(os.path.basename(self.path_name))[0]

            # Build RGB Video
            rgb_video_name = f"record_rgb_{centerline_name}_{time.time()}"
            rgb_video = os.path.join(self.record_dir, f"{rgb_video_name}.mp4")
            if sys.platform.startswith("linux"):
                print("[INFO] Converting RGB images to video with ffmpeg (Linux)...")
                cmd_rgb = [
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
                    rgb_video,
                ]
            elif sys.platform.startswith("win"):
                print("[INFO] Converting RGB images to video with ffmpeg (Windows)...")
                file_list_path = os.path.join(self.record_dir, "rgb_frames.txt")
                rgb_frames = sorted(os.listdir(self.rgb_dir))
                with open(file_list_path, "w") as f:
                    for frame in rgb_frames:
                        full_path = os.path.join(self.rgb_dir, frame).replace("\\", "/")
                        f.write(f"file '{full_path}'\n")
                cmd_rgb = [
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
                    rgb_video,
                ]
            subprocess.run(cmd_rgb, check=True)
            print(f"[INFO] RGB video saved as {os.path.basename(rgb_video)}")

            # Build Depth Video (if enabled)
            if self.depth_bool == "1" and self.save_vis_depth == "1":
                depth_video_name = f"record_depth_{centerline_name}_{time.time()}"
                depth_video = os.path.join(self.record_dir, f"{depth_video_name}.mp4")
                if sys.platform.startswith("linux"):
                    print(
                        "[INFO] Converting depth images to video with ffmpeg (Linux)..."
                    )
                    file_list_path = os.path.join(self.record_dir, "depth_frames.txt")
                    depth_frames = sorted(os.listdir(self.depth_dir))
                    with open(file_list_path, "w") as f:
                        for frame in depth_frames:
                            # Skip frames with 'raw' in their name
                            if "raw" not in frame:
                                full_path = os.path.join(self.depth_dir, frame).replace(
                                    "\\", "/"
                                )
                                f.write(f"file '{full_path}'\n")
                    cmd_depth = [
                        "ffmpeg",
                        "-y",  # Overwrite if exists.
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
                        depth_video,
                    ]
                elif sys.platform.startswith("win"):
                    print(
                        "[INFO] Converting depth images to video with ffmpeg (Windows)..."
                    )
                    file_list_path = os.path.join(self.record_dir, "depth_frames.txt")
                    depth_frames = sorted(os.listdir(self.depth_dir))
                    with open(file_list_path, "w") as f:
                        for frame in depth_frames:
                            full_path = os.path.join(self.depth_dir, frame).replace(
                                "\\", "/"
                            )
                            # Skip raw depth images (only visualization images).
                            if "raw" not in frame:
                                f.write(f"file '{full_path}'\n")

                    cmd_depth = [
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
                        depth_video,
                    ]
                subprocess.run(cmd_depth, check=True)
                print(f"[INFO] Depth video saved as {os.path.basename(depth_video)}")

            # Save trajectory in TUM format
            vtp_trajectory = os.path.join(self.data_folder, self.path_name)
            fs_trajectory = save_frames_single_branch(vtp_trajectory)
            gt_file_wTc = os.path.join(self.record_dir, "gt", "gt_wTc.txt")
            gt_file_cTw = os.path.join(self.record_dir, "gt", "gt_cTw.txt")

            # If it doesnt exist, create the gt folder
            os.makedirs(os.path.join(self.record_dir, "gt"), exist_ok=True)

            convert_fs_to_tum(fs_trajectory, gt_file_wTc, convention="wTc")
            convert_fs_to_tum(fs_trajectory, gt_file_cTw, convention="cTw")
            print(f"[INFO] Trajectory saved as {gt_file_wTc} and {gt_file_cTw}")

            # Copy Everything to Permanent Storage
            # Use self.videos_dir from config to determine final destination.
            print(
                "[INFO] Copying all recorded data from temp folder to permanent storage..."
            )
            final_path = os.path.join(
                self.data_folder,
                self.videos_dir,
                f"record_{centerline_name}_{time.time()}",
            )
            shutil.copytree(self.record_dir, final_path)
            print(f"[INFO] All recorded data copied to {final_path}")

            # Remove the temporary record folder.
            shutil.rmtree(self.record_dir)

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

    def get_rotation_from_index(self, index):
        """
        Returns a 3x3 rotation matrix from the FS frame at the given index.
        The rotation is built from the tangent, normal, and binormal vectors.
        """
        R = np.eye(3)
        R[:, 0] = self.tangents[index]
        R[:, 1] = self.normals[index]
        R[:, 2] = self.binormals[index]
        return R


if __name__ == "__main__":
    app = MyApp()
    app.run()
