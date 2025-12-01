import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from direct.showbase.ShowBase import ShowBase  # type: ignore
import configparser
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import math
import cv2

# Read width and height from config.ini before importing panda3d
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
from direct.gui.DirectGui import DirectLabel, OnscreenText  # type: ignore

# src import
from src.server import sim_server, start_server
from src.draw import (
    draw_elements,
    draw_path,
    draw_results_trajectories,
    draw_robot_tip,
    draw_trajectory,
    highlight_arrow,
    unhighlight_arrow,
)
from src.utils import (
    build_all_branches_path,
    curvilinear_abscissa,
    filter_trajectory_positions,
    get_depth_image,
    get_rotation_from_index,
    get_vtp_line_points,
    save_fs_frames_multibranch,
    trajectory_snapping,
)

# utils import
from utils.set_FS_frame import (
    interpolate_line,
    compute_tangent_vectors,
    compute_MRF,
    smooth_vectors,
    save_frames_single_branch,
    convert_fs_to_tum,
    convert_tum_to_fs,
)
from utils.align_trajectory import align_umeyama

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Bronchoscopy Simulation")
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
parser.add_argument(
    "-results",
    action="store_true",
    help="Enable results visualization mode",
)
parser.add_argument(
    "-random",
    type=int,
    default=None,
    help="Randomly pick N centerlines to combine (overrides all_branches_bool)",
)

args = parser.parse_args()


class BronchoSim(ShowBase):
    def __init__(self):
        print("[INFO] Starting BronchoSim...")
        ShowBase.__init__(self)

        # Base app setup
        self.setup_init()

        # Mode-dependent setup
        self.setup_mode()
        print("[INFO] Initialization done")

        # Draw elements
        draw_elements(self)

        # Start breathing task if enabled
        if hasattr(self, "breathing_enabled") and self.breathing_enabled:
            self.setup_breathing_task()

    def get_scene_graph_parent_and_offset(self):
        """
        Returns the appropriate parent node for scene elements that should
        participate in the breathing animation, and the offset to transform
        world coordinates to that parent's local space.
        """
        if (
            self.breathing_enabled
            and hasattr(self, "breathing_pivot_node")
            and self.breathing_pivot_node is not None
            and hasattr(self, "start_point")
            and self.start_point is not None
        ):
            # Pivot is active. Elements should be parented to it.
            # The pivot node is positioned at self.start_point in world coordinates.
            # The offset to subtract from world coordinates to make them local to the pivot is self.start_point.
            offset = LVector3f(  # type: ignore
                self.start_point[0], self.start_point[1], self.start_point[2]
            )
            return self.breathing_pivot_node, offset
        else:
            # No breathing, or pivot not set, or start_point not available.
            # Parent to render, no offset needed.
            return self.render, LVector3f(0, 0, 0)  # type: ignore

    ## SETUP METHODS
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

    def setup_depth(self):
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
        myMaterial.setShininess(20.0) # Lower shininess for broader, wetter highlights
        myMaterial.setSpecular((0.4, 0.3, 0.3, 1)) # Pinkish specular for tissue
        myMaterial.setAmbient((0.3, 0.3, 0.3, 1))
        myMaterial.setDiffuse((0.8, 0.8, 0.8, 1)) # White/Grey diffuse to let texture show through
        self.scene.setMaterial(myMaterial, 1)

        # Load and apply texture with Triplanar Shader
        try:
            tex_path = self.texture_name
            self.tex = self.loader.loadTexture(tex_path)
            self.scene.setTexture(self.tex, 1)
            
            # Load Shader
            triplanar_shader = Shader.load(
                Shader.SL_GLSL,
                vertex="shaders/triplanar.vert",
                fragment="shaders/triplanar.frag"
            )
            self.scene.setShader(triplanar_shader)
            
            # Set Shader Inputs (Uniforms)
            self.scene.setShaderInput("texScale", 0.05) # Adjust scale for triplanar (world units)
            self.scene.setShaderInput("lightColor", Vec3(1.5, 1.5, 1.5)) # Match point light intensity
            self.scene.setShaderInput("ambientColor", Vec3(0.2, 0.2, 0.2)) # Match ambient light
            self.scene.setShaderInput("shininess", 20.0)
            self.scene.setShaderInput("k_specular", Vec3(0.4, 0.3, 0.3))
            self.scene.setShaderInput("k_diffuse", Vec3(0.8, 0.8, 0.8))
            
            # We need to update lightPos every frame in update_scene or attach it to camera?
            # Ideally, we pass the camera position as lightPos.
            # Since the light is attached to the camera, lightPos = cameraPos.
            # We will update "lightPos" in the update loop.
            
            print(f"[INFO] Applied Triplanar Shader with texture {tex_path}")
        except Exception as e:
            print(f"[WARNING] Could not load shader/texture: {e}")

        # (Optional) enable auto-shader
        self.render.setShaderAuto()

        # Add a brighter ambient light
        ambientLight = AmbientLight("ambientLight")  # type: ignore
        ambientLight.setColor((0.2, 0.2, 0.2, 1))  # Slightly brighter ambient
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        # Directional light removed to simulate internal organ environment (no sun)
        # directionalLight = DirectionalLight("directionalLight")  # type: ignore
        # directionalLight.setColor((1, 1, 1, 1))
        # directionalLightNP = self.render.attachNewNode(directionalLight)
        # directionalLightNP.setHpr(45, -45, 0)
        # self.render.setLight(directionalLightNP)
        # self.directionalLightNP = directionalLightNP

        # Add a point light that moves with the camera
        pointLight = PointLight("pointLight")  # type: ignore
        # Increase intensity > 1 for brighter light
        pointLight.setColor((1.5, 1.5, 1.5, 1)) 
        # Tweak attenuation: constant=1, linear=0, quadratic=0.02
        # Lower quadratic factor to allow light to reach further
        pointLight.setAttenuation((1, 0, 0.01))

        self.pointLightNP = self.camera.attachNewNode(pointLight)
        self.pointLightNP.setPos(0, 0, 0)  # Right at the camera
        self.render.setLight(self.pointLightNP)

        # Optionally store them so we can reference or tweak later
        self.ambientLightNP = ambientLightNP

    def setup_breathing_pivot(self):
        """
        Sets up the pivot node for the breathing animation.
        Assumes self.scene is loaded and self.start_point is available.
        The scene will be reparented to this pivot node.
        """
        # self.breathing_pivot_node is initialized to None in setup_init
        if (
            self.breathing_enabled
            and hasattr(self, "start_point")
            and self.start_point is not None
            and hasattr(self, "scene")
            and self.scene is not None
        ):

            print(
                f"[INFO] Setting up breathing pivot around start_point: {self.start_point} for scene: {self.scene.getName()}"
            )

            # Create pivot node, parented to render, and position it at start_point (world coordinates)
            self.breathing_pivot_node = self.render.attachNewNode("breathing_pivot")
            pivot_pos_world = LPoint3f(  # type: ignore
                self.start_point[0], self.start_point[1], self.start_point[2]
            )
            self.breathing_pivot_node.setPos(pivot_pos_world)

            # Reparent the scene to the pivot node.
            # wrtReparentTo preserves the scene's current world transform by adjusting its local transform
            # relative to the new parent (the pivot_node).
            self.scene.wrtReparentTo(self.breathing_pivot_node)
            print(
                f"[INFO] Scene '{self.scene.getName()}' reparented to breathing_pivot_node. Pivot at {self.breathing_pivot_node.getPos(self.render)}"
            )

        elif self.breathing_enabled:
            missing_details = []
            if not (hasattr(self, "start_point") and self.start_point is not None):
                missing_details.append("start_point not available")
            if not (hasattr(self, "scene") and self.scene is not None):
                missing_details.append("scene not available")
            print(
                f"[WARNING] Breathing effect enabled but prerequisites ({', '.join(missing_details)}) not met. Pivot not set up."
            )
            self.breathing_pivot_node = None  # Ensure it's None if setup fails

        # Print breathing parameters
        if self.breathing_enabled:
            print("[INFO] Breathing parameters:")
            print(f"  - Breathing enabled: {self.breathing_enabled}")
            print(f"  - Breathing min scale: {self.breathing_min_scale}")
            print(f"  - Breathing max scale: {self.breathing_max_scale}")
            print(f"  - Breathing period: {self.breathing_period_seconds} seconds")

    def setup_breathing_task(self):
        prerequisites_met = True
        missing_prerequisites = []

        if not (hasattr(self, "start_point") and self.start_point is not None):
            prerequisites_met = False
            missing_prerequisites.append("start_point not available")
        if not (hasattr(self, "scene") and self.scene is not None):
            prerequisites_met = False
            missing_prerequisites.append("scene not available")
        if not (
            hasattr(self, "breathing_pivot_node")
            and self.breathing_pivot_node is not None
        ):
            prerequisites_met = False
            missing_prerequisites.append("breathing_pivot_node not set up")

        if prerequisites_met:
            self.taskMgr.add(self.update_breathing_effect, "updateBreathingEffectTask")
            print("[INFO] Breathing task started.")
        else:
            print(
                f"[WARNING] Breathing effect enabled but prerequisites not met ({', '.join(missing_prerequisites)}). Breathing task not started."
            )

    def setup_init(self):
        # Read the configuration file
        self.app_config = configparser.ConfigParser()
        self.app_config.read("config.ini")

        # PATHS
        self.data_folder = self.app_config["PATHS"]["data_folder"]
        self.path_name = self.app_config["PATHS"]["path_name"]
        self.model_name = self.app_config["PATHS"]["model_name"]
        self.negative_model_name = self.app_config["PATHS"]["negative_model_name"]
        self.videos_dir = self.app_config["PATHS"]["record_dir"]
        self.logs_dir = self.app_config["PATHS"]["logs_dir"]
        self.all_branches_bool = self.app_config["PATHS"]["all_branches_bool"]
        self.texture_name = self.app_config["PATHS"]["texture_name"]

        # RECORD
        self.legacy_record_method = self.app_config["RECORD"].getboolean(
            "legacy_record_method", False
        )

        # DRAW
        self.draw_circles_bool = self.app_config["DRAW"]["draw_circles_bool"]
        self.draw_centerline_bool = self.app_config["DRAW"]["draw_centerline_bool"]
        self.draw_frames_bool = self.app_config["DRAW"]["draw_frames_bool"]
        self.draw_reference_frames_bool = self.app_config["DRAW"][
            "draw_reference_frames_bool"
        ]

        # SLAM
        self.sim_server_bool = self.app_config["SLAM"]["sim_server_bool"]

        # CAMERA
        self.depth_bool = self.app_config["CAMERA"]["depth_bool"]
        self.save_vis_depth = self.app_config["CAMERA"]["save_vis_depth_bool"]
        self.orientation_smoothing = self.app_config["CAMERA"].getfloat(
            "orientation_smoothing", fallback=0.2
        )
        self.orientation_smoothing = min(max(self.orientation_smoothing, 0.0), 1.0)

        # BREATHING
        self.breathing_enabled = self.app_config["BREATHING"].getboolean(
            "enabled", False
        )
        self.breathing_min_scale = self.app_config["BREATHING"].getfloat(
            "min_scale", 0.98
        )
        self.breathing_max_scale = self.app_config["BREATHING"].getfloat(
            "max_scale", 1.02
        )
        self.breathing_period_seconds = self.app_config["BREATHING"].getfloat(
            "period_seconds", 5.0
        )
        if self.breathing_period_seconds <= 0:  # Prevent division by zero
            self.breathing_period_seconds = 5.0
            print("[WARNING] Breathing period must be positive. Defaulting to 5.0s.")

        # Define path of the .vtp file
        self.path_path = self.data_folder + self.path_name

        # Read parameters from command line
        self.view_mode = args.view
        self.live_mode = args.live
        self.record_mode = args.record
        self.autopilot = args.autopilot
        self.results_mode = args.results

        print("\nCommand line arguments:")
        print(f"-View: {self.view_mode}")
        print(f"-Live: {self.live_mode}")
        print(f"-Record: {self.record_mode}")
        print(f"-Autopilot: {self.autopilot}")
        print(f"-Results mode: {self.results_mode}\n")

        # Initialize keyMap with default values to ensure it always exists
        self.keyMap = {"robot_tip_forward": False, "robot_tip_backward": False}

        # Set background color
        self.setBackgroundColor(0, 0.168627, 0.211765, 1.0)

        # Quit app on "q"
        self.accept("q", self.quit_app)

        # Task for updating the scene
        self.taskMgr.add(self.update_scene, "updateScene")

        # Set antialiasing
        self.render.setAntialias(AntialiasAttrib.MAuto)  # type: ignore

        # Init variables
        self.connected = False
        self.blink_timer = 0
        self.blink_interval = 1
        self.robot_tip_visible = False
        self.breathing_pivot_node = None

        # Finish initial setup
        self.setup_points()
        self.setup_matrices()

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

        # Initialize smoothed orientation vectors for camera smoothing
        self.smoothed_tangent = self.tangents[0].copy() if len(self.tangents) > 0 else None
        self.smoothed_normal = self.normals[0].copy() if len(self.normals) > 0 else None

        # Set first and end point
        self.start_point = self.interpolated_points[0]
        self.end_point = self.interpolated_points[-1]

        # Initialize the robot tip
        self.robot_tip = self.interpolated_points[
            0
        ]  # Setting the first point as the start
        self.robot_tip_node = None

        # Initialize traversal indices for camera/robot updates
        self.current_index = 0
        self.next_index = 1 if len(self.interpolated_points) > 1 else 0

        # Compute line length
        self.line_length = curvilinear_abscissa(
            self.end_point,
            self.interpolated_points,
            self.all_branches_bool,
            self.record_mode,
        )
        print("[INFO] Centerline length: ", self.line_length, "mm")

    def setup_line_multibranch(self, fs_frames):
        # Extract translation and axes from each FS frame:
        self.interpolated_points = np.array([fs[:3, 3] for fs in fs_frames])
        self.tangents = np.array([fs[:3, 0] for fs in fs_frames])
        self.normals = np.array([fs[:3, 1] for fs in fs_frames])
        self.binormals = np.array([fs[:3, 2] for fs in fs_frames])

    # Initialize smoothed orientation vectors for camera smoothing
        self.smoothed_tangent = self.tangents[0].copy() if len(self.tangents) > 0 else None
        self.smoothed_normal = self.normals[0].copy() if len(self.normals) > 0 else None

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

    def setup_live_mode(self):
        # Start the server in a thread
        self.listen_thread = threading.Thread(
            target=start_server, args=(self,), daemon=True
        )
        self.listen_thread.start()

        if self.sim_server_bool == "1":
            # Start the simulation server
            self.sim_server_thread = threading.Thread(
                target=sim_server, args=(self,), daemon=True
            )
            self.sim_server_thread.start()

    def setup_matrices(self):
        # Init transformation matrices
        self.w_T_c = np.eye(4)
        self.o_T_fs = np.eye(4)

        R_n = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        self.fsi_T_ci = np.eye(4)
        self.fsi_T_ci[:3, :3] = R_n

        self.o_T_w = self.setup_o_T_w()
        # print("[INFO] o_T_w: ", self.o_T_w)

    def setup_mode(self):
        self.setup_view()

        if self.results_mode:
            print("[INFO] Results visualization mode enabled.")
            self.view_mode = "tp"
            self.live_mode = False
            self.record_mode = False
            self.autopilot = False
            self.sim_server_bool = "0"

            self.setup_results()
            draw_results_trajectories(self)
        else:
            # Set up camera parameters
            if self.view_mode == "fp":
                self.setup_camera_params()

            self.robot_tip_visible = True

            if self.live_mode == False:
                self.setup_key_controls()
            else:
                self.trajectory_history_position = []
                self.trajectory_history_wTc = []

        if self.live_mode and self.autopilot:
            print(
                "[WARNING] Live mode and autopilot are mutually exclusive. Disabling autopilot."
            )
            self.autopilot = False

        if (
            self.depth_bool == "1"
            and self.record_mode == True
            and self.results_mode == False
        ):
            self.setup_depth()

        if self.autopilot:
            print("[INFO] Autopilot enabled")

        if self.live_mode == True:
            self.setup_live_mode()

        if self.record_mode == True:
            if self.legacy_record_method == True:
                self.setup_video_recorder()
            else:
                self.collect_sequence_dataset_init()
        else:
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

    def setup_points(self):
        import random

        self.selected_branch_names = None  # Track selected branch names for -random

        if hasattr(args, "random") and args.random is not None:
            # Randomly select N centerlines from the folder
            centerline_folder_name = self.app_config["PATHS"]["all_branches_folder"]
            centerline_folder_path = os.path.join(
                self.data_folder, centerline_folder_name
            )
            all_files = [
                f for f in os.listdir(centerline_folder_path) if f.endswith(".vtp")
            ]
            if len(all_files) < args.random:
                print(
                    f"[ERROR] Not enough centerlines to pick {args.random} random ones."
                )
                sys.exit(1)
            selected = random.sample(all_files, args.random)
            selected_rel = [os.path.join(centerline_folder_name, f) for f in selected]
            print(f"[INFO] Randomly selected centerlines: {selected_rel}")
            from src.utils import build_random_branches_path

            # Store just the branch names (e.g., b4, b9, b1) for output
            self.selected_branch_names = [
                os.path.splitext(os.path.basename(f))[0] for f in selected
            ]

            fs_frames, points = build_random_branches_path(
                selected_rel, self.data_folder
            )
            if not fs_frames:
                print("[ERROR] No branches found. Exiting...")
                sys.exit(1)
            self.setup_line_multibranch(fs_frames)
            print("[INFO] Random path built successfully")
            self.points = points
        elif self.all_branches_bool == "1":
            # Crate a trajectory traversing all the branches in the folder forward and backward
            print("[INFO] Building final path combining all branches...")
            fs_frames, points = build_all_branches_path(
                self.app_config, self.data_folder
            )

            if not fs_frames:
                print("[ERROR] No branches found. Exiting...")
                sys.exit(1)

            self.setup_line_multibranch(fs_frames)
            print("[INFO] Final path built successfully")
            self.points = points

        else:
            # Get centerline points from the .vtp
            points = get_vtp_line_points(self.path_path)

            # Setup
            self.setup_line(points)
            self.points = points

    def setup_results(self):

        self.draw_original_slam_bool = self.app_config["RESULTS"][
            "draw_original_slam_bool"
        ]
        self.draw_gt_bool = self.app_config["RESULTS"]["draw_gt_bool"]
        self.draw_centerline_bool = self.app_config["RESULTS"]["draw_centerline_bool"]
        self.draw_snapped_slam_bool = self.app_config["RESULTS"][
            "draw_snapped_slam_bool"
        ]
        self.filter_snapped_trajectory_bool = self.app_config["RESULTS"][
            "filter_snapped_trajectory_bool"
        ]
        self.filter_snapped_trajectory_sigma = float(
            self.app_config["RESULTS"]["filter_snapped_trajectory_sigma"]
        )

        res_centerline_fs_path_config = os.path.join(
            self.data_folder, self.app_config["RESULTS"]["res_centerline_fs"]
        )
        res_centerline_tum_path_config = os.path.join(
            self.data_folder, self.app_config["RESULTS"]["res_centerline_tum"]
        )
        res_gt_path_config = os.path.join(
            self.data_folder, self.app_config["RESULTS"]["res_gt"]
        )
        res_slam_path_config = os.path.join(
            self.data_folder, self.app_config["RESULTS"]["res_slam"]
        )
        temp_output_file = "trash.txt"  # Used by convert_tum_to_fs if write_file=True

        # Load TUM trajectories and convert to lists of 4x4 FS-like frames
        self.res_centerline_frames = convert_tum_to_fs(
            res_centerline_tum_path_config,
            temp_output_file,
            res_centerline_fs_path_config,
            write_file=False,
        )
        print(f"[INFO] Loaded {len(self.res_centerline_frames)} centerline frames.")

        res_gt_frames = convert_tum_to_fs(
            res_gt_path_config,
            temp_output_file,
            res_centerline_fs_path_config,
            write_file=False,
        )
        print(f"[INFO] Loaded {len(res_gt_frames)} GT frames.")

        print("[INFO] Loading and converting SLAM trajectory...")
        res_slam_frames = convert_tum_to_fs(
            res_slam_path_config,
            temp_output_file,
            res_centerline_fs_path_config,
            write_file=False,
        )
        print(f"[INFO] Loaded {len(res_slam_frames)} SLAM frames.")

        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)

        def resample_trajectory_points(points_array, num_target_points):
            num_original_points = len(points_array)
            if num_original_points == num_target_points:
                return points_array
            if num_original_points < 1 or num_target_points < 1:
                print("[WARNING] Cannot resample trajectory with less than 1 point.")
                return points_array  # Or return empty, or raise error
            if num_original_points == 1:  # Replicate the single point
                return np.tile(points_array, (num_target_points, 1))

            original_indices = np.linspace(
                0, num_original_points - 1, num_original_points
            )
            target_indices = np.linspace(0, num_original_points - 1, num_target_points)

            resampled_points = np.zeros((num_target_points, points_array.shape[1]))
            for i in range(points_array.shape[1]):  # For each dimension (e.g., x, y, z)
                resampled_points[:, i] = np.interp(
                    target_indices, original_indices, points_array[:, i]
                )
            return resampled_points

        def apply_sRt_to_frames(frames_list, s, R, t_vec):
            aligned_frames = []
            for frame in frames_list:
                pos = frame[:3, 3]
                rot = frame[:3, :3]
                aligned_pos = s * np.dot(R, pos) + t_vec
                aligned_rot = np.dot(R, rot)
                new_frame = np.eye(4)
                new_frame[:3, :3] = aligned_rot
                new_frame[:3, 3] = aligned_pos
                aligned_frames.append(new_frame)
            return aligned_frames

        # Align gt to centerline
        self.res_gt_aligned_frames = list(
            res_gt_frames
        )  # Default to original if alignment fails/skipped
        if self.res_centerline_frames and res_gt_frames:
            print("[INFO] Aligning trajectories...")
            centerline_pos_orig = np.array(
                [f[:3, 3] for f in self.res_centerline_frames]
            )
            gt_pos_orig = np.array([f[:3, 3] for f in res_gt_frames])

            len_c = len(centerline_pos_orig)
            len_g = len(gt_pos_orig)

            if len_c > 0 and len_g > 0:
                centerline_pos_align = centerline_pos_orig
                gt_pos_align = gt_pos_orig

                if len_c != len_g:
                    if len_c > len_g:  # Resample centerline to match GT length
                        centerline_pos_align = resample_trajectory_points(
                            centerline_pos_orig, len_g
                        )
                    else:  # Resample GT to match centerline length
                        gt_pos_align = resample_trajectory_points(gt_pos_orig, len_c)

                s_gt_c, R_gt_c, t_gt_c = align_umeyama(
                    model=centerline_pos_align,  # Target
                    data=gt_pos_align,  # Source to be aligned
                    known_scale=False,
                )

                # Apply transform to the original (full) GT trajectory frames
                self.res_gt_aligned_frames = apply_sRt_to_frames(
                    res_gt_frames, s_gt_c, R_gt_c, t_gt_c
                )
            else:
                print(
                    "[WARNING] Cannot align GT to centerline: one or both trajectories have zero length."
                )
        else:
            print(
                "[WARNING] Centerline or GT trajectory is empty. Skipping GT to Centerline alignment."
            )

        # Align slam to (aligned) gt
        self.res_slam_aligned_frames = list(res_slam_frames)  # Default to original
        if self.res_gt_aligned_frames and res_slam_frames:
            gt_aligned_pos_orig = np.array(
                [f[:3, 3] for f in self.res_gt_aligned_frames]
            )
            slam_pos_orig = np.array([f[:3, 3] for f in res_slam_frames])

            len_gt_aligned = len(gt_aligned_pos_orig)
            len_slam = len(slam_pos_orig)

            if len_gt_aligned > 0 and len_slam > 0:
                gt_aligned_pos_align = gt_aligned_pos_orig
                slam_pos_align = slam_pos_orig

                if len_gt_aligned != len_slam:
                    if (
                        len_gt_aligned > len_slam
                    ):  # Resample aligned GT to match SLAM length
                        gt_aligned_pos_align = resample_trajectory_points(
                            gt_aligned_pos_orig, len_slam
                        )
                    else:  # Resample SLAM to match aligned GT length
                        slam_pos_align = resample_trajectory_points(
                            slam_pos_orig, len_gt_aligned
                        )

                s_slam_gt, R_slam_gt, t_slam_gt = align_umeyama(
                    model=gt_aligned_pos_align,  # Target
                    data=slam_pos_align,  # Source to be aligned
                    known_scale=False,
                )
                # Apply transform to the original (full) SLAM trajectory frames
                self.res_slam_aligned_frames = apply_sRt_to_frames(
                    res_slam_frames, s_slam_gt, R_slam_gt, t_slam_gt
                )
            else:
                print(
                    "[WARNING] Cannot align SLAM to GT: one or both trajectories have zero length."
                )
        else:
            print(
                "[WARNING] Aligned GT or SLAM trajectory is empty. Skipping SLAM to GT alignment."
            )

        # Snap SLAM trajectory to be within the model
        self.res_slam_snapped_frames = []
        self.snap_trajectory_threshold_radius = 4  # mm # TEMPORARY SET HERE
        if self.res_slam_aligned_frames and self.res_centerline_frames:
            print(
                f"[INFO] Snapping SLAM trajectory with threshold radius: {self.snap_trajectory_threshold_radius}..."
            )
            self.res_slam_snapped_frames = trajectory_snapping(
                self.res_slam_aligned_frames,
                self.res_centerline_frames,
                self.snap_trajectory_threshold_radius,
            )
            # Filter the snapped trajectory if enabled
            if (
                self.filter_snapped_trajectory_bool == "1"
                and self.res_slam_snapped_frames
            ):
                self.res_slam_snapped_frames = filter_trajectory_positions(
                    self.res_slam_snapped_frames,
                    self.filter_snapped_trajectory_sigma,
                )
        else:
            print(
                "[WARNING] Cannot snap SLAM trajectory: SLAM or centerline trajectory is empty."
            )

        # Add legend for trajectories
        self.legend_nodes = []
        legend_start_pos = (-1.8, 0, 0.9)
        legend_offset = 0.07
        text_scale = 0.05

        # Define legend items: (text, color_tuple_rgba)
        # Colors should match those in draw_results_trajectories from draw.py
        legend_items = []
        if self.draw_centerline_bool == "1" and self.res_centerline_frames:
            legend_items.append(("Centerline (GT Path)", (0.9, 0.2, 0.2, 1)))  # Red
        if self.draw_gt_bool == "1" and self.res_gt_aligned_frames:
            legend_items.append(
                ("Ground Truth (Aligned to Centerline)", (0.2, 0.3, 0.9, 1))
            )  # Blue
        if self.draw_original_slam_bool == "1" and self.res_slam_aligned_frames:
            legend_items.append(
                ("SLAM Output (Aligned to GT)", (0, 0, 1, 1))
            )  # Original Blue
        if self.draw_snapped_slam_bool == "1" and self.res_slam_snapped_frames:
            legend_items.append(
                ("SLAM Snapped (to Centerline)", (0.1, 0.8, 0.1, 1))
            )  # Green

        for i, (text, color) in enumerate(legend_items):
            label = OnscreenText(
                text=text,
                pos=(legend_start_pos[0], legend_start_pos[2] - i * legend_offset),
                scale=text_scale,
                fg=color,
                align=TextNode.ALeft,  # type: ignore
                mayChange=False,
            )
            self.legend_nodes.append(label)

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

        if self.live_mode == False and self.view_mode == "tp":
            # Initially draw the path up to the first point
            draw_path(self, self.interpolated_points, 0)

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

    def setup_view(self):
        # Load the model
        if self.view_mode == "fp":
            self.setup_fp()

        elif self.view_mode == "tp":
            self.setup_tp()

        self.setup_breathing_pivot()

    ## UPDATE METHODS
    def update_breathing_effect(self, task):
        if (
            not self.breathing_enabled
            or self.breathing_pivot_node is None
            or self.scene is None
        ):
            # If no pivot, or scene removed, or disabled, stop or don't run the task logic
            return task.cont

        elapsed_time = globalClock.getFrameTime()  # type: ignore

        # Ensure period is not zero to avoid division by zero
        if self.breathing_period_seconds <= 0:
            return task.cont

        cycle_time = elapsed_time % self.breathing_period_seconds
        normalized_time = cycle_time / self.breathing_period_seconds  # 0 to 1

        # Sin wave from -1 to 1
        sin_value = math.sin(normalized_time * 2 * math.pi)

        # Map sin_value [-1, 1] to [min_scale, max_scale]
        # (sin_value + 1) / 2 maps to [0, 1]
        scale_factor = (sin_value + 1) / 2
        current_scale_value = self.breathing_min_scale + scale_factor * (
            self.breathing_max_scale - self.breathing_min_scale
        )

        self.breathing_pivot_node.setScale(current_scale_value)

        return task.cont

    def update_camera_to_robot_tip(self):
        # Use the current and next indices from the movement logic
        current_index = self.current_index
        next_index = self.next_index

        # Get the positions of the current and next points
        current_pos = self.interpolated_points[current_index]
        next_pos = self.interpolated_points[next_index]

        # Get the orientation vectors for the current and next points
        tangent_current = self.tangents[current_index]
        normal_current = self.normals[current_index]
        tangent_next = self.tangents[next_index]
        normal_next = self.normals[next_index]

        # Calculate the interpolation factor (alpha)
        segment_vec = next_pos - current_pos
        segment_len = np.linalg.norm(segment_vec)
        tip_vec = self.robot_tip - current_pos
        dist_on_segment = np.linalg.norm(tip_vec)

        if segment_len > 1e-5:
            alpha = dist_on_segment / segment_len
            alpha = np.clip(alpha, 0.0, 1.0)  # Clamp between 0 and 1
        else:
            alpha = 0.0

        # Interpolate the tangent and normal vectors
        tangent = tangent_current * (1 - alpha) + tangent_next * alpha
        normal = normal_current * (1 - alpha) + normal_next * alpha

        # Apply exponential smoothing to reduce jitter when indices change rapidly
        smoothing = getattr(self, "orientation_smoothing", 0.0)
        if smoothing > 0.0:
            if not hasattr(self, "smoothed_tangent") or self.smoothed_tangent is None:
                self.smoothed_tangent = tangent
            else:
                self.smoothed_tangent = (
                    (1 - smoothing) * self.smoothed_tangent + smoothing * tangent
                )

            if not hasattr(self, "smoothed_normal") or self.smoothed_normal is None:
                self.smoothed_normal = normal
            else:
                self.smoothed_normal = (
                    (1 - smoothing) * self.smoothed_normal + smoothing * normal
                )

            tangent = self.smoothed_tangent
            normal = self.smoothed_normal

        # Normalize and orthogonalize the interpolated vectors
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        # Ensure the normal vector remains orthogonal to the tangent
        normal = normal - tangent * np.dot(normal, tangent)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            fallback = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(fallback, tangent)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            normal = np.cross(tangent, np.cross(fallback, tangent))
            normal = normal / np.linalg.norm(normal)

        if smoothing > 0.0:
            self.smoothed_tangent = tangent
            self.smoothed_normal = normal

        # Set the camera position at the robot tip
        self.camera.setPos(LVector3f(*self.robot_tip))  # type: ignore

        # Calculate the focal point using the interpolated tangent vector
        focal_point = self.robot_tip + tangent

        # Set the camera to look at the focal point with the interpolated normal as the up vector
        self.camera.lookAt(LVector3f(*focal_point), LVector3f(*-normal))  # type: ignore

        # Update the directional light's orientation to match the camera's orientation
        # if in first-person view mode
        # if self.view_mode == "fp":
        #     cameraHpr = self.camera.getHpr()
        #     self.directionalLightNP.setHpr(cameraHpr)

        # Adjust lighting to follow the camera
        # self.directionalLightNP.setPos(
        #     self.camera.getX(), self.camera.getY(), self.camera.getZ()
        # )

    def update_robot_tip_position(self, dt, forward=True):

        if self.live_mode == False:
            # Define the speed of movement along the line
            movement_speed = 1  # Adjust as needed

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
            closest_index = int(np.argmin(distances))

            # Keep shared traversal indices in sync for camera/orientation logic with monotonic updates
            if not hasattr(self, "current_index"):
                self.current_index = closest_index

            if forward:
                if closest_index >= self.current_index:
                    self.current_index = closest_index
            else:
                if closest_index <= self.current_index:
                    self.current_index = closest_index

            self.current_index = int(
                np.clip(self.current_index, 0, len(self.interpolated_points) - 1)
            )
            if forward:
                self.next_index = min(
                    self.current_index + 1, len(self.interpolated_points) - 1
                )
            else:
                self.next_index = max(self.current_index - 1, 0)

            if self.view_mode == "tp" and self.live_mode == False:
                # Redraw the path up to the robot tip
                draw_path(self, self.interpolated_points, closest_index)

            # Get the current point on the centerline using the closest_index
            current_point = self.interpolated_points[closest_index]

            # Compute the curvilinear abscissa
            self.current_ca = curvilinear_abscissa(
                current_point,
                self.interpolated_points,
                self.all_branches_bool,
                self.record_mode,
            )

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

                    # Store position in trajectory history (if it's a new position)
                    if len(self.trajectory_history_position) == 0 or not np.array_equal(
                        self.trajectory_history_position[-1], self.robot_tip
                    ):
                        self.trajectory_history_position.append(np.copy(self.robot_tip))

                    # Store wTc in trajectory history
                    if len(self.trajectory_history_wTc) == 0 or not np.array_equal(
                        self.trajectory_history_wTc[-1], w_T_c_matrix
                    ):
                        self.trajectory_history_wTc.append(np.copy(w_T_c_matrix))

                    # print(f"Robot tip position: {self.robot_tip}")

                except (SyntaxError, AttributeError) as e:
                    print(f"Error in update_robot_tip_position: {e}")
                    pass

        # Update the visual representation
        draw_robot_tip(self)

        if self.live_mode and self.view_mode == "tp":
            self.update_trajectory()

    def update_key_map(self, controlName, controlState):
        self.keyMap[controlName] = controlState

        if controlName == "robot_tip_forward":
            if self.record_mode == False:
                if controlState:
                    highlight_arrow(self, "up")
                else:
                    unhighlight_arrow(self, "up")
        elif controlName == "robot_tip_backward":
            if self.record_mode == False:
                if controlState:
                    highlight_arrow(self, "down")
                else:
                    unhighlight_arrow(self, "down")

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
                        self.update_tip_position_all_branches(dt, forward=False)
                    else:
                        self.update_robot_tip_position(dt, forward=False)
        else:
            self.update_robot_tip_position(dt)

        # Update the camera position and orientation
        if self.view_mode == "fp":
            self.update_camera_to_robot_tip()

            if self.draw_centerline_bool == "1":
                # Update the trajectory
                self.update_trajectory()
            
            # Update shader light position (attached to camera)
            if hasattr(self, "scene"):
                self.scene.setShaderInput("lightPos", self.camera.getPos(self.render))

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
            if self.legacy_record_method == True:
                self.record_frame()
            else:
                self.collect_sequence_dataset_step()

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
            # movement_speed = 5000  # units per second
            movement_speed = 0.5
            step = movement_speed * dt

            # Calculate the distance of the robot tip from the start of the current segment
            dist_from_start = np.linalg.norm(self.robot_tip - current_pos)
            new_dist_on_segment = dist_from_start + step

            # Don't overshoot the next point.
            if new_dist_on_segment > dist:
                new_dist_on_segment = dist

            # Calculate new position from the start of the segment
            new_pos = current_pos + direction * new_dist_on_segment
            self.robot_tip = new_pos

            # When we've nearly reached the next point, snap to it and reset interpolation.
            if dist - new_dist_on_segment < 1e-3:
                self.robot_tip = next_pos
                self.interp_alpha = 0.0  # reset orientation interpolation
                self.current_index = next_index
        else:
            print(
                "[INFO] Translation difference is near zero. Update only tip orientation."
            )
            # Orientation interpolation via SLERP
            R_current = get_rotation_from_index(
                self.current_index, self.tangents, self.normals, self.binormals
            )
            R_next = get_rotation_from_index(
                next_index, self.tangents, self.normals, self.binormals
            )

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

        if self.view_mode == "tp" and self.live_mode == False:
            # Redraw the path up to the robot tip
            draw_path(self, self.interpolated_points, self.current_index)

        # Compute the curvilinear abscissa
        current_point = self.interpolated_points[self.current_index]
        self.current_ca = curvilinear_abscissa(
            current_point,
            self.interpolated_points,
            self.all_branches_bool,
            self.record_mode,
        )

        # Update next index based on the new current_index.
        if forward:
            self.next_index = min(
                self.current_index + 1, len(self.interpolated_points) - 1
            )
        else:
            self.next_index = max(self.current_index - 1, 0)

    def update_trajectory(self):
        # Draw the trajectory from the current robot tip position
        draw_trajectory(self)

    # RECORD METHODS
    def collect_sequence_dataset_init(self):
        """
        Initializes the recording process for the Deep-Lung-ST dataset format.
        Creates the directory structure:
        /record_root/
          ├── video.npy            (Saved incrementally or at end)
          └── trajectory.npy       (Saved incrementally or at end)
        """
        if not self.record_mode:
            return

        # Generate a unique sequence name based on timestamp
        seq_name = f"seq_{int(time.time())}"
        self.dataset_seq_dir = os.path.join(self.data_folder, self.videos_dir, seq_name)
        
        # If output dir exists, append random suffix to avoid overwrite
        if os.path.exists(self.dataset_seq_dir):
            seq_name = f"seq_{int(time.time())}_{np.random.randint(0,1000)}"
            self.dataset_seq_dir = os.path.join(self.data_folder, self.videos_dir, seq_name)
            
        os.makedirs(self.dataset_seq_dir, exist_ok=True)
        print(f"[DATASET] Starting collection for sequence: {seq_name}")
        print(f"[DATASET] Saving to: {self.dataset_seq_dir}")

        # In-memory buffers for the sequence data
        # We collect all data in RAM and dump to .npy at the end to ensure shape consistency
        # For extremely long sequences, you might need to use memory mapping or append mode,
        # but for typical recording sessions (few minutes), RAM is fine.
        self.dataset_buffer_rgb = []
        self.dataset_buffer_pose = [] # [x, y, z, qx, qy, qz, qw]
        self.dataset_frame_count = 0

    def collect_sequence_dataset_step(self):
        """
        Captures the current frame and robot pose.
        Must be called every update loop if record_mode is True.
        """
        if not self.record_mode or not hasattr(self, 'dataset_buffer_rgb'):
            return

        # 1. Capture RGB Frame
        screenshot = self.win.getScreenshot()
        rgb_data = screenshot.getRamImage()
        if rgb_data is None:
            return
            
        # Convert raw buffer to numpy
        rgb = np.frombuffer(rgb_data, np.uint8)
        rgb.shape = (screenshot.getYSize(), screenshot.getXSize(), screenshot.getNumComponents())
        rgb = np.flipud(rgb) # Panda3D texture is upside down
        
        # Remove Alpha if present (Keep only RGB)
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
            
        # Resize to 128x128 if needed (Model Expectation), or keep native and resize in Dataloader.
        # Keeping native is safer for now, but ensure config matches model.
        # If you want to force 128x128 here:
        # rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        # Append to buffer (H, W, C)
        # We will transpose to (N, C, H, W) or (N, H, W, C) at save time.
        self.dataset_buffer_rgb.append(rgb)

        # 2. Capture Robot Pose (Ground Truth)
        # We need World_T_Camera (Camera Pose in World Frame)
        # In Panda3D, camera.getPos() returns position in Parent frame (usually World/Render)
        # camera.getQuat() returns rotation quaternion
        
        # Position (mm)
        pos = self.camera.getPos(self.render)
        x, y, z = pos.getX(), pos.getY(), pos.getZ()
        
        # Rotation (Quaternion: x, y, z, w)
        # Panda3D Quat is (r, i, j, k) -> (w, x, y, z)
        quat = self.camera.getQuat(self.render)
        qw, qx, qy, qz = quat.getR(), quat.getI(), quat.getJ(), quat.getK()
        
        # Store as [x, y, z, qx, qy, qz, qw] (Standard SciPy format)
        pose_data = np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)
        self.dataset_buffer_pose.append(pose_data)
        
        self.dataset_frame_count += 1
        
    def collect_sequence_dataset_finalize(self):
        """
        Saves the buffered data to .npy files and cleans up.
        """
        if not self.record_mode or not hasattr(self, 'dataset_seq_dir'):
            return

        print(f"\n[DATASET] Finalizing sequence... ({self.dataset_frame_count} frames)")
        
        if self.dataset_frame_count == 0:
            print("[DATASET] Warning: No frames collected. Skipping save.")
            shutil.rmtree(self.dataset_seq_dir) # Cleanup empty folder
            return

        # 1. Save Video (.npy)
        # Format: (N, 3, H, W) is PyTorch standard, but (N, H, W, 3) is standard for numpy/cv2
        # The Dataset class you have handles (N, H, W, 3) -> (N, 3, H, W).
        # So we save as (N, H, W, 3) uint8 for easy inspection and compatibility.
        video_array = np.stack(self.dataset_buffer_rgb, axis=0) # (N, H, W, 3)
        video_path = os.path.join(self.dataset_seq_dir, "video.npy")
        np.save(video_path, video_array)
        print(f"[DATASET] Saved video.npy shape={video_array.shape}")

        # 2. Save Trajectory (.npy)
        traj_array = np.stack(self.dataset_buffer_pose, axis=0) # (N, 7)
        traj_path = os.path.join(self.dataset_seq_dir, "trajectory.npy")
        np.save(traj_path, traj_array)
        print(f"[DATASET] Saved trajectory.npy shape={traj_array.shape}")
        
        # 3. Save Meta-data (Optional but useful)
        # Store which branches were used if random
        if hasattr(self, "selected_branch_names") and self.selected_branch_names:
            with open(os.path.join(self.dataset_seq_dir, "branches.txt"), "w") as f:
                f.write(",".join(self.selected_branch_names))

        # Clear buffers to free RAM
        self.dataset_buffer_rgb = []
        self.dataset_buffer_pose = []
        print("[DATASET] Collection complete.")

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
            depth = get_depth_image(self.depthTex)  # normalized, from 0 to 1
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

        # Clean up legend if it exists
        if hasattr(self, "legend_nodes"):
            for node in self.legend_nodes:
                node.destroy()
            self.legend_nodes = []

        # Save record
        if self.legacy_record_method:
            if self.record_mode and hasattr(self, "record_dir"):
                # If random branches were used, write their names to a file in the output folder
                if hasattr(self, "selected_branch_names") and self.selected_branch_names:
                    branches_txt = os.path.join(self.record_dir, "selected_branches.txt")
                    with open(branches_txt, "w") as f:
                        f.write(", ".join(self.selected_branch_names) + "\n")
                    print(f"[INFO] Written selected branches to {branches_txt}")
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
                # TODO: save trajectory in case of all branches used
                if self.all_branches_bool == "1":
                    fs_trajectory = save_fs_frames_multibranch(
                        self.data_folder,
                        self.interpolated_points,
                        self.tangents,
                        self.normals,
                        self.binormals,
                    )
                else:
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
        else:
            self.collect_sequence_dataset_finalize()
        # Save trajectory in TUM format
        if (
            self.live_mode
            and hasattr(self, "trajectory_history_wTc")
            and hasattr(self, "logs_dir")
        ):
            # Ensure logs_dir exists
            logs_dir_path = os.path.join(self.data_folder, self.logs_dir)
            os.makedirs(logs_dir_path, exist_ok=True)

            # Define filename for the live trajectory
            live_trajectory_filename = "live_trajectory_wTc.txt"
            live_trajectory_filepath = os.path.join(
                logs_dir_path, live_trajectory_filename
            )

            if self.trajectory_history_wTc:
                with open(live_trajectory_filepath, "w") as f:
                    for idx, wTc_matrix in enumerate(self.trajectory_history_wTc):
                        # Using index as a simple timestamp.
                        # For more accurate timing, timestamps should be recorded when poses are received.
                        timestamp = float(idx)

                        # Translation
                        t = wTc_matrix[:3, 3]
                        tx, ty, tz = t[0], t[1], t[2]

                        # Rotation to Quaternion
                        R_matrix = wTc_matrix[:3, :3]
                        rotation = Rotation.from_matrix(R_matrix)
                        q = rotation.as_quat()  # Returns [qx, qy, qz, qw]
                        qx, qy, qz, qw = q[0], q[1], q[2], q[3]

                        # Write in TUM format: timestamp tx ty tz qx qy qz qw
                        f.write(
                            f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                        )

                print(
                    f"[INFO] Live trajectory (wTc) saved to {live_trajectory_filepath}"
                )
            else:
                print("[INFO] No live trajectory data to save.")

        self.userExit()
