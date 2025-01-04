import argparse
import threading
import time
from direct.showbase.ShowBase import ShowBase  # type: ignore
from panda3d.core import *  # type: ignore
from direct.task import Task  # type: ignore
from direct.gui.DirectGui import DirectLabel  # type: ignore
import pyvista as pv  # type: ignore
from set_FS_frame import (
    interpolate_line,
    compute_tangent_vectors,
    compute_normal_vectors,
    compute_binormal_vectors,
    compute_MRF,
)
import numpy as np  # type: ignore
import socket

data_folder = "data/mesh/easier_slam_test/"
# path_name = "path.vtp"
# path_name = "centerline_b1.vtp"
path_name = "centerline_b2.vtp"
# path_name = "centerline_b3.vtp"
# path_name = "centerline_b4.vtp"
# path_name = "centerline_b5.vtp"
# negative_model_name = "easier_slam_test_negative.obj"
negative_model_name = "easier_slam_test_moved_negative.obj"
# model_name = "easier_slam_test.obj"
model_name = "easier_slam_test_moved.obj"
centerline_frames = "centerline_frames.txt"

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
args = parser.parse_args()

# Configuration settings
loadPrcFileData("", "win-size 800 600")  # type: ignore
loadPrcFileData("", "window-title Panda3D Full Camera Control")  # type: ignore
loadPrcFileData("", "load-file-type p3assimp")  # type: ignore


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Define path of the .vtp file
        self.path = data_folder + path_name

        # Check the view mode
        self.view_mode = args.view

        # Check the live mode
        self.live_mode = args.live
        print(f"View mode: {self.view_mode}\n Live mode: {self.live_mode}")

        # Init transformation matrices
        self.Tcw = np.eye(4)
        self.w_T_o = self.setup_w_T_o()

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
        self.green_point_visible = True  # Initial visibility status

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
        print("Number of points: ", len(points))

        # Temporarily draw the horizontal line
        # points = [(0, 0, 3), (1, 0, 3), (2, 0, 3), (3, 0, 3), (4, 0, 3), (5, 0, 3), (6, 0, 3), (7, 0, 3), (8, 0, 3), (9, 0, 3), (10, 0, 3)]

        # Setup
        self.setup_line(points)
        self.draw_elements(points)
        self.points = points

        # Load the model
        if self.view_mode == "fp":

            # Load the negative model to visualize the internal part
            """
            To create the negative solid using FreeCAD:
            - oper the part menu
            - create a sphere that can contain the model
            - import the .obj file of the model
            - create shape from mesh for the imported model
            - make solid from the created shape
            - usel boolean difference between the sphere and the solid
            - export the boolean cut as .obj
            """
            # self.model = (
            #     "data/mesh/vascularmodel/0063_H_PULMGLN_SVD/sim/0063_negative.obj"
            # )
            self.model = data_folder + negative_model_name

            print("initializing First Person View Mode...")
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
            print("Initializinig Third Person View Mode...")

            # Load the standard model to visualize the external part
            # self.model = "data/mesh/vascularmodel/0063_H_PULMGLN_SVD/sim/0063.obj"
            self.model = data_folder + model_name

            # Load the phantom model
            self.scene = self.loader.loadModel(self.model)
            self.scene.reparentTo(self.render)

            # Set transparency level (0.5 for 50% transparency) to see the green point mmoving inside
            self.scene.setTransparency(TransparencyAttrib.MDual)  # type: ignore
            self.scene.setColorScale(1, 1, 1, 0.5)

            # Initially draw the path up to the first point
            self.draw_path(self.interpolated_points, 0)

        print("Initialization done")

        if self.live_mode == True:

            # Start the server in a thread
            self.listen_thread = threading.Thread(target=self.start_server, daemon=True)
            self.listen_thread.start()

            # Start the simulation server
            self.sim_server_thread = threading.Thread(
                target=self.sim_server, daemon=True
            )
            self.sim_server_thread.start()

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
                        # print(f"Received: {data.decode()}")
                        # Parse the received data and update the transformation matrix
                        self.Tcw = data.decode()
                    print("Connection closed")

    def sim_server(self, host="127.0.0.1", port=12345):
        time.sleep(1)  # Give the server time to start
        try:
            # Create a socket and connect to the server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))

            while True:
                # Create a mock transformation matrix
                mock_Tcw = np.eye(4)
                # Add some random noise to the translation
                mock_Tcw[:3, 3] = np.random.normal(0, 1, 3)
                # Convert to string and send
                data = str(mock_Tcw.tolist())
                s.sendall(data.encode())
                time.sleep(0.1)  # Add small delay between sends

        except ConnectionRefusedError:
            print("Could not connect to server. Is it running?")
        except Exception as e:
            print(f"Error in sim_server: {e}")
        finally:
            s.close()

    def draw_elements(self, points):
        # Draw some frames
        # self.draw_FS_frames(points, num_points=10, draw_tangent=True, draw_normal=True, draw_binormal=True)

        # Drawing circles
        # self.draw_circles_around_points(radius=0.1, num_segments=12)

        # Draw the green point
        self.draw_green_point()

    def setup_key_controls(self):
        self.keyMap = {"green_point_forward": False, "green_point_backward": False}

        # Bind arrow keys for moving the green point
        self.accept("arrow_up", self.update_key_map, ["green_point_forward", True])
        self.accept("arrow_up-up", self.update_key_map, ["green_point_forward", False])
        self.accept("arrow_down", self.update_key_map, ["green_point_backward", True])
        self.accept(
            "arrow_down-up", self.update_key_map, ["green_point_backward", False]
        )

    def setup_line(self, points):
        # Load the .vtp file and interpolate the line
        self.interpolated_points = interpolate_line(points, num_points=100)
        self.tangents = compute_tangent_vectors(self.interpolated_points)

        # Compute normal and binormal vectors in the standard way
        # self.normals = compute_normal_vectors(self.tangents)
        # self.binormals = compute_binormal_vectors(self.tangents, self.normals)

        # Compute the Frenet-Serret frame using the MRF algorithm
        self.normals, self.binormals = compute_MRF(self.tangents)

        # Set first and end point
        self.start_point = self.interpolated_points[0]
        self.end_point = self.interpolated_points[-1]

        # Initialize the green point
        self.green_point = self.interpolated_points[
            0
        ]  # Setting the first point as the start
        self.green_point_node = None

        # Compute line length
        self.line_length = self.curvilinear_abscissa(self.end_point)
        print("Line length: ", self.line_length)

    def setup_w_T_o(self):
        """Load the transformation matrix from centerline_frames.txt"""
        w_T_o = np.eye(4)

        # Read the first line from centerline_frames.txt
        with open(data_folder + centerline_frames, "r") as f:
            # Extract data from the first line
            first_line = f.readline().strip()
            data = first_line.split(",")
            point = np.array([float(x) for x in data[0:3]])
            tangent = np.array([float(x) for x in data[3:6]])
            normal = np.array([float(x) for x in data[6:9]])
            binormal = np.array([float(x) for x in data[9:12]])

            # Create the transformation matrix
            w_T_o[:3, 0] = tangent
            w_T_o[:3, 1] = normal
            w_T_o[:3, 2] = binormal
            w_T_o[:3, 3] = point

        print(f"Transformation matrix w_T_o: \n{w_T_o}")

        return w_T_o

    ## LINE UTILS
    def curvilinear_abscissa(self, point):
        # Compute the from the start point to the current point
        return np.linalg.norm(point - self.start_point)

    def get_vtp_line_points(self):
        # Load the .vtp file
        line_model = pv.read(self.path)

        # Convert the points to a list of tuples
        points = [tuple(point) for point in line_model.points]

        # Return the points so as to be able to use them in create_line
        return points

    ## UPDATE METHODS
    def update_camera_to_green_point(self):
        # Find the index of the closest point to the green point
        distances = np.linalg.norm(self.interpolated_points - self.green_point, axis=1)
        closest_index = np.argmin(distances)

        # Get the corresponding tangent, normal, and binormal vectors
        tangent = LVector3f(*self.tangents[closest_index])  # type: ignore
        normal = LVector3f(*self.normals[closest_index])  # type: ignore
        binormal = LVector3f(*self.binormals[closest_index])  # type: ignore

        # Set the camera position at the green point
        self.camera.setPos(LVector3f(*self.green_point))  # type: ignore

        # Calculate the focal point using the tangent vector
        focal_point = self.green_point + tangent

        # Set the camera to look at the focal point with the binormal as the up vector
        self.camera.lookAt(LVector3f(*focal_point), normal)  # type: ignore

        # Update the directional light's orientation to match the camera's orientation
        # if in first-person view mode
        if self.view_mode == "fp":
            cameraHpr = self.camera.getHpr()
            self.directionalLightNP.setHpr(cameraHpr)

        # Adjust lighting to follow the camera
        self.directionalLightNP.setPos(
            self.camera.getX(), self.camera.getY(), self.camera.getZ()
        )

    def update_green_point_position(self, dt, forward=True):
        # Define the speed of movement along the line
        movement_speed = 5  # Adjust as needed

        # Calculate distances from self.green_point to each point in self.interpolated_points
        distances = np.linalg.norm(self.interpolated_points - self.green_point, axis=1)
        current_index = np.argmin(distances)

        if forward:
            # Check if the green point is at the last point
            if current_index >= len(self.interpolated_points) - 1:
                return  # Stop moving forward
            next_index = current_index + 1
        else:
            # Check if the green point is at the first point
            if current_index == 0:
                return  # Stop moving backward
            next_index = current_index - 1

        # Calculate the direction and distance to the next point
        direction = (
            self.interpolated_points[next_index]
            - self.interpolated_points[current_index]
        )
        distance_to_next_point = np.linalg.norm(direction)
        direction = direction / distance_to_next_point  # Normalize the direction vector

        # Calculate the movement step
        step_size = movement_speed * dt
        if step_size > distance_to_next_point:
            step_size = (
                distance_to_next_point  # Limit step to not overshoot the next point
            )

        # Update the position
        new_position = self.green_point + direction * step_size
        self.green_point = new_position

        # Find the index of the closest point to the green point
        distances = np.linalg.norm(self.interpolated_points - self.green_point, axis=1)
        closest_index = np.argmin(distances)

        if self.view_mode == "tp":
            # Redraw the path up to the green point
            self.draw_path(self.interpolated_points, closest_index)

        # Update the visual representation
        self.draw_green_point()

    def update_key_map(self, controlName, controlState):
        self.keyMap[controlName] = controlState

        if controlName == "green_point_forward":
            if controlState:
                self.highlight_arrow("up")
            else:
                self.unhighlight_arrow("up")
        elif controlName == "green_point_backward":
            if controlState:
                self.highlight_arrow("down")
            else:
                self.unhighlight_arrow("down")

    def update_scene(self, task):
        dt = globalClock.getDt()  # type: ignore

        # Update the green point position
        if self.live_mode == False:
            if self.keyMap["green_point_forward"]:
                self.update_green_point_position(dt, forward=True)
            if self.keyMap["green_point_backward"]:
                self.update_green_point_position(dt, forward=False)
        else:
            # Update the green point position based on the transformation matrix
            try:
                # Convert string to numpy array and multiply matrices
                Tcw_matrix = np.array(eval(self.Tcw))
                translation = Tcw_matrix[:3, 3] + 1000 * self.w_T_o[:3, 3]

                # Update the green point position
                self.green_point = translation
                # print(f"Green point position: {self.green_point}")
                # Update the visual representation
                self.draw_green_point()
            except (SyntaxError, AttributeError) as e:
                # Skip update if data is not valid
                pass

        # Update the camera position and orientation
        if self.view_mode == "fp":
            self.update_camera_to_green_point()

            # Update the trajectory
            self.update_trajectory()

        # Blinking logic
        self.blink_timer += dt
        if self.blink_timer >= self.blink_interval:
            self.blink_timer = 0  # Reset timer
            self.green_point_visible = not self.green_point_visible  # Toggle visibility
            if self.green_point_node:
                self.green_point_node.setTransparency(
                    TransparencyAttrib.MDual  # type: ignore
                )  # Enable transparency
                self.green_point_node.setAlphaScale(
                    1 if self.green_point_visible else 0.5
                )  # Set visibility

        return Task.cont

    def update_trajectory(self):
        # Draw the trajectory from the current green point position
        self.draw_trajectory()

    ## DRAW METHODS
    def draw_circles_around_points(self, radius=1, num_segments=12):
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
        num_points=10,
        draw_tangent=True,
        draw_normal=True,
        draw_binormal=True,
    ):

        # Interpolate the line for smoothing
        interpolated_points = interpolate_line(points)

        # Ensure num_points is less than the length of interpolated_points
        num_points = min(num_points, len(interpolated_points))

        # Calculate step, ensuring it's not zero
        step = max(1, len(interpolated_points) // num_points)

        # Sample points along the line for drawing frames
        sampled_points = interpolated_points[::step]

        # Compute tangent vectors for the interpolated points
        tangents = compute_tangent_vectors(interpolated_points)

        # Compute normal and binormal vectors in the standard way
        # normals = compute_normal_vectors(tangents)
        # binormals = compute_binormal_vectors(tangents, normals)

        # Compute the Frenet-Serret frame using the MRF algorithm
        normals, binormals = compute_MRF(tangents)

        # Replace zero norms with 1 to prevent division by zero
        norms = np.linalg.norm(binormals, axis=1)
        norms[norms == 0] = 1
        binormals = binormals / norms[:, np.newaxis]

        # Sample points along the line for drawing frames
        step = len(interpolated_points) // num_points
        sampled_points = interpolated_points[::step]

        # Draw the frames
        for i, point in enumerate(sampled_points):
            if draw_tangent:
                self.draw_vector(point, tangents[i], (1, 0, 0, 1))  # Red for tangent
            if draw_normal:
                self.draw_vector(point, normals[i], (0, 1, 0, 1))  # Green for normal
            if draw_binormal:
                self.draw_vector(point, binormals[i], (0, 0, 1, 1))  # Blue for binormal

    def draw_green_point(self):
        if self.green_point_node:
            self.green_point_node.removeNode()  # Remove the old node if it exists

        # Create the green point visual
        green_point_visual = self.loader.loadModel(
            "models/smiley"
        )  # Ensure this is a valid model path
        green_point_visual.setScale(3)  # Scale to appropriate size
        green_point_visual.setColor(0, 1, 0, 1)  # Set color to green
        green_point_visual.setPos(LVector3f(*self.green_point))  # type: ignore

        # Create a new node and parent the visual to it
        green_point_node = self.render.attachNewNode("GreenPointNode")
        green_point_visual.reparentTo(green_point_node)
        self.green_point_node = green_point_node

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
        points = interpolate_line(points, num_points=50)

        # Create the line
        line = LineSegs()  # type: ignore
        line.setThickness(5.0)
        line.setColor(
            8 / 255, 232 / 255, 222 / 255, 1
        )  # Same color as the arrow button

        # Start drawing the line from the green point
        green_point = self.green_point
        first_point = LVector3f(green_point[0], green_point[1], green_point[2])  # type: ignore
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


if __name__ == "__main__":
    app = MyApp()
    app.run()
