#!/usr/bin/env python3

import os
import sys
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Geom,
    GeomNode,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    TransparencyAttrib,
    LineSegs,
    AmbientLight,
    DirectionalLight,
    PointLight,
    LVector3f,
)


class SimplePointCloudViewer(ShowBase):
    def __init__(self, ply_file_path):
        ShowBase.__init__(self)

        # Window setup
        self.setBackgroundColor(0, 0.1, 0.2)  # Dark blue background

        # Set up camera
        self.cam.setPos(0, -20, 5)
        self.cam.lookAt(0, 0, 0)

        # Setup camera controls similar to navigation.py
        self.setupControls()

        # Setup lighting
        self.setupLighting()

        # Create coordinate axes
        self.createAxes()

        # Load and display the point cloud
        self.loadPointCloud(ply_file_path)

        # Show file info
        print(f"Loaded point cloud: {os.path.basename(ply_file_path)}")
        print(f"Points: {self.num_points}")

    def setupControls(self):
        # Disable default mouse control
        self.disableMouse()

        # Camera control variables
        self.keys = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            "q": False,
            "e": False,
            "shift": False,
        }

        # Mouse variables
        self.mouse_looking = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_speed = 0.5
        self.keyboard_speed = 0.5

        # Set keyboard events
        for key in self.keys:
            self.accept(key, self.setKey, [key, True])
            self.accept(key + "-up", self.setKey, [key, False])

        # Mouse wheel for zoom
        self.accept("wheel_up", self.zoom, [True])
        self.accept("wheel_down", self.zoom, [False])

        # Mouse look on right-click (like in navigation.py)
        self.accept("mouse3", self.startMouseLook)  # Right mouse button
        self.accept("mouse3-up", self.stopMouseLook)

        # Shift for speed boost
        self.accept("shift", self.setKey, ["shift", True])
        self.accept("shift-up", self.setKey, ["shift", False])

        # Exit on escape
        self.accept("escape", sys.exit)

        # Add camera movement task
        self.taskMgr.add(self.moveCamera, "MoveCameraTask")

    def setKey(self, key, value):
        self.keys[key] = value

    def zoom(self, in_direction):
        # Move camera forward/backward for zoom
        current_pos = self.cam.getPos()
        direction = self.cam.getQuat().getForward()
        zoom_amount = 1.0 if in_direction else -1.0
        self.cam.setPos(current_pos + direction * zoom_amount)

    def startMouseLook(self):
        self.mouse_looking = True
        self.last_mouse_x = self.mouseWatcherNode.getMouseX()
        self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def stopMouseLook(self):
        self.mouse_looking = False

    def moveCamera(self, task):
        dt = globalClock.getDt()

        # Mouse look
        if self.mouse_looking and self.mouseWatcherNode.hasMouse():
            mouse_x = self.mouseWatcherNode.getMouseX()
            mouse_y = self.mouseWatcherNode.getMouseY()

            dx = mouse_x - self.last_mouse_x
            dy = mouse_y - self.last_mouse_y

            # Update camera heading and pitch
            heading = self.cam.getH() - dx * 50 * self.mouse_speed
            pitch = self.cam.getP() - dy * 50 * self.mouse_speed

            # Clamp pitch to prevent flipping
            pitch = max(-89, min(89, pitch))

            self.cam.setH(heading)
            self.cam.setP(pitch)

            self.last_mouse_x = mouse_x
            self.last_mouse_y = mouse_y

        # Keyboard movement
        speed = self.keyboard_speed * (5 if self.keys["shift"] else 1)

        # Move forward/backward
        if self.keys["w"]:
            self.cam.setPos(self.cam, 0, speed * dt, 0)
        if self.keys["s"]:
            self.cam.setPos(self.cam, 0, -speed * dt, 0)

        # Move left/right
        if self.keys["a"]:
            self.cam.setPos(self.cam, -speed * dt, 0, 0)
        if self.keys["d"]:
            self.cam.setPos(self.cam, speed * dt, 0, 0)

        # Move up/down
        if self.keys["e"]:
            self.cam.setPos(self.cam, 0, 0, speed * dt)
        if self.keys["q"]:
            self.cam.setPos(self.cam, 0, 0, -speed * dt)

        return task.cont

    def setupLighting(self):
        # Add ambient light
        ambient_light = AmbientLight("ambientLight")
        ambient_light.setColor((0.4, 0.4, 0.4, 1))
        ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_light_np)

        # Add directional light
        directional_light = DirectionalLight("directionalLight")
        directional_light.setColor((0.8, 0.8, 0.8, 1))
        directional_light_np = self.render.attachNewNode(directional_light)
        directional_light_np.setHpr(45, -45, 0)
        self.render.setLight(directional_light_np)

        # Add a point light attached to the camera
        point_light = PointLight("pointLight")
        point_light.setColor((0.6, 0.6, 0.6, 1))
        point_light.setAttenuation((1, 0, 0.001))
        self.point_light_np = self.cam.attachNewNode(point_light)
        self.render.setLight(self.point_light_np)

    def createAxes(self):
        # Create coordinate axes (x=red, y=green, z=blue)
        lines = LineSegs()

        # X axis - Red
        lines.setColor(1, 0, 0, 1)
        lines.setThickness(2)
        lines.moveTo(0, 0, 0)
        lines.drawTo(5, 0, 0)

        # Y axis - Green
        lines.setColor(0, 1, 0, 1)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 5, 0)

        # Z axis - Blue
        lines.setColor(0, 0, 1, 1)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 0, 5)

        node = lines.create()
        self.render.attachNewNode(node)

    def loadPointCloud(self, ply_file_path):
        # Initialize a counter for the points
        self.num_points = 0

        try:
            # Read the PLY file
            points = []
            header_done = False
            num_vertices = 0

            with open(ply_file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if not header_done:
                        if line.startswith("element vertex"):
                            num_vertices = int(line.split()[2])
                        if line == "end_header":
                            header_done = True
                    else:
                        # Parse vertex data
                        values = line.split()
                        if len(values) >= 3:  # Ensure at least X, Y, Z
                            x, y, z = map(float, values[:3])
                            points.append((x, y, z))
                            self.num_points += 1

                        if self.num_points >= num_vertices:
                            break

            # Convert points to numpy array
            points = np.array(points)

            # Create Panda3D geometry for the point cloud
            self.createPointCloudGeometry(points)

        except Exception as e:
            print(f"Error loading point cloud: {e}")

    def createPointCloudGeometry(self, points):
        # Create the point cloud geometry
        vdata = GeomVertexData("points", GeomVertexFormat.getV3c4(), Geom.UHStatic)
        vdata.setNumRows(len(points))

        # Create writers for vertex and color
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")

        # Add each point with a color based on its position
        for point in points:
            vertex.addData3f(point[0], point[1], point[2])

            # Simple color scheme based on position
            # Normalize coords to 0-1 range for coloring
            max_coord = max(abs(np.max(points)), abs(np.min(points)))
            r = min(1.0, abs(point[0] / max_coord * 0.8 + 0.2))
            g = min(1.0, abs(point[1] / max_coord * 0.8 + 0.2))
            b = min(1.0, abs(point[2] / max_coord * 0.8 + 0.2))

            color.addData4f(r, g, b, 1.0)

        # Create the points primitive
        points_prim = GeomPoints(Geom.UHStatic)
        points_prim.addNextVertices(len(points))

        # Create the geometry and attach it to a node
        geom = Geom(vdata)
        geom.addPrimitive(points_prim)
        node = GeomNode("point_cloud")
        node.addGeom(geom)

        # Create a NodePath for the point cloud
        point_cloud = self.render.attachNewNode(node)

        # Set the size of the points
        point_cloud.setRenderMode(0x1, 3)  # Adjust point size (3 pixels)

        # Enable transparency
        point_cloud.setTransparency(TransparencyAttrib.MAlpha)

        self.point_cloud = point_cloud


def main():
    # Check if a PLY file path is provided
    if len(sys.argv) < 2:
        print("Usage: python pc_viewer.py <path_to_ply_file>")
        return

    ply_file_path = sys.argv[1]

    if not os.path.exists(ply_file_path):
        print(f"Error: File {ply_file_path} does not exist")
        return

    # Create and run the viewer
    viewer = SimplePointCloudViewer(ply_file_path)
    viewer.run()


if __name__ == "__main__":
    main()
