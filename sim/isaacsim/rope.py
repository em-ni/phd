#!/usr/bin/env python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
This script demonstrates how to create a cable/rope mechanism using individual prims.
The rope is made of rigid body capsules connected with spherical joints.
"""

from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

import time
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, UsdShade, UsdLux
from omni.physx.scripts import physicsUtils
from omni.timeline import get_timeline_interface
from isaacsim.core.api import World
import omni.ui as ui
import math

from omni.isaac.dynamic_control import _dynamic_control

debug = False

# Set Light stage to Grey Studio
stage = simulation_app.context.get_stage()
light_stage = stage.GetPrimAtPath("/Environment")
if not light_stage:
    light_stage = stage.DefinePrim("/Environment", "Xform")
    
dome_light = UsdLux.DomeLight.Define(stage, "/Environment/GreyStudio")
dome_light.CreateIntensityAttr().Set(1000.0)

class SphericalJoint:
    def __init__(self, stage, joint_path, body0_path, body1_path, local_pos0, local_pos1, 
                 damping=0.0, stiffness=0.0, enable_drives=True):
        self.stage = stage
        self.joint_path = joint_path
        self.body0_path = body0_path
        self.body1_path = body1_path
        self.local_pos0 = local_pos0
        self.local_pos1 = local_pos1
        self.damping = damping
        self.stiffness = stiffness
        self.enable_drives = enable_drives
        self.joint = None
        self._create_joint()

    def _create_joint(self):
        self.joint = UsdPhysics.Joint.Define(self.stage, self.joint_path)
        joint_prim = self.joint.GetPrim()
        for axis in ["transX", "transY", "transZ", "rotX"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limitAPI.CreateLowAttr(1.0)
            limitAPI.CreateHighAttr(-1.0)
        self.joint.GetBody0Rel().SetTargets([self.body0_path])
        self.joint.GetBody1Rel().SetTargets([self.body1_path])
        self.joint.CreateLocalPos0Attr().Set(self.local_pos0)
        self.joint.CreateLocalPos1Attr().Set(self.local_pos1)
        self.joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        self.joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        max_angle = 0.785
        for axis in ["rotY", "rotZ"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limitAPI.CreateLowAttr(-max_angle)
            limitAPI.CreateHighAttr(max_angle)
            if self.enable_drives:
                driveAPI = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
                driveAPI.CreateTypeAttr("force")
                driveAPI.CreateDampingAttr(self.damping)
                driveAPI.CreateStiffnessAttr(self.stiffness)

    def get_joint(self):
        return self.joint

class RopeCreator:
    def __init__(self, stage, default_prim_path, physics_material_path, pivot_point=None, 
                 link_half_length=0.1, rope_length=2.0,
                 rope_color=None, rope_damping=2.0, rope_stiffness=0.0,
                 enable_joint_drives=True, rope_segment_mass=0.00005):
        self.stage = stage
        self.default_prim_path = default_prim_path
        self.physics_material_path = physics_material_path
        self.pivot_point = pivot_point if pivot_point is not None else Gf.Vec3f(0.0, 0.0, 1.0)
        self.link_half_length = link_half_length
        self.link_radius = 0.5 * self.link_half_length
        self.rope_length = rope_length
        self.rope_color = rope_color if rope_color is not None else Gf.Vec3f(0.2, 0.6, 0.8)
        self.rope_damping = rope_damping
        self.rope_stiffness = rope_stiffness
        self.contact_offset = link_half_length / 100.0
        self.enable_joint_drives = enable_joint_drives
        self.rope_segment_mass = rope_segment_mass
        self.segments = []
        self.joints = []

    def create_capsule(self, path, position):
        capsule_geom = UsdGeom.Capsule.Define(self.stage, path)
        capsule_geom.CreateHeightAttr(self.link_half_length)
        capsule_geom.CreateRadiusAttr(self.link_radius)
        capsule_geom.CreateAxisAttr("X")
        capsule_geom.CreateDisplayColorAttr().Set([self.rope_color])
        capsule_geom.AddTranslateOp().Set(position)
        UsdPhysics.CollisionAPI.Apply(capsule_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsule_geom.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(capsule_geom.GetPrim())
        mass_api.CreateMassAttr().Set(self.rope_segment_mass)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(capsule_geom.GetPrim())
        physx_collision_api.CreateRestOffsetAttr().Set(0.0)
        physx_collision_api.CreateContactOffsetAttr().Set(self.contact_offset)
        physicsUtils.add_physics_material_to_prim(self.stage, capsule_geom.GetPrim(), self.physics_material_path)
        return capsule_geom

    def create_joint(self, joint_path, body0_path, body1_path, local_pos0, local_pos1):
        spherical_joint = SphericalJoint(self.stage, joint_path, body0_path, body1_path, local_pos0, local_pos1,
                                         damping=self.rope_damping, stiffness=self.rope_stiffness,
                                         enable_drives=self.enable_joint_drives)
        return spherical_joint.get_joint()

    def create_ropes(self):
        link_length = 2.0 * self.link_half_length
        num_links = int(self.rope_length / link_length)
        scope_path = self.default_prim_path.AppendChild("Rope")
        UsdGeom.Scope.Define(self.stage, scope_path)
        z = self.pivot_point[2]
        for link_ind in range(num_links):
            x = self.pivot_point[0] + link_ind * link_length
            position = Gf.Vec3f(x, self.pivot_point[1], z)
            capsule_path = scope_path.AppendChild(f"capsule_{link_ind}")
            capsule_geom = self.create_capsule(capsule_path, position)
            self.segments.append(capsule_geom.GetPrim())
        joint_x = self.link_half_length * 0.5
        for link_ind in range(num_links - 1):
            joint_path = scope_path.AppendChild(f"joint_{link_ind}")
            body0_path = self.segments[link_ind].GetPath()
            body1_path = self.segments[link_ind + 1].GetPath()
            local_pos0 = Gf.Vec3f(joint_x, 0, 0)
            local_pos1 = Gf.Vec3f(-joint_x, 0, 0)
            joint = self.create_joint(joint_path, body0_path, body1_path, local_pos0, local_pos1)
            self.joints.append(joint)

def create_cylinder(stage, parent_path, name, radius=0.5, height=1.0, position=None):
    if position is None:
        position = Gf.Vec3f(0.0, 0.0, 1.0)
    cylinder_path = parent_path.AppendChild(name)
    cylinder_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
    cylinder_geom.CreateRadiusAttr().Set(radius)
    cylinder_geom.CreateHeightAttr().Set(height)
    cylinder_geom.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    cylinder_geom.AddTranslateOp().Set(position)
    UsdPhysics.CollisionAPI.Apply(cylinder_geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cylinder_geom.GetPrim())
    return cylinder_path

def create_rope_mechanism(stage, default_prim_path, mechanism_name, position, orientation_degrees):
    mechanism_path = default_prim_path.AppendChild(mechanism_name)
    mechanism_xform = UsdGeom.Xform.Define(stage, mechanism_path)
    mechanism_xform.AddTranslateOp().Set(position)
    mechanism_xform.AddRotateXOp().Set(orientation_degrees[0])
    mechanism_xform.AddRotateYOp().Set(orientation_degrees[1])
    mechanism_xform.AddRotateZOp().Set(orientation_degrees[2])

    anchor_path = mechanism_path.AppendChild("KinematicAnchor")
    anchor_xform = UsdGeom.Xform.Define(stage, anchor_path)
    anchor_xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(anchor_xform.GetPrim())
    anchor_rigid_body = UsdPhysics.RigidBodyAPI.Apply(anchor_xform.GetPrim())
    anchor_rigid_body.CreateKinematicEnabledAttr().Set(True)
    
    cylinder_radius = 0.01
    cylinder_height = 0.1
    rope_link_half_length = 0.004
    rope_link_radius = 0.004
    rope_length = 0.5
    rope_weight = 0.1
    rope_segment_mass = rope_weight / (rope_length / (2 * rope_link_half_length))
    cylinder_spacing = 2 * cylinder_radius + 2 * rope_link_radius
    
    cylinder1_pos = Gf.Vec3f(0.0, 0.0, 0.0)
    cylinder_path = create_cylinder(stage, mechanism_path, "cylinder", radius=cylinder_radius, height=cylinder_height, position=cylinder1_pos)
    
    revolute_joint1_path = mechanism_path.AppendChild("cylinder_revolute")
    revolute_joint1 = UsdPhysics.RevoluteJoint.Define(stage, revolute_joint1_path)
    revolute_joint1.CreateBody0Rel().SetTargets([anchor_path])
    revolute_joint1.CreateBody1Rel().SetTargets([cylinder_path])
    revolute_joint1.CreateAxisAttr().Set("Y")
    revolute_joint1.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revolute_joint1.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revolute_joint1.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    revolute_joint1.CreateLocalRot1Attr().Set(Gf.Quatf(0.707, 0.707, 0.0, 0.0))
    
    drive1 = UsdPhysics.DriveAPI.Apply(revolute_joint1.GetPrim(), "angular")
    drive1.CreateTypeAttr().Set("force")
    drive1.CreateTargetVelocityAttr().Set(0.0)
    drive1.CreateDampingAttr().Set(100.0)
    drive1.CreateStiffnessAttr().Set(0.0)
    drive1.CreateMaxForceAttr().Set(1000000.0)

    rope_pivot_point = Gf.Vec3f(cylinder_radius + rope_link_half_length, 0.0, 0.0)
    physics_material_path = mechanism_path.AppendChild("PhysicsMaterial")
    UsdShade.Material.Define(stage, physics_material_path)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(physics_material_path))
    material.CreateStaticFrictionAttr().Set(0.5)
    material.CreateDynamicFrictionAttr().Set(0.0)
    material.CreateRestitutionAttr().Set(0)

    rope_creator = RopeCreator(stage, mechanism_path, physics_material_path, pivot_point=rope_pivot_point,
                               rope_length=rope_length, link_half_length=rope_link_half_length,
                               rope_damping=1e10, rope_stiffness=1e6, rope_segment_mass=rope_segment_mass)
    rope_creator.create_ropes()

    cylinder2_pos = Gf.Vec3f(cylinder_spacing, 0.0, 0.0)
    cylinder2_path = create_cylinder(stage, mechanism_path, "cylinder2", radius=cylinder_radius, height=cylinder_height, position=cylinder2_pos)
    
    revolute_joint2_path = mechanism_path.AppendChild("cylinder2_revolute")
    revolute_joint2 = UsdPhysics.RevoluteJoint.Define(stage, revolute_joint2_path)
    revolute_joint2.CreateBody0Rel().SetTargets([anchor_path])
    revolute_joint2.CreateBody1Rel().SetTargets([cylinder2_path])
    revolute_joint2.CreateAxisAttr().Set("Y")
    revolute_joint2.CreateLocalPos0Attr().Set(cylinder2_pos) 
    revolute_joint2.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    revolute_joint2.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    revolute_joint2.CreateLocalRot1Attr().Set(Gf.Quatf(0.707, 0.707, 0.0, 0.0))
    
    drive2 = UsdPhysics.DriveAPI.Apply(revolute_joint2.GetPrim(), "angular")
    drive2.CreateTypeAttr().Set("force")
    drive2.CreateTargetVelocityAttr().Set(0.0)
    drive2.CreateDampingAttr().Set(100.0)
    drive2.CreateStiffnessAttr().Set(0.0)
    drive2.CreateMaxForceAttr().Set(1000000.0)

    first_capsule_path = mechanism_path.AppendChild("Rope").AppendChild("capsule_0")
    attachment_joint_path = mechanism_path.AppendChild("cylinder_rope_attachment")
    attachment_joint = UsdPhysics.SphericalJoint.Define(stage, attachment_joint_path)
    attachment_joint.GetBody0Rel().SetTargets([cylinder_path])
    attachment_joint.GetBody1Rel().SetTargets([first_capsule_path])
    attachment_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(cylinder_radius + rope_link_radius, 0.0, 0.0))
    attachment_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    attachment_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attachment_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attachment_drive = UsdPhysics.DriveAPI.Apply(attachment_joint.GetPrim(), "angular")
    attachment_drive.CreateTypeAttr().Set("force")
    attachment_drive.CreateDampingAttr().Set(1e10)
    attachment_drive.CreateStiffnessAttr().Set(1e10)
    
    return {'mechanism_path': mechanism_path, 'mechanism_xform': mechanism_xform, 'anchor_path': anchor_path,
            'anchor_rigid_body': anchor_rigid_body, 'cylinder1_path': cylinder_path, 'cylinder2_path': cylinder2_path,
            'drive1': drive1, 'drive2': drive2, 'rope_creator': rope_creator}

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
stage = simulation_app.context.get_stage()
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
default_prim_path = Sdf.Path("/World")
UsdGeom.Xform.Define(stage, default_prim_path)
stage.SetDefaultPrim(stage.GetPrimAtPath(default_prim_path))

mechanism_position = Gf.Vec3f(0.0, 0.0, 1.0)
mechanism_orientation = Gf.Vec3f(90.0+360.0, 0.0, 90.0)
mechanism = create_rope_mechanism(stage, default_prim_path, "RopeMechanism", mechanism_position, mechanism_orientation)

timeline = get_timeline_interface()
timeline.play()

while not timeline.is_playing():
    simulation_app.update()
    time.sleep(0.01)

print("Simulation started. Press Ctrl+C to exit")
cylinder_speed = 0.0
mechanism_x, mechanism_y, mechanism_z = mechanism_position
mechanism_rot_x, mechanism_rot_y, mechanism_rot_z = mechanism_orientation
mechanism['mechanism_xform'].GetOrderedXformOps()[0].Set(Gf.Vec3f(mechanism_x, mechanism_y, mechanism_z))
mechanism['mechanism_xform'].GetOrderedXformOps()[1].Set(mechanism_rot_x)
mechanism['mechanism_xform'].GetOrderedXformOps()[2].Set(mechanism_rot_y)
mechanism['mechanism_xform'].GetOrderedXformOps()[3].Set(mechanism_rot_z)
dc_interface = _dynamic_control.acquire_dynamic_control_interface()
cylinder1_path_str = mechanism['cylinder1_path'].pathString
cylinder2_path_str = mechanism['cylinder2_path'].pathString

window = ui.Window("Mechanism Controls", width=350, height=400)
with window.frame:
    with ui.VStack(spacing=5):
        ui.Label("Cylinder Rotation Control", height=20)
        ui.Spacer(height=5)
        ui.Label("Speed (-50 to 50):")
        speed_slider = ui.FloatSlider(min=-50.0, max=50.0, step=0.1, width=300)
        ui.Spacer(height=20)
        ui.Label("Mechanism Position Control", height=20)
        ui.Spacer(height=5)
        ui.Label("Position X (-2 to 2):")
        pos_x_slider = ui.FloatSlider(min=-2.0, max=2.0, step=0.01, width=300)
        pos_x_slider.model.set_value(mechanism_x)
        ui.Label("Position Y (-2 to 2):")
        pos_y_slider = ui.FloatSlider(min=-2.0, max=2.0, step=0.01, width=300)
        pos_y_slider.model.set_value(mechanism_y)
        ui.Label("Position Z (0 to 3):")
        pos_z_slider = ui.FloatSlider(min=0.0, max=3.0, step=0.01, width=300)
        pos_z_slider.model.set_value(mechanism_z)
        ui.Spacer(height=20)
        ui.Label("Mechanism Rotation Control", height=20)
        ui.Spacer(height=5)
        ui.Label("Rotation X (0 to 360):")
        rot_x_slider = ui.FloatSlider(min=0.0, max=360.0, step=1.0, width=300)
        rot_x_slider.model.set_value(mechanism_rot_x)
        ui.Label("Rotation Y (0 to 360):")
        rot_y_slider = ui.FloatSlider(min=0.0, max=360.0, step=1.0, width=300)
        rot_y_slider.model.set_value(mechanism_rot_y)
        ui.Label("Rotation Z (0 to 360):")
        rot_z_slider = ui.FloatSlider(min=0.0, max=360.0, step=1.0, width=300)
        rot_z_slider.model.set_value(mechanism_rot_z)
        
        def on_speed_change(model): global cylinder_speed; cylinder_speed = model.as_float
        def on_pos_x_change(model): global mechanism_x; mechanism_x = model.as_float
        def on_pos_y_change(model): global mechanism_y; mechanism_y = model.as_float
        def on_pos_z_change(model): global mechanism_z; mechanism_z = model.as_float
        def on_rot_x_change(model): global mechanism_rot_x; mechanism_rot_x = model.as_float
        def on_rot_y_change(model): global mechanism_rot_y; mechanism_rot_y = model.as_float
        def on_rot_z_change(model): global mechanism_rot_z; mechanism_rot_z = model.as_float
        
        speed_slider.model.add_value_changed_fn(on_speed_change)
        pos_x_slider.model.add_value_changed_fn(on_pos_x_change)
        pos_y_slider.model.add_value_changed_fn(on_pos_y_change)
        pos_z_slider.model.add_value_changed_fn(on_pos_z_change)
        rot_x_slider.model.add_value_changed_fn(on_rot_x_change)
        rot_y_slider.model.add_value_changed_fn(on_rot_y_change)
        rot_z_slider.model.add_value_changed_fn(on_rot_z_change)

loop_counter = 0
cylinder1_handle = 0
cylinder2_handle = 0

try:
    while timeline.is_playing():
        mechanism['drive1'].GetTargetVelocityAttr().Set(cylinder_speed)
        mechanism['drive2'].GetTargetVelocityAttr().Set(-cylinder_speed)

        simulation_app.update()

        if debug:
            if loop_counter % 500 == 0:
                print("--- DEBUG INFO ---")
                print(f"Frame: {loop_counter}")
                print(f"Slider Target Velocity: {cylinder_speed:.2f} rad/s")
                
                try:
                    if cylinder1_handle == 0:
                        cylinder1_handle = dc_interface.get_rigid_body(cylinder1_path_str)
                    if cylinder2_handle == 0:
                        cylinder2_handle = dc_interface.get_rigid_body(cylinder2_path_str)

                    if cylinder1_handle != 0:
                        vel1 = dc_interface.get_rigid_body_angular_velocity(cylinder1_handle)
                        print(f"Actual Angular Velocity (Cylinder 1): {vel1.y:.2f} rad/s")
                    if cylinder2_handle != 0:
                        vel2 = dc_interface.get_rigid_body_angular_velocity(cylinder2_handle)
                        print(f"Actual Angular Velocity (Cylinder 2): {vel2.y:.2f} rad/s")

                except Exception as e:
                    print(f"Error fetching velocities: {e}")

                print("--------------------")

        loop_counter += 1
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Simulation stopped by user")
finally:
    timeline.stop()
    simulation_app.close()