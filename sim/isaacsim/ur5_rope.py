#!/usr/bin/env python3

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="UR5 with rope mechanism end effector.")
parser.add_argument("--robot", type=str, default="ur5", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from config.ur5_config import UR5_CFG
import torch
import math
import time
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, UsdShade, UsdLux
import time
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema, UsdShade, UsdLux
from omni.physx.scripts import physicsUtils
from omni.timeline import get_timeline_interface
from isaacsim.core.api import World
from isaacsim.core.utils.prims import get_prim_at_path
import omni.ui as ui

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
        # Create a D6 joint
        self.joint = UsdPhysics.Joint.Define(self.stage, self.joint_path)
        joint_prim = self.joint.GetPrim()

        # Lock all translational degrees of freedom
        for axis in ["transX", "transY", "transZ", "rotX"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
            limitAPI.CreateLowAttr(1.0)
            limitAPI.CreateHighAttr(-1.0)

        # Set the bodies to connect
        self.joint.GetBody0Rel().SetTargets([self.body0_path])
        self.joint.GetBody1Rel().SetTargets([self.body1_path])

        # Set local positions
        self.joint.CreateLocalPos0Attr().Set(self.local_pos0)
        self.joint.CreateLocalPos1Attr().Set(self.local_pos1)

        # Set local rotations (identity quaternions)
        self.joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        self.joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        # Add angular limits to prevent excessive coiling (±45 degrees = ±0.785 radians)
        max_angle = 0.785  # 45 degrees in radians
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
                 link_half_length=0.1, rope_length=2.0, num_ropes=1, rope_spacing=0.1,
                 rope_color=None, rope_damping=2.0, rope_stiffness=0.0, contact_offset=0.001,
                 enable_joint_drives=True, rope_segment_mass=0.00005, rope_axis="X"):
        self.stage = stage
        self.default_prim_path = default_prim_path
        self.physics_material_path = physics_material_path
        self.pivot_point = pivot_point if pivot_point is not None else Gf.Vec3f(0.0, 0.0, 1.0)
        
        # Configure rope parameters
        self.link_half_length = link_half_length
        self.link_radius = 0.5 * self.link_half_length
        self.rope_length = rope_length
        self.num_ropes = num_ropes
        self.rope_spacing = rope_spacing
        self.rope_color = rope_color if rope_color is not None else Gf.Vec3f(0.2, 0.6, 0.8)  # Blue-ish color
        self.rope_damping = rope_damping
        self.rope_stiffness = rope_stiffness
        self.contact_offset = contact_offset
        self.enable_joint_drives = enable_joint_drives
        self.rope_segment_mass = rope_segment_mass
        self.rope_axis = rope_axis  # 'X', 'Y', or 'Z'
        
        # Store segment prims and joints for the rope
        self.segments = []  # List of segment prims
        self.joints = []    # List of joint prims

    def create_capsule(self, path, position):
        capsule_geom = UsdGeom.Capsule.Define(self.stage, path)
        capsule_geom.CreateHeightAttr(self.link_half_length)
        capsule_geom.CreateRadiusAttr(self.link_radius)
        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.rope_axis, UsdGeom.Tokens.x)
        capsule_geom.CreateAxisAttr(axis_token)
        capsule_geom.CreateDisplayColorAttr().Set([self.rope_color])
        
        # Set position
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
        # Create a spherical joint using the new abstraction
        spherical_joint = SphericalJoint(
            self.stage,
            joint_path,
            body0_path,
            body1_path,
            local_pos0,
            local_pos1,
            damping=self.rope_damping,
            stiffness=self.rope_stiffness,
            enable_drives=self.enable_joint_drives
        )
        return spherical_joint.get_joint()

    def create_ropes(self):
        # Calculate spacing and dimensions
        link_length = 2.0 * self.link_half_length
        num_links = int(self.rope_length / link_length)
        y_start = -(self.num_ropes // 2) * self.rope_spacing

        for rope_ind in range(self.num_ropes):
            scope_path = self.default_prim_path.AppendChild(f"Rope{rope_ind}")
            UsdGeom.Scope.Define(self.stage, scope_path)
            
            y = y_start + rope_ind * self.rope_spacing
            z = self.pivot_point[2]  # Use pivot point's z coordinate
            
            # Create all capsule links along the selected axis
            for link_ind in range(num_links):
                if self.rope_axis == "X":
                    x = self.pivot_point[0] + link_ind * link_length
                    position = Gf.Vec3f(x, self.pivot_point[1] + y, z)
                elif self.rope_axis == "Y":
                    # Grow downward along -Y from pivot
                    y_pos = self.pivot_point[1] - link_ind * link_length
                    position = Gf.Vec3f(self.pivot_point[0], y_pos, z)
                else:  # 'Z'
                    z_pos = self.pivot_point[2] + link_ind * link_length
                    position = Gf.Vec3f(self.pivot_point[0], self.pivot_point[1] + y, z_pos)
                
                capsule_path = scope_path.AppendChild(f"capsule_{link_ind}")
                capsule_geom = self.create_capsule(capsule_path, position)
                self.segments.append(capsule_geom.GetPrim())
            
            # Create joints between consecutive capsules
            joint_x = self.link_half_length
            for link_ind in range(num_links - 1):
                joint_path = scope_path.AppendChild(f"joint_{link_ind}")
                
                body0_path = self.segments[link_ind].GetPath()
                body1_path = self.segments[link_ind + 1].GetPath()
                
                if self.rope_axis == "X":
                    local_pos0 = Gf.Vec3f(joint_x, 0, 0)
                    local_pos1 = Gf.Vec3f(-joint_x, 0, 0)
                elif self.rope_axis == "Y":
                    local_pos0 = Gf.Vec3f(0, joint_x, 0)
                    local_pos1 = Gf.Vec3f(0, -joint_x, 0)
                else:
                    local_pos0 = Gf.Vec3f(0, 0, joint_x)
                    local_pos1 = Gf.Vec3f(0, 0, -joint_x)
                
                joint = self.create_joint(joint_path, body0_path, body1_path, local_pos0, local_pos1)
                self.joints.append(joint)

    def get_segments(self):
        """Get all segment prims for the rope"""
        return self.segments
    
    def get_joints(self):
        """Get all joint prims for the rope"""
        return self.joints
    
    def get_first_segment(self):
        """Get the first segment prim for the rope"""
        return self.segments[0] if self.segments else None
    
    def get_last_segment(self):
        """Get the last segment prim for the rope"""
        return self.segments[-1] if self.segments else None

def create_cylinder(stage, parent_path, name, radius=0.5, height=1.0, position=None, body0_path=None):
    """
    Create a cylinder with physics properties and a revolute joint.
    
    Args:
        stage: The USD stage
        parent_path: Path to the parent prim
        name: Name for the cylinder prim
        radius: Radius of the cylinder in meters
        height: Height of the cylinder in meters
        position: Position of the cylinder (Gf.Vec3f), defaults to (0, 0, 2)
    
    Returns:
        tuple: (cylinder_path, joint_path) - Paths to the created cylinder and joint
    """
    if position is None:
        position = Gf.Vec3f(0.0, 0.0, 1.0)
        
    # Create cylinder
    cylinder_path = parent_path.AppendChild(name)
    cylinder_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
    cylinder_geom.CreateRadiusAttr().Set(radius)
    cylinder_geom.CreateHeightAttr().Set(height)
    cylinder_geom.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    
    # Add physics properties
    UsdPhysics.CollisionAPI.Apply(cylinder_geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cylinder_geom.GetPrim())
    
    # Create revolute joint
    joint_path = parent_path.AppendChild(f"{name}_joint")
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    
    # Set joint bodies
    # Anchor the cylinder to the specified body0 (e.g., Robot tool0). If not provided, fall back to parent_path.
    if body0_path is None:
        body0_path = parent_path
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([cylinder_path])
    
    # Set joint axis
    joint.CreateAxisAttr().Set("X")
    
    # Add joint drive
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateTargetVelocityAttr().Set(10.0)  # Rotate at 1 rad/s
    drive.CreateDampingAttr().Set(1.0)
    drive.CreateStiffnessAttr().Set(10.0)
    
    return cylinder_path, joint_path, drive


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for UR5 with rope mechanism."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["arm"].stiffness = 400.0
    robot.actuators["arm"].damping = 40.0

def create_rope_mechanism_at_ee(stage, tool0_path: Sdf.Path, env_idx=0, local_offset=Gf.Vec3f(0.0, 0.0, 0.05)):
    """Create two cylinders (rollers) and a rope between them, all aligned to tool0 Z axis.

    - Cylinders long axis: Z
    - Revolute joint axis: Z
    - Rope built from capsules along +X starting at gap between cylinders.
    """
    mechanism_path = tool0_path.AppendChild("RopeMechanism")
    UsdGeom.Xform.Define(stage, mechanism_path)
    UsdGeom.Xform.Get(stage, mechanism_path).AddTranslateOp().Set(local_offset)

    # Physics material (kept minimal)
    physics_material_path = mechanism_path.AppendChild("PhysicsMaterial")
    UsdShade.Material.Define(stage, physics_material_path)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(physics_material_path))
    material.CreateStaticFrictionAttr().Set(0.5)
    material.CreateDynamicFrictionAttr().Set(0.5)
    material.CreateRestitutionAttr().Set(0.0)

    # Roller parameters
    cylinder_radius = 0.012
    cylinder_height = 0.05
    offset_from_ee = 0.028
    rope_link_half_length = 0.005
    rope_link_radius = rope_link_half_length * 0.5
    gap_between_cylinders = 2 * rope_link_radius + 0.001  # small gap for rope to fit through

    # Place cylinders side-by-side along X with a gap, axes parallel (Z)
    half_gap = 0.5 * gap_between_cylinders
    x_offset_to_center = cylinder_radius + half_gap
    cylinder1_pos = Gf.Vec3f(-x_offset_to_center, 0.0, offset_from_ee)
    cylinder2_pos = Gf.Vec3f(+x_offset_to_center, 0.0, offset_from_ee)

    # Cylinder 1
    cylinder1_path = mechanism_path.AppendChild("cylinder1")
    cyl1 = UsdGeom.Cylinder.Define(stage, cylinder1_path)
    cyl1.CreateRadiusAttr().Set(cylinder_radius)
    cyl1.CreateHeightAttr().Set(cylinder_height)
    cyl1.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    cyl1.AddTranslateOp().Set(cylinder1_pos)
    UsdPhysics.CollisionAPI.Apply(cyl1.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cyl1.GetPrim())

    joint1_path = mechanism_path.AppendChild("cylinder1_joint")
    joint1 = UsdPhysics.RevoluteJoint.Define(stage, joint1_path)
    joint1.CreateBody0Rel().SetTargets([tool0_path])
    joint1.CreateBody1Rel().SetTargets([cylinder1_path])
    joint1.CreateAxisAttr().Set("Z")
    joint1.CreateLocalPos0Attr().Set(cylinder1_pos)
    joint1.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint1.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint1.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    drive1 = UsdPhysics.DriveAPI.Apply(joint1.GetPrim(), "angular")
    drive1.CreateTypeAttr().Set("force")
    drive1.CreateTargetVelocityAttr().Set(5.0)
    drive1.CreateDampingAttr().Set(1.0)
    drive1.CreateStiffnessAttr().Set(5.0)

    # Cylinder 2
    cylinder2_path = mechanism_path.AppendChild("cylinder2")
    cyl2 = UsdGeom.Cylinder.Define(stage, cylinder2_path)
    cyl2.CreateRadiusAttr().Set(cylinder_radius)
    cyl2.CreateHeightAttr().Set(cylinder_height)
    cyl2.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    cyl2.AddTranslateOp().Set(cylinder2_pos)
    UsdPhysics.CollisionAPI.Apply(cyl2.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cyl2.GetPrim())

    joint2_path = mechanism_path.AppendChild("cylinder2_joint")
    joint2 = UsdPhysics.RevoluteJoint.Define(stage, joint2_path)
    joint2.CreateBody0Rel().SetTargets([tool0_path])
    joint2.CreateBody1Rel().SetTargets([cylinder2_path])
    joint2.CreateAxisAttr().Set("Z")
    joint2.CreateLocalPos0Attr().Set(cylinder2_pos)
    joint2.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint2.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint2.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    drive2 = UsdPhysics.DriveAPI.Apply(joint2.GetPrim(), "angular")
    drive2.CreateTypeAttr().Set("force")
    drive2.CreateTargetVelocityAttr().Set(5.0)
    drive2.CreateDampingAttr().Set(1.0)
    drive2.CreateStiffnessAttr().Set(5.0)

    # Physics material for rope contact
    physics_material_path = mechanism_path.AppendChild("PhysicsMaterial")
    UsdShade.Material.Define(stage, physics_material_path)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(physics_material_path))
    material.CreateStaticFrictionAttr().Set(0.7)
    material.CreateDynamicFrictionAttr().Set(0.6)
    material.CreateRestitutionAttr().Set(0.0)

    # Rope starting at the midpoint between cylinders, then going downward along -Y
    pivot_z = offset_from_ee
    rope_creator = RopeCreator(
        stage,
        mechanism_path,
        physics_material_path,
        pivot_point=Gf.Vec3f(0.0, 0.0, pivot_z),
        rope_length=0.4,
        link_half_length=rope_link_half_length,
        num_ropes=1,
        rope_spacing=0.0,
        rope_damping=1e5,
        rope_stiffness=0.0,
        rope_segment_mass=1e-5,
        enable_joint_drives=False,
        rope_axis="Y",
    )
    rope_creator.create_ropes()

    # Attach rope to cylinder1 inner rim via FixedJoint
    rope_scope_path = mechanism_path.AppendChild("Rope0")
    first_capsule_path = rope_scope_path.AppendChild("capsule_0")
    attach_path = mechanism_path.AppendChild("cylinder1_rope_attachment")
    attach_joint = UsdPhysics.FixedJoint.Define(stage, attach_path)
    attach_joint.GetBody0Rel().SetTargets([cylinder1_path])
    attach_joint.GetBody1Rel().SetTargets([first_capsule_path])
    attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(cylinder_radius + rope_link_radius, 0.0, 0.0))
    attach_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    attach_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    return mechanism_path, drive1, drive2


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    stage = sim.stage

    # Create a dictionary to hold joint position targets
    joint_pos_des_dict = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -math.pi / 2,
        "elbow_joint": 2.0,
        "wrist_1_joint": math.pi / 2,
        "wrist_2_joint": math.pi / 2,
        "wrist_3_joint": math.pi / 2,
    }

    # Create a window for the joint controls
    joint_control_window = ui.Window("UR5 Joint Control", width=400, height=300)
    with joint_control_window.frame:
        with ui.VStack():
            for joint_name, joint_value in joint_pos_des_dict.items():
                with ui.HStack():
                    ui.Label(joint_name, width=150)
                    # Use a lambda to capture the joint_name in the closure
                    def on_value_changed(model, name=joint_name):
                        joint_pos_des_dict[name] = model.get_value_as_float()
                    
                    model = ui.FloatSlider(min=-2*math.pi, max=2*math.pi, step=0.01).model
                    model.set_value(joint_value)
                    model.add_value_changed_fn(on_value_changed)
            # Additional slider to control both cylinders together (opposite directions)
            cylinder_control = {"pos": 0.0}
            with ui.HStack():
                ui.Label("rollers", width=150)
                def on_cyl_value_changed(model):
                    cylinder_control["pos"] = model.get_value_as_float()
                cyl_model = ui.FloatSlider(min=-1000.0, max=1000.0, step=1).model
                cyl_model.set_value(0.0)
                cyl_model.add_value_changed_fn(on_cyl_value_changed)

    # Robot entity configuration
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[".*_joint"],
        body_names=["tool0"],
    )
    robot_entity_cfg.resolve(scene)
    
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    # Create rope mechanisms for each environment
    mechanisms = []
    drives1 = []
    drives2 = []
    
    for env_idx in range(scene.num_envs):
        # Get tool0 prim path for this env and ensure it exists
        tool0_path = Sdf.Path(f"/World/envs/env_{env_idx}/Robot/tool0")
        if not stage.GetPrimAtPath(tool0_path).IsValid():
            print(f"[WARN] tool0 prim not found at {tool0_path}. Rope mechanism skipped for env {env_idx}.")
            continue

    # Create mechanism parented to tool0 with a small forward offset
    mechanism_path, drive1, drive2 = create_rope_mechanism_at_ee(stage, tool0_path, env_idx)
    mechanisms.append(mechanism_path)
    drives1.append(drive1)
    drives2.append(drive2)

    sim_dt = sim.get_physics_dt()
    count = 0
    position1 = 0.0
    position2 = 0.0
    cylinder_speed = 5.0

    while simulation_app.is_running():
        # robot control
        joint_pos_des = torch.tensor(list(joint_pos_des_dict.values()), device=sim.device).repeat(scene.num_envs, 1)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.write_data_to_sim()

        # Update cylinder positions
        position1 += cylinder_speed * 0.01
        position2 -= cylinder_speed * 0.01  # opposite direction
        # If user moves the GUI slider, adjust joint target VELOCITIES (opposite directions)
        if 'cylinder_control' in locals():
            slider_vel = cylinder_control["pos"]
            for d1, d2 in zip(drives1, drives2):
                d1.CreateTargetVelocityAttr().Set(slider_vel)
                d2.CreateTargetVelocityAttr().Set(-slider_vel)
        for d1, d2 in zip(drives1, drives2):
            d1.CreateTargetPositionAttr().Set(position1)
            d2.CreateTargetPositionAttr().Set(position2)

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Enable physics
    UsdPhysics.Scene.Define(sim.stage, "/physicsScene")
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()