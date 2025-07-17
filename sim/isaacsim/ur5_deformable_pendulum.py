"""
This script demonstrates how to work with a robotic arm and a deformable object.
"""

import argparse
from isaaclab.app import AppLauncher # type: ignore


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object and a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils # type: ignore
import isaaclab.sim as sim_utils # type: ignore
import isaaclab.utils.math as math_utils # type: ignore
from isaaclab.assets import DeformableObject, DeformableObjectCfg, AssetBaseCfg # type: ignore
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg # type: ignore
from isaaclab.managers import SceneEntityCfg # type: ignore
from isaaclab.markers import VisualizationMarkers # type: ignore
from isaaclab.markers.config import FRAME_MARKER_CFG # type: ignore
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg # type: ignore
from isaaclab.utils import configclass # type: ignore
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # type: ignore
from isaaclab.utils.math import subtract_frame_transforms # type: ignore
import omni.ui as ui # type: ignore
import isaacsim.core.utils.stage as stage_utils # type: ignore
from pxr import UsdPhysics
from omni.physx.scripts import deformableUtils

##
# Pre-defined configs
##
from config.ur5_config import UR5_CFG
import torch
import math



@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with a robot and a deformable object."""

    # ground plane
    ground = AssetBaseCfg(
        collision_group = -1,
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        collision_group = -1,
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
    )

    # Deformable Cylinder
    cylinder: DeformableObjectCfg = DeformableObjectCfg(
        collision_group = -1,
        prim_path="/World/Cylinder",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.02,
            height=1.0,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.1)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 1.0)),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    cylinder = scene["cylinder"]

    object_path = "/World/Cylinder"
    obstacle_prim = stage_utils.get_current_stage().GetPrimAtPath(object_path)
    mass_body = UsdPhysics.MassAPI.Apply(obstacle_prim)
    mass_body.CreateMassAttr().Set(1.0)
    deformableUtils.add_physx_deformable_body(
        stage_utils.get_current_stage(),
        object_path,
        kinematic_enabled=True,
        collision_simplification=True,
        simulation_hexahedral_resolution=8,
        self_collision=True,
        solver_position_iteration_count=20
    )

    # Nodal kinematic targets of the cylinder
    nodal_kinematic_target_cylinder = cylinder.data.nodal_kinematic_target.clone()
    initial_nodal_state = cylinder.data.default_nodal_state_w.clone()

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

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

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    # Marker for points on the plane
    point_marker_cfg = FRAME_MARKER_CFG.copy()
    point_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    plane_points_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="/Visuals/plane_points"))

    # Marker for the cylinder base
    base_marker_cfg = FRAME_MARKER_CFG.copy()
    base_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    base_marker = VisualizationMarkers(base_marker_cfg.replace(prim_path="/Visuals/cylinder_base"))

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*_joint"], body_names=["tool0"])
    robot_entity_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Find top vertices of the cylinder to constrain
    nodal_pos_w_cylinder = cylinder.data.default_nodal_state_w[0, :, :3]
    top_z = torch.max(nodal_pos_w_cylinder[:, 2])
    top_indices = torch.where(nodal_pos_w_cylinder[:, 2] >= top_z - 0.001)[0]
    print(f"Number of top vertices: {len(top_indices)}\n")

    # print the coordinates of all the vertices of the cylinder
    print("All vertices of the cylinder in world coordinates:")
    print(nodal_pos_w_cylinder)

    # Find the central point of the cylinder base by taking the one with minimum x and y coordinates among the top vertices
    base_indices = top_indices[nodal_pos_w_cylinder[top_indices, 0].argmin() : nodal_pos_w_cylinder[top_indices, 1].argmin() + 1]
    cylinder_base_origin = nodal_pos_w_cylinder[base_indices].mean(dim=0)

    # Find the difference between the top_indices vertexes and the base origin
    # print the coordinates of all the vertices of the cylinder
    differences = nodal_pos_w_cylinder[top_indices] - cylinder_base_origin
    print("\nDifferences between top vertices and base origin:")
    print(differences)

    # Associate each index in top_indices with its corresponding difference from the base origin and store it in a dictionary
    differences_dict = {idx.item(): diff for idx, diff in zip(top_indices, differences)}
    print("\nDifferences dictionary:")
    for idx, diff in differences_dict.items():
        print(f"Index {idx}: Difference {diff}")

    # Find the vertex with the max distance from the base origin
    max_distance_idx = torch.argmax(torch.norm(differences, dim=1))
    max_distance_vertex = top_indices[max_distance_idx] 
    print(f"\nVertex with max distance from base origin: {max_distance_vertex.item()}")

    # Compute the offset between the base origin and the max distance vertex
    max_distance_offset = differences[max_distance_idx] / 2
    print(f"Offset for max distance vertex: {max_distance_offset}")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # # reset
        # if count % 500 == 0:
        #     # reset counters
        #     sim_time = 0.0
        #     count = 0

        #     # reset the nodal state of the object
        #     nodal_state_cylinder = cylinder.data.default_nodal_state_w.clone()
        #     pos_w = torch.rand(cylinder.num_instances, 3, device=sim.device) * 0.1 + scene.env_origins
        #     quat_w = math_utils.random_orientation(cylinder.num_instances, device=sim.device)
        #     nodal_state_cylinder[..., :3] = cylinder.transform_nodal_pos(
        #         nodal_state_cylinder[..., :3], pos_w, quat_w
        #     )
        #     cylinder.write_nodal_state_to_sim(nodal_state_cylinder)

        #     # Write the nodal state to the kinematic target and free all vertices
        #     nodal_kinematic_target_cylinder[..., :3] = nodal_state_cylinder[..., :3]
        #     nodal_kinematic_target_cylinder[..., 3] = 1.0
        #     cylinder.write_nodal_kinematic_target_to_sim(nodal_kinematic_target_cylinder)

        #     # reset buffers
        #     cylinder.reset()

        #     print("----------------------------------------")
        #     print("[INFO]: Resetting object state...")


        # robot control
        joint_pos_des = torch.tensor(list(joint_pos_des_dict.values()), device=sim.device).repeat(scene.num_envs, 1)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.write_data_to_sim()

        # CYLINDER CONTROL
        # # Apply sinusoidal length variation to cylinder
        # scale_factor = 1.0 + 0.3 * torch.sin(torch.tensor(sim_time * 2.0))
        # current_nodal_state = cylinder.data.nodal_state_w.clone()
        # current_nodal_state[0, :, 2] *= scale_factor
        # cylinder.write_nodal_state_to_sim(current_nodal_state)

        if count > 100:
            # Find the equation of the plane
            # The end-effector pose is the origin of the plane
            ee_plane_pose = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            plane_point_w = ee_plane_pose[:, 0:3]

            # The equation of the plane is n · (x - p) = 0
            # where n is the normal, p is a point on the plane, and x is any point on the plane.
            # We can find two basis vectors for the plane (u, v) which are orthogonal to the normal.
            # The local x and y axes of the end-effector frame are suitable.
            local_x_axis = torch.tensor([1.0, 0.0, 0.0], device=sim.device).repeat(scene.num_envs, 1)
            local_y_axis = torch.tensor([0.0, 1.0, 0.0], device=sim.device).repeat(scene.num_envs, 1)

            u_w = math_utils.quat_apply(ee_plane_pose[:, 3:7], local_x_axis)
            v_w = math_utils.quat_apply(ee_plane_pose[:, 3:7], local_y_axis)

            # Generate some points on the plane for visualization
            # A point on the plane can be described as p + a*u + b*v
            num_points_per_side = 5
            points_on_plane = []
            for i in range(-num_points_per_side, num_points_per_side + 1):
                for j in range(-num_points_per_side, num_points_per_side + 1):
                    # Scale factors for the basis vectors
                    a = i * 0.02
                    b = j * 0.02
                    # Calculate the point in the world frame
                    point = plane_point_w + a * u_w + b * v_w
                    points_on_plane.append(point)

            points_on_plane = torch.cat(points_on_plane, dim=0)

            # Visualize the points on the plane
            num_points = (2 * num_points_per_side + 1) ** 2
            plane_orientations = ee_plane_pose[:, 3:7].repeat_interleave(num_points, dim=0)
            plane_points_marker.visualize(points_on_plane, plane_orientations)

            # update the kinematic target for the cylinder such that the top_indices are constrained to the plane indentified above
            nodal_kinematic_target_cylinder[0, :, 3] = 1.0  # Free all vertices
            nodal_kinematic_target_cylinder[0, top_indices, 3] = 0.0  # Constrain the top vertices

            # Calculate the target positions of the top vertices based on the differences from the base origin
            for idx in top_indices:
                # Get the difference from the dictionary
                diff = differences_dict[idx.item()]
                nodal_kinematic_target_cylinder[0, idx, :3] = plane_point_w[0] + (diff[0] - max_distance_offset[0])*u_w + -(diff[1] - max_distance_offset[1])*v_w

            # write kinematic target to simulation
            cylinder.write_nodal_kinematic_target_to_sim(nodal_kinematic_target_cylinder)
        else:
            # update the kinematic target for the cylinder
            # Free all vertices initially
            nodal_kinematic_target_cylinder[0, :, 3] = 1.0
            # Constrain the top vertices
            nodal_kinematic_target_cylinder[0, top_indices, 3] = 0.0
            # Set their target position
            nodal_kinematic_target_cylinder[0, top_indices, :3] = cylinder.data.default_nodal_state_w[0, top_indices, :3]
            cylinder.write_nodal_kinematic_target_to_sim(nodal_kinematic_target_cylinder)

        ## UPDATE THE SIMULATION
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # update markers
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # Visualize the cylinder base
        base_marker.visualize(cylinder_base_origin.unsqueeze(0))


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # set the joint position target to the default state to hold the pose
    robot = scene["robot"]
    robot.set_joint_position_target(robot.data.default_joint_pos)
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

