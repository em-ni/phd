# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable object and interact with it.

.. code-block:: bash

    # Usage
    "C:\IsaacLab\isaaclab.bat" -p deformable_pendulum.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create a single origin for one robot
    origins = [[0.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Deformable Cylinder
    cfg_cylinder = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cylinder",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.05,
            height=1.0,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.1)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 1.0)),
        debug_vis=True,
    )
    cylinder_object = DeformableObject(cfg=cfg_cylinder)

    # return the scene information
    scene_entities = {"cylinder_object": cylinder_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cylinder_object = entities["cylinder_object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target_cylinder = cylinder_object.data.nodal_kinematic_target.clone()
    initial_nodal_state = cylinder_object.data.default_nodal_state_w.clone()

    # Find top vertices of the cylinder to constrain
    nodal_pos_w_cylinder = cylinder_object.data.default_nodal_state_w[0, :, :3]
    top_z = torch.max(nodal_pos_w_cylinder[:, 2])
    # A small tolerance to find all vertices on the top surface
    top_indices = torch.where(nodal_pos_w_cylinder[:, 2] >= top_z - 0.001)[0]

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset the nodal state of the object
            nodal_state_cylinder = cylinder_object.data.default_nodal_state_w.clone()
            # apply random pose to the object
            pos_w = torch.rand(cylinder_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cylinder_object.num_instances, device=sim.device)
            nodal_state_cylinder[..., :3] = cylinder_object.transform_nodal_pos(
                nodal_state_cylinder[..., :3], pos_w, quat_w
            )

            # write nodal state to simulation
            cylinder_object.write_nodal_state_to_sim(nodal_state_cylinder)

            # Write the nodal state to the kinematic target and free all vertices
            nodal_kinematic_target_cylinder[..., :3] = nodal_state_cylinder[..., :3]
            nodal_kinematic_target_cylinder[..., 3] = 1.0
            cylinder_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target_cylinder)

            # reset buffers
            cylinder_object.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")

        # Apply sinusoidal length variation
        scale_factor = 1.0 + 0.3 * torch.sin(torch.tensor(sim_time * 2.0))  # Varies between 0.7 and 1.3
        current_nodal_state = initial_nodal_state.clone()
        current_nodal_state[0, :, 2] *= scale_factor
        cylinder_object.write_nodal_state_to_sim(current_nodal_state)

        # update the kinematic target for the cylinder
        # Free all vertices initially
        nodal_kinematic_target_cylinder[0, :, 3] = 1.0
        # Constrain the top vertices
        nodal_kinematic_target_cylinder[0, top_indices, 3] = 0.0
        # Set their target position to their initial position to hold them in place
        nodal_kinematic_target_cylinder[0, top_indices, :3] = cylinder_object.data.default_nodal_state_w[0, top_indices, :3]
        # write kinematic target to simulation
        cylinder_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target_cylinder)

        # write internal data to simulation
        cylinder_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cylinder_object.update(sim_dt)
        # print the root position
        if count % 100 == 0:
            print(f"Root position (in world): {cylinder_object.data.root_pos_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()