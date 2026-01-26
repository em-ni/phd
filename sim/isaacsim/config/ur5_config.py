# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import math

##
# Configuration
##


UR5_CFG = ArticulationCfg(
    collision_group = -1,
    debug_vis = True,
    spawn=sim_utils.UsdFileCfg(
        usd_path="./assets/ur5_mod.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -math.pi / 2,
            "elbow_joint": 2.0,
            "wrist_1_joint": math.pi / 2,
            "wrist_2_joint": math.pi / 2,
            "wrist_3_joint": math.pi / 2,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=1000.0,
            effort_limit_sim=870.0,
            stiffness=80000.0,
            damping=4000.0,
        ),
    },
)
"""Configuration of UR-5 arm using implicit actuator models."""
