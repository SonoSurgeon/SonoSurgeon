# SPDX-License-Identifier: BSD-3-Clause
"""
Unitree G1 (29-DoF) — SINGLE ArticulationCfg for SonoGym tasks (DEX3 whole-body).

Hand/gripper variant used here: DEX3 whole-body. Change only the USD path if needed.
"""

#  1) IMPORTS & PATHS                                                       
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from spinal_surgery import ASSETS_DATA_DIR
     

H12_CFG_TOOLS_BASEFIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/h1-base-fix-usd/h1_toolmount_BASEFIX.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,  # Enable acceleration computation
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,

            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,

            # torso
            "torso_joint": 0.0,

            # arms joints (no fingers anymore)
            "left_shoulder_pitch_joint": -2.5,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": -2.5,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 2.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        # initialize all joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=None,
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr="torso_joint",
            effort_limit=1000,
            stiffness=500,
            damping=50,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=1000,
            velocity_limit=2.5,
             stiffness={  # increase the stiffness (kp)
                 "left_shoulder_.*_joint": 800.0,
                 "left_elbow_joint": 800.0,
                 "left_wrist_.*_joint": 1000.0,
                 "right_shoulder_.*_joint": 800.0,
                 "right_elbow_joint": 800.0,
                 "right_wrist_.*_joint": 1000.0,
            },
             damping={    # increase the damping (kd)
                 "left_shoulder_.*_joint": 200.0,
                 "left_elbow_joint": 200.0,
                 "left_wrist_.*_joint": 200.0,
                 "right_shoulder_.*_joint": 100.0,
                 "right_elbow_joint": 100.0,
                 "right_wrist_.*_joint": 100.0,
             },
            armature=None,
        ),
    },
)

H12_TOOLS_SURGERY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/h1-base-fix-usd/h1_toolmount_BASEFIX.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,  # Enable acceleration computation
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,

            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,

            # torso
            "torso_joint": 0.72,

            # arms joints (no fingers anymore)
            "left_shoulder_pitch_joint": -1.95,
            "left_shoulder_roll_joint": -0.35,
            "left_shoulder_yaw_joint": 1.05,
            "left_elbow_joint": 2.68,
            "left_wrist_roll_joint": 0.12,
            "left_wrist_pitch_joint": 0.27,
            "left_wrist_yaw_joint": 0.57,
            
            "right_shoulder_pitch_joint": -2.2,
            "right_shoulder_roll_joint": -0.83,
            "right_shoulder_yaw_joint": 2.24,
            "right_elbow_joint": 0.53, #0.53,
            "right_wrist_roll_joint": 0.4,
            "right_wrist_pitch_joint": -0.46,
            "right_wrist_yaw_joint": 1.26,
        },
        # initialize all joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=None,
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr="torso_joint",
            effort_limit=1000,
            stiffness=500,
            damping=50,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=1000,
            velocity_limit=2.5,
             stiffness={  # increase the stiffness (kp)
                 "left_shoulder_.*_joint": 600.0,
                 "left_elbow_joint": 600.0,
                 "left_wrist_.*_joint": 600.0,
                 "right_shoulder_.*_joint": 400.0,
                 "right_elbow_joint": 400.0,
                 "right_wrist_.*_joint": 400.0,
            },
             damping={    # increase the damping (kd)
                 "left_shoulder_.*_joint": 60.0,
                 "left_elbow_joint": 60.0,
                 "left_wrist_.*_joint": 60.0,
                 "right_shoulder_.*_joint": 40.0,
                 "right_elbow_joint": 40.0,
                 "right_wrist_.*_joint": 40.0,
             },
            armature=None,
        ),
    },
)