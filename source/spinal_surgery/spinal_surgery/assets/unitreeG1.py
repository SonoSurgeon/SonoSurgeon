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
     

"""STANDARD UNITREE G1 ROBOT CFG | WHOLEBODY WITH DEX3 HANDS"""                   
G1_CFG = ArticulationCfg(
    #  2.1) SPAWN / ASSET SETUP 
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # sonogym = 8 modify in case of penetration errors
            solver_velocity_iteration_count=2,  # sonogym = 0
        ),
    ),

    # 2.2) INITIAL STATE (POSE + DEFAULT JOINTS)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),   # overwritten in scene
        joint_pos={
            "left_hip_yaw_joint":  +0.06,
            "right_hip_yaw_joint": -0.06,
            # Legs (neutral stand; prevents knee collapse)  .* = both left/right
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,

            # Arms (bring elbows forward to avoid self-collision at start)
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,

            # DEX3 fingers (keep fully open; we mount tools rigidly anyway)
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,

            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,   # limits joints to 90% of RoM

    # 2.3) ACTUATOR GROUPS (PD-GAINS / LIMITS) 
    
    actuators={
        # LEGS 
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*", 
            ],
            # Unitree's default --> TUNE
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 300.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 400.0,
                ".*waist.*": 600.0,
            },
            damping={
                ".*_hip_yaw_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_pitch_joint": 4.0,
                ".*_knee_joint": 6.0,
                ".*waist.*": 8.0,
            },
            armature=0.01,
        ),

        # FEET 
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },
            stiffness=120.0,
            damping=3.0,
            armature=0.01,
        ),

        # SHOULDERS
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
            },
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),

        # ARMS 
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness=50.0,  
            damping=2.0,      
            armature=0.01,
        ),

        # WRIST
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness=40.0,   
            damping=2.0,     
            armature=0.01,
        ),

        # HANDS
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*_joint",
                ".*_hand_middle_.*_joint",
                ".*_hand_thumb_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=100.0, 
            damping=10.0, 
            armature=0.1, 
        ),
    },
)

"""ROBOT WITH HAND PRIMS REMOVED | WHOLEBODY"""
G1_PROVA = ArticulationCfg(
    # 2.1) Spawn / asset setup
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_tools.usd",  # no hands
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,   # increase if you see penetrations
            solver_velocity_iteration_count=2,   # tune if jittery
        ),
    ),

    # 2.2) Initial state (pose + default joints)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),   #  overwritten in scene
        joint_pos={
            # mild toe-out for stance
            "left_hip_yaw_joint":  +0.06,
            "right_hip_yaw_joint": -0.06,

            # legs (neutral stand)
            ".*_hip_pitch_joint":   -0.20,
            ".*_knee_joint":         0.42,
            ".*_ankle_pitch_joint": -0.23,

            # arms (bring elbows slightly forward)
            ".*_elbow_joint":            0.87,
            "left_shoulder_roll_joint":  0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.90,   # 90% of joint ROM

    # 2.3) Actuator groups (PD gains / limits)
    actuators={
        # LEGS + WAIST
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint":  88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint":     139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint":  32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint":     20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                ".*_hip_yaw_joint":   300.0,
                ".*_hip_roll_joint":  300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint":      400.0,
                ".*waist.*":          600.0,
            },
            damping={
                ".*_hip_yaw_joint":   4.0,
                ".*_hip_roll_joint":  4.0,
                ".*_hip_pitch_joint": 4.0,
                ".*_knee_joint":      6.0,
                ".*waist.*":          8.0,
            },
            armature=0.01,
        ),

        # FEET
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint":  35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint":  30.0,
            },
            stiffness=120.0,
            damping=3.0,
            armature=0.01,
        ),

        # SHOULDERS (pitch/roll)
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint":  25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint":  37.0,
            },
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),

        # ARMS (shoulder_yaw + elbow)
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint":        25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint":        37.0,
            },
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),

        # WRISTS (keep: roll/pitch/yaw)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit_sim={
                ".*_wrist_yaw_joint":   5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint":   22.0,
                ".*_wrist_roll_joint":  37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        # NOTE: no 'hands' actuator group here (hands are removed in the USD)
    },
)

"""ROBOT WITH HAND PRIMS REMOVED | BASE FIXED"""
G1_NOHANDS_BASE_FIX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_NOHANDS_BASEFIX.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,

        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pit   ch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.7,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
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
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 300.0,
                 ".*_elbow_joint": 400.0,
                 ".*_wrist_.*_joint": 400.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 3.0,
                 ".*_elbow_joint": 2.5,
                 ".*_wrist_.*_joint": 2.5,
             },
            armature=None,
        ),
    },
)

G1_NOHANDS_BASE_FIX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_NOHANDS_BASEFIX.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,

        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pit   ch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.7,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
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
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 300.0,
                 ".*_elbow_joint": 400.0,
                 ".*_wrist_.*_joint": 400.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 3.0,
                 ".*_elbow_joint": 2.5,
                 ".*_wrist_.*_joint": 2.5,
             },
            armature=None,
        ),
    },
)

"""ROBOT WITH SURGICAL TOOLS ALREADY MOUNTED | BASE FIXED"""
G1_TOOLS_BASE_FIX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_tools.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,

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
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": -1.57,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 1.57,
            
            "right_shoulder_pitch_joint": 0.0, #-1.57,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0, #1.0,
            "right_wrist_roll_joint": -1.57,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0, #1.57,
        },
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
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=1.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 1000.0,
                "waist_roll_joint": 1000.0,
                "waist_pitch_joint": 1000.0
            },
            damping={
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
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
                 "left_shoulder_.*_joint": 400.0,
                 "left_elbow_joint": 400.0,
                 "left_wrist_.*_joint": 500.0,
                 "right_shoulder_.*_joint": 800.0,
                 "right_elbow_joint": 800.0,
                 "right_wrist_.*_joint": 1000.0,
            },
             damping={    # increase the damping (kd)
                 "left_shoulder_.*_joint": 50.0,
                 "left_elbow_joint": 50.0,
                 "left_wrist_.*_joint": 50.0,
                 "right_shoulder_.*_joint": 100.0,
                 "right_elbow_joint": 100.0,
                 "right_wrist_.*_joint": 100.0,
             },
            armature=None,
        ),
    },
)


"""ROBOT WITH SURGICAL TOOLS ALREADY MOUNTED | BASE FIXED"""
G1_TOOLS_SURGERY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_tools.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,

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
            
            # waist joints
            "waist_yaw_joint": 0.6,
            "waist_roll_joint": -0.35,
            "waist_pitch_joint": 0.5,
            
            # arms joints
            "left_shoulder_pitch_joint": -1.57,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 1.57,
            
            "right_shoulder_pitch_joint": -1.7,
            "right_shoulder_roll_joint": -1.35,
            "right_shoulder_yaw_joint": 1.0,
            "right_elbow_joint": 0.5,
            "right_wrist_roll_joint": 0.5,
            "right_wrist_pitch_joint": -1,
            "right_wrist_yaw_joint": 1.6,
        },
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
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=1.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 1000.0,
                "waist_roll_joint": 1000.0,
                "waist_pitch_joint": 1000.0
            },
            damping={
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
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
                 "left_shoulder_.*_joint": 400.0,
                 "left_elbow_joint": 400.0,
                 "left_wrist_.*_joint": 500.0,
                 "right_shoulder_.*_joint": 800.0,
                 "right_elbow_joint": 800.0,
                 "right_wrist_.*_joint": 1000.0,
            },
             damping={    # increase the damping (kd)
                 "left_shoulder_.*_joint": 50.0,
                 "left_elbow_joint": 50.0,
                 "left_wrist_.*_joint": 50.0,
                 "right_shoulder_.*_joint": 100.0,
                 "right_elbow_joint": 100.0,
                 "right_wrist_.*_joint": 100.0,
             },
            armature=None,
        ),
    },
)

G1_TOOLS_BM_CFG = ArticulationCfg(
    #  2.1) SPAWN / ASSET SETUP 
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_tools_bm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # sonogym = 8 modify in case of penetration errors
            solver_velocity_iteration_count=2,  # sonogym = 0
        ),
    ),

    # 2.2) INITIAL STATE (POSE + DEFAULT JOINTS)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),   # overwritten in scene
        joint_pos={
            "left_hip_yaw_joint":  +0.06,
            "right_hip_yaw_joint": -0.06,
            # Legs (neutral stand; prevents knee collapse)  .* = both left/right
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,

            # Arms (bring elbows forward to avoid self-collision at start)
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,   # limits joints to 90% of RoM

    # 2.3) ACTUATOR GROUPS (PD-GAINS / LIMITS) 
    
    actuators={
        # LEGS 
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*", 
            ],
            # Unitree's default --> TUNE
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 300.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 400.0,
                ".*waist.*": 600.0,
            },
            damping={
                ".*_hip_yaw_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_pitch_joint": 4.0,
                ".*_knee_joint": 6.0,
                ".*waist.*": 8.0,
            },
            armature=0.01,
        ),

        # FEET 
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },
            stiffness=120.0,
            damping=3.0,
            armature=0.01,
        ),


        # ARMS 
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_.*_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_.*_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness={
                "left_shoulder_.*_joint": 500.0,
                "right_shoulder_.*_joint": 1000.0,
                "left_elbow_joint": 500.0,
                "right_elbow_joint": 1000.0,
            },  
            damping={
                "left_shoulder_.*_joint": 50.0,
                "right_shoulder_.*_joint": 100.0,
                "left_elbow_joint": 50.0,
                "right_elbow_joint": 100.0,
            },       
            armature=0.01,
        ),

        # WRIST
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness={
                "left_wrist_.*_joint": 500.0,
                "right_wrist_.*_joint": 1000.0,
            },   
            damping={
                "left_wrist_.*_joint": 50.0,
                "right_wrist_.*_joint": 100.0,
            },    
            armature=0.01,
        ),
    },
)