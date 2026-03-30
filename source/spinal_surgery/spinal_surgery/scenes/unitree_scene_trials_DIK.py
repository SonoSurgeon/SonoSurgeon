# unitree_scene_trials.py
# Copyright...
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn SonoGym scene with Unitree G1 + tools + IK controller.
Launch Isaac Sim first.

Usage:
  ./isaaclab.sh -p path/to/unitree_scene_trials.py --num_envs 1
"""

import argparse
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="SonoGym scene with Unitree G1 + tools + IK.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--reset_seconds",
    type=float,
    default=5.0,
    help="Reset every X seconds of *sim time* (<=0 disables).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_inv,
    matrix_from_quat,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.robot_arm_ik import G1_29_ArmIK as G1_ArmIK  

# Other libs
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import cProfile
import numpy as np
import torch

# Helpers: explicit Torch <-> NumPy bridges for IK 
def _t2np(t):
    """Torch tensor -> NumPy array (detached, on CPU). If already numpy, return as-is."""
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def _np2t(a, device, dtype=torch.float32):
    """NumPy array -> Torch tensor on given device/dtype."""
    return torch.from_numpy(a).to(device=device, dtype=dtype)


################### Scene parameters from YAML (bed + patient) ########################

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))

# patient pose (human)
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)

# bed pose
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)
scale_bed = bed_cfg["scale"]

# datasets (human USD/labels/CT)
patient_ids = patient_cfg["id_list"]
human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/{p}/combined_wrapwrap/combined_wrapwrap.usd"
    for p in patient_ids
]
label_res = patient_cfg["label_res"]

# robot cfg (yaw, ecc.)
robot_cfg = scene_cfg["robot"]

# Marker world position
MARKER_POS = (-0.3, -1.3, 0.7)

# copy standard G1 cfg (NO HANDS BASE FIX)
G1_SCENE_CFG: ArticulationCfg = G1_TOOLS_BASE_FIX_CFG.copy()
G1_SCENE_CFG.prim_path = "/World/envs/env_.*/G1"

# override init pose + orientation
G1_SCENE_CFG.init_state.pos = [0.0, -1.6, 0.5]
q_xyzw = R.from_euler("z", 90, degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # xyzw → wxyz
G1_SCENE_CFG.init_state.rot = q_wxyz

# IK controller settings
IK_ENABLE = True
EE_LINK_NAME = "left_wrist_yaw_link"  # oppure "tool_right_link"

# Scene config-
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal SonoGym scene + Unitree G1 + tools."""

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # medical bed (rigid)
    bed = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_no_contact/hospital_bed.usd",
            scale=(scale_bed, scale_bed, scale_bed),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=INIT_STATE_BED,
    )

    # patient (rigid)
    human = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Human",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=human_usd_list,
            random_choice=False,
            scale=(label_res, label_res, label_res),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=INIT_STATE_HUMAN,
    )

    # marker
    marker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Marker",
        spawn=sim_utils.SphereCfg(
            radius=0.003,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=100.0,
                angular_damping=100.0,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
                max_depenetration_velocity=0.01,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=MARKER_POS, rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # robot (Unitree G1)
    g1 = G1_SCENE_CFG

########################### RUN LOOP + IK ################################

def run(sim: SimulationContext, scene: InteractiveScene):
    g1: Articulation = scene["g1"]
    marker: RigidObject = scene["marker"]
    
    # IK setup: find joint & body ids of controlled arm
    SIDE = "left" if "left" in EE_LINK_NAME else ("right" if "right" in EE_LINK_NAME else None)
    joint_pattern = rf"{SIDE}_(shoulder|elbow|wrist)_.*"
    robot_entity_cfg = SceneEntityCfg("g1", joint_names=[joint_pattern], body_names=[EE_LINK_NAME])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    # IK setup: find both end-effector body ids
    robot_ee_both_cfg = SceneEntityCfg("g1", joint_names=[], body_names=["tool_left_link", "tool_right_link"])
    robot_ee_both_cfg.resolve(scene)
    left_ee_id, right_ee_id = robot_ee_both_cfg.body_ids

    ik_params = {"lambda_val": 1e-3}
    pose_diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params=ik_params,
    )
    pose_diff_ik_controller = DifferentialIKController(
        pose_diff_ik_cfg,
        scene.num_envs,
        device=sim.device,
    )

    root_pos0 = None
    root_rot0 = None
    joint_names = list(g1.data.joint_names)
    default_pos = g1.data.default_joint_pos.clone()
    zero_vel = g1.data.default_joint_vel.clone() * 0.0

    # Time bookkeeping
    sim_dt = sim.get_physics_dt()
    step_i = 0
    reset_T = float(args_cli.reset_seconds)
    sim_time_acc = 0.0

    frame_vis = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/frames",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),  # rimpicciolisci se troppo grande
                ),
            },
        )
    )

    while simulation_app.is_running():
        if step_i == 0:

            # cache root state for soft reset
            root_state = g1.data.root_state_w.clone()
            root_pos0 = root_state[:, 0:3].clone()
            root_rot0 = root_state[:, 3:7].clone()

            g1.write_joint_state_to_sim(default_pos, zero_vel)
            g1.reset()
            scene.reset()

            # Base pose in WORLD
            base_w = g1.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]

            # tool pose in WORLD
            ee_parent_w = g1.data.body_state_w[:, US_ee_jacobi_idx, 0:7]  # (N,7)

            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                ee_parent_w[:, 0:3],
                ee_parent_w[:, 3:7],
            )

            pose_diff_ik_controller.reset()
            ik_commands_pose = torch.zeros(
                scene.num_envs,
                pose_diff_ik_controller.action_dim,
                device=sim.device,
            )
            # Neutral command: keep current pose as initial target
            pose_diff_ik_controller.set_command(ik_commands_pose, ee_pos_b, ee_quat_b)


        if IK_ENABLE:
            # Base pose in WORLD
            base_w = g1.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]

            # Marker in WORLD
            marker_w = marker.data.root_state_w[:, 0:7]  # (N,7)
            marker_pos_w = marker_w[:, 0:3]
            marker_quat_w = marker_w[:, 3:7]

            # Marker in BASE
            marker_pos_b, marker_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w, marker_pos_w, marker_quat_w
            )
            marker_pos_b_t = torch.as_tensor(marker_pos_b, dtype=torch.float32)
            marker_quat_b_t = torch.as_tensor(marker_quat_b, dtype=torch.float32)

            ee_parent_w = g1.data.body_state_w[:, US_ee_jacobi_idx, 0:7]

            marker_indices = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

            # disegna il frame sull'EE (WORLD frame)
            frame_vis.visualize(ee_parent_w[:, :3], ee_parent_w[:, 3:], marker_indices=marker_indices)


            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                ee_parent_w[:, 0:3],
                ee_parent_w[:, 3:7],
            )

            base_to_ee_target_pose = torch.cat(
                [marker_pos_b_t, marker_quat_b_t],
                dim=-1,
            )

            pose_diff_ik_controller.set_command(base_to_ee_target_pose)

            US_jacobian = g1.root_physx_view.get_jacobians()[
                :, US_ee_jacobi_idx - 1, :, robot_entity_cfg.joint_ids
            ]

            base_RotMat = matrix_from_quat(quat_inv(base_quat_w))  # [N,3,3]
            US_jacobian[:, 0:3, :] = torch.bmm(
                base_RotMat, US_jacobian[:, 0:3, :]
            )
            US_jacobian[:, 3:6, :] = torch.bmm(
                base_RotMat, US_jacobian[:, 3:6, :]
            )

            US_joint_pos = g1.data.joint_pos[:, robot_entity_cfg.joint_ids]

            joint_pos_des = pose_diff_ik_controller.compute(
                ee_pos_b,
                ee_quat_b,
                US_jacobian,
                US_joint_pos,
            )

            # Apply PD targets to the arm joints
            g1.set_joint_position_target(
                joint_pos_des,
                joint_ids=torch.tensor(
                    robot_entity_cfg.joint_ids,
                    device=sim.device,
                    dtype=torch.long,
                ),
            )

            if step_i % 100 == 0:
                body_state = g1.data.body_state_w[:, US_ee_jacobi_idx, :]  # [N,13]
                ee_pos_w = body_state[:, 0:3].clone()

                print(f"L_target_pos_WORLD={marker_pos_w},")
                print(f"EE_pos_w_WORLD={ee_pos_w.detach().cpu().numpy()}")
                print(f"EE_rot_w_WORLD={ee_parent_w[:, 3:].detach().cpu().numpy()}")

        scene.write_data_to_sim()
        sim.step()
        step_i += 1
        scene.update(sim_dt)
        sim_time_acc += sim_dt

        if reset_T > 0.0 and sim_time_acc >= reset_T:
            root_state = g1.data.root_state_w.clone()
            root_state[:, 0:3] = root_pos0
            root_state[:, 3:7] = root_rot0
            root_state[:, 7:13] *= 0.0
            g1.write_root_state_to_sim(root_state)
            g1.write_joint_state_to_sim(default_pos, zero_vel)
            g1.reset()
            scene.reset()
            sim_time_acc = 0.0
            print(f"[INFO] Soft reset at sim_t={step_i * sim_dt:.2f}s")


######################## MAIN ###################################

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # camera
    sim.set_camera_view([0.0, -1.41, 1.03], [0.0, -2.94, -0.16])

    # robot with mounted tools
    G1_SCENE_CFG.spawn.usd_path = (
        f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_tools.usd"
    )

    scene = InteractiveScene(
        RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    )

    sim.reset()
    print("[INFO] Setup complete. Running…")

    scene.reset()
    run(sim, scene)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")

    simulation_app.close()