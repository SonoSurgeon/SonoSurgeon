# guided_surgery_play_left_policy_move_target.py
# SPDX-License-Identifier: BSD-3-Clause
#
# SonoGym / Isaac Lab
# - RIGHT arm: deterministic IK "move towards target" (tip follows planned trajectory direction)
# - LEFT arm: RL policy (loaded from agent.pt) -> joint position targets for left arm joints only
#
# Usage:
#   ./isaaclab.sh -p path/to/guided_surgery_play_left_policy_move_target.py --num_envs 1 --agent_path /path/to/agent.pt
#
# Notes:
# - agent.pt is a skrl-style checkpoint dict with keys like: policy, value, optimizer, value_preprocessor.
# - This script loads ONLY ckpt["policy"] into a PyTorch module you specify via --policy_class.
# - Default observation vector is built inside this script (pose errors + proprioception).
#   If your trained policy expects different obs, adapt build_left_obs().
#
# -----------------------------------------------------------------------------

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# ------------------------
# CLI
# ------------------------
parser = argparse.ArgumentParser(
    description="SonoGym play: right arm IK move-to-target + left arm RL policy (agent.pt)."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument(
    "--left_agent",
    type=str,
    default="/home/idsia/SonoGym/logs/skrl/US_guidance_G1/p20/checkpoints/best_agent.pt",
    help="Path to skrl checkpoint (agent.pt).",
)
parser.add_argument(
    "--policy_class",
    type=str,
    default="spinal_surgery.scenes.cfgs.agents:USPolicy3",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------
# Isaac Lab imports
# ------------------------
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    quat_inv,
    matrix_from_quat,
    quat_from_matrix,
    quat_mul,
    euler_xyz_from_quat,
)

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# ------------------------
# SonoGym imports (project-specific)
# ------------------------
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import nibabel as nib
import importlib

from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTDiscreteMotionGenerator,
    GTMotionGenerator,
)

def import_from_string(path: str):
    # "package.module:ClassName" oppure "__main__:ClassName"
    if ":" not in path:
        raise ValueError(f"policy_class must be like 'module:ClassName', got {path!r}")
    mod_name, cls_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


##############
# ------------------------
# Lightweight debug logging
# ------------------------
LOG_EVERY = 60          # ogni quanti step (fase policy / fase IK)
LOG_ENV = 0             # env index da loggare (0)
LOG_IMG_STATS = True
LOG_ACT_STATS = True
LOG_CMD_STATS = True

def _img_stats(x: torch.Tensor):
    # x: (N,C,H,W) float
    a = x[LOG_ENV]
    return {
        "min": float(a.min().item()),
        "max": float(a.max().item()),
        "mean": float(a.mean().item()),
        "std": float(a.std(unbiased=False).item()),
        "shape": tuple(a.shape),
        "dtype": str(a.dtype).replace("torch.", ""),
        "device": str(a.device),
    }

def _vec_stats(v: torch.Tensor):
    # v: (N,D)
    a = v[LOG_ENV]
    return {"vals": [float(x) for x in a.detach().cpu().tolist()]}

def _fmt(d: dict):
    # stampa compatta su una riga
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={v}")
    return " ".join(parts)
####################

# ------------------------
# Scene parameters from YAML
# ------------------------
scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))
sim_cfg = scene_cfg["sim"]
motion_plan_cfg = scene_cfg["motion_planning"]
patient_cfg = scene_cfg["patient"]
target_anatomy = patient_cfg["target_anatomy"]

# patient pose
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

# datasets
patient_ids = patient_cfg["id_list"]
human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id for p_id in patient_ids
]
human_stl_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id for p_id in patient_ids
]
human_raw_list = [f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id for p_id in patient_ids]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

target_stl_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/{p_id}/{target_anatomy}.stl"
    for p_id in patient_ids
]
target_traj_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/{p_id}/standard_right_traj_{str(target_anatomy)[-2:]}.stl"
    for p_id in patient_ids
]

label_res = patient_cfg["label_res"]

# robot selection
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")

LEFT_EE_NAME = "left_wrist_yaw_link"
RIGHT_EE_NAME = "right_wrist_yaw_link"

if robot_type == "g1":
    ROBOT_CFG: ArticulationCfg = G1_TOOLS_BASE_FIX_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/G1"
    ROBOT_KEY = "g1"
elif robot_type == "h1":
    ROBOT_CFG: ArticulationCfg = H12_CFG_TOOLS_BASEFIX.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/H1"
    ROBOT_KEY = "h1"
else:
    raise ValueError(f"Unknown robot type in YAML: {robot_type!r} (expected 'g1' or 'h1').")

# init pose from YAML
pos_init = scene_cfg[robot_type]["pos"]
ROBOT_CFG.init_state.pos = pos_init
q_xyzw = R.from_euler("z", scene_cfg[robot_type]["yaw"], degrees=True).as_quat()
ROBOT_CFG.init_state.rot = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

# drill tip offset (right wrist -> tip)
DRILL_TO_TIP_POS = np.array([0.305, 0.0, 0.0], dtype=np.float32)
q_xyzw = R.from_euler("YXZ", [-90, 180, 90], degrees=True).as_quat().astype(np.float32)
DRILL_TO_TIP_QUAT_WXYZ = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

# ------------------------
# Scene config
# ------------------------
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    medical_bed = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_colored/hospital_bed.usd",
            scale=(scale_bed, scale_bed, scale_bed),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=INIT_STATE_BED,
    )

    human = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Human",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_file_list,
            random_choice=False,
            scale=(label_res, label_res, label_res),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=INIT_STATE_HUMAN,
    )

    if robot_type == "g1":
        g1 = ROBOT_CFG
    else:
        h1 = ROBOT_CFG

# ------------------------
# Main run loop
# ------------------------
def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list):
    device = sim.device
    robot_name = "g1" if robot_type == "g1" else "h1"

    robot: Articulation = scene[robot_name]
    human: RigidObject = scene["human"]

    # Joint regex (left arm only for policy; right includes waist+right arm for IK)
    left_joint_pattern = rf"(left_(shoulder|elbow|wrist)_.*)"
    left_full_joint_pattern = rf"(waist_(pitch|roll|yaw)_joint|left_(shoulder|elbow|wrist)_.*)"
    right_joint_pattern = rf"(waist_(pitch|roll|yaw)_joint|right_(shoulder|elbow|wrist)_.*)"

    left_entity_cfg = SceneEntityCfg(robot_name, joint_names=[left_joint_pattern], body_names=[LEFT_EE_NAME])
    left_entity_cfg.resolve(scene)
    left_ee_id = left_entity_cfg.body_ids[-1]
    left_full_entity_cfg = SceneEntityCfg(robot_name, joint_names=[left_full_joint_pattern], body_names=[LEFT_EE_NAME])
    left_full_entity_cfg.resolve(scene)

    right_entity_cfg = SceneEntityCfg(robot_name, joint_names=[right_joint_pattern], body_names=[RIGHT_EE_NAME])
    right_entity_cfg.resolve(scene)
    right_ee_id = right_entity_cfg.body_ids[-1]

    left_joint_ids = left_entity_cfg.joint_ids
    left_full_joint_ids = left_full_entity_cfg.joint_ids
    right_joint_ids = right_entity_cfg.joint_ids

    # RIGHT arm IK controller
    ik_cfg_r = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 1e-1},
    )
    diff_ik_r = DifferentialIKController(ik_cfg_r, scene.num_envs, device=device)

    # LEFT arm IK controller
    ik_cfg_l = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 1e-1},
    )
    diff_ik_l = DifferentialIKController(ik_cfg_l, scene.num_envs, device=device)

    # LEFT arm + waist IK controller
    ik_cfg_lw = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.08},
    )
    diff_ik_lw = DifferentialIKController(ik_cfg_lw, scene.num_envs, device=device)

    R21 = np.array(
        [
            [0.0, 0.0, 1.0],  # x' = z
            [1.0, 0.0, 0.0],  # y' = x
            [0.0, 1.0, 0.0],  # z' = y
        ],
        dtype=float,
    )

    RotMat = (
        torch.as_tensor(R21, dtype=torch.float32, device=sim.device)
        .unsqueeze(0)
        .expand(scene.num_envs, -1, -1)
    )  # (N, 3, 3)

    # Fixed transforms wrist->tip (batched)
    wrist_to_tip_pos = torch.tensor(DRILL_TO_TIP_POS, dtype=torch.float32, device=device).unsqueeze(0).repeat(scene.num_envs, 1)
    wrist_to_tip_quat = torch.tensor(DRILL_TO_TIP_QUAT_WXYZ, dtype=torch.float32, device=device).unsqueeze(0).repeat(scene.num_envs, 1)

    # Inverse tip->wrist
    tip_to_wrist_pos, tip_to_wrist_quat = subtract_frame_transforms(
        wrist_to_tip_pos,
        wrist_to_tip_quat,
        torch.zeros_like(wrist_to_tip_pos),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(scene.num_envs, 1),
    )

    # Optional visuals
    frame_vis = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/frames",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                ),
            },
        )
    )

    # US slicer (used ONLY to get left-hand target pose from patient anatomy)
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
    us_generative_cfg = YAML().load(
        open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
    )
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r"))

    ROBOT_HEIGHT = scene_cfg[robot_type]["height"]
    ROBOT_HEIGHT_IMG = scene_cfg[robot_type]["height_img"]

    US_slicer = USSlicer(
        us_cfg,
        label_map_list,
        ct_map_list,
        sim_cfg["if_use_ct"],
        human_stl_list,
        scene.num_envs,
        sim_cfg["patient_xz_range"],
        sim_cfg["patient_xz_init_range"][0],
        sim.device,
        label_convert_map,
        us_cfg["image_size"],
        us_cfg["resolution"],
        img_thickness=1,
        visualize=sim_cfg["vis_seg_map"],
        sim_mode=scene_cfg["sim"]["us"],
        us_generative_cfg=us_generative_cfg,
        height=ROBOT_HEIGHT,
        height_img=ROBOT_HEIGHT_IMG,
    )
    goal_cmd_pose = (
        torch.tensor(motion_plan_cfg["patient_xz_goal"], device=sim.device)
        .reshape((1, -1))
        .repeat(scene.num_envs, 1)
    )
    gt_motion_generator = GTDiscreteMotionGenerator(
        goal_cmd_pose=goal_cmd_pose,
        scale=torch.tensor(motion_plan_cfg["scale"], device=sim.device),
        num_envs=scene.num_envs,
        surface_map_list=US_slicer.surface_map_list,
        surface_normal_list=US_slicer.surface_normal_list,
        label_res=label_res,
        US_height=US_slicer.height,
    )

    init_cmd_pose_min = (
        torch.tensor(sim_cfg["patient_xz_init_range"][0], device=sim.device)
        .reshape((1, -1))
        .repeat(scene.num_envs, 1)
    )
    init_cmd_pose_max = (
        torch.tensor(sim_cfg["patient_xz_init_range"][1], device=sim.device)
        .reshape((1, -1))
        .repeat(scene.num_envs, 1)
    )
    
    US_slicer.current_x_z_x_angle_cmd = (init_cmd_pose_min + init_cmd_pose_max) / 2

    vertebra_viewer = VertebraViewer(
        scene.num_envs,
        len(patient_ids),
        target_stl_file_list,
        target_traj_file_list,
        if_vis=False,
        res=label_res,
        device=device,
    )

    # ------------------------
    # Load policies
    # ------------------------
    agent_path = str(Path(args_cli.left_agent).expanduser())

    ckpt = torch.load(agent_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint is not a dict. Got: {type(ckpt)}")

    if "policy" not in ckpt:
        raise KeyError(f"'policy' key not found in checkpoint. Keys: {list(ckpt.keys())}")

    policy_state = ckpt["policy"]  # <-- ORA ESISTE

    PolicyClass = import_from_string(args_cli.policy_class)
    policy = PolicyClass(image_size_hw=tuple(us_cfg["image_size"])).to(device)

    missing, unexpected = policy.load_state_dict(policy_state, strict=False)
    print(f"[INFO] Loaded policy from: {agent_path}")
    print(f"[INFO] load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    policy.eval()

    # ------------------------
    # Soft reset caches
    # ------------------------
    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = robot.data.default_joint_vel.clone() * 0.0
    human_init_root_state = human.data.root_state_w.clone()

    sim_dt = sim.get_physics_dt()
    step_i = 0
    sim_time_acc = 0.0
    counter = 0
    is_rendering = sim.has_gui() or sim.has_rtx_sensors()

    # Full joint ids we will command in ONE call
    full_joint_ids_list = sorted(set(left_joint_ids + right_joint_ids))
    full_joint_ids = torch.tensor(full_joint_ids_list, device=device, dtype=torch.long)
    _jid_to_col = {jid: i for i, jid in enumerate(full_joint_ids_list)}
    left_full_cols = torch.tensor([_jid_to_col[j] for j in left_joint_ids], device=device, dtype=torch.long)
    right_full_cols = torch.tensor([_jid_to_col[j] for j in right_joint_ids], device=device, dtype=torch.long)

    # ------------------------
    # Stage 2 conf
    # ------------------------
    obs_scale = 0.1
    action_scale_left = torch.tensor([0.5, 0.5, 0.05], device=device).view(1, 3)
    max_action_left = torch.tensor([1, 1, 0.1], device=device).view(1, 3)    
    t2 = 8.0
    t2_avg = 3.0

    steps2 = max(1, int(round(t2 / sim_dt)))
    steps2_avg = max(1, int(round(t2_avg / sim_dt)))
    avg_start = max(0, steps2 - steps2_avg)

    counter1 = 0

    while simulation_app.is_running():

        if counter % 6 == 0:
            # reset robot joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            # reset human
            human.write_root_state_to_sim(human_init_root_state)
            human.reset()

            # IK reset
            diff_ik_l.reset()
            diff_ik_r.reset()
            diff_ik_lw.reset()

            # initial EE pose
            robot_root_pose_w = robot.data.root_state_w[:, 0:7]
            left_ee_pose_w = robot.data.body_state_w[:, left_ee_id, 0:7]

            left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
                robot_root_pose_w[:, 0:3],
                robot_root_pose_w[:, 3:7],
                left_ee_pose_w[:, 0:3],
                left_ee_pose_w[:, 3:7],
            )

            ik_commands_pose = torch.zeros(
                scene.num_envs,
                diff_ik_lw.action_dim,
                device=sim.device,
            )
            diff_ik_lw.set_command(ik_commands_pose, left_ee_pos_b, left_ee_quat_b)

            # update human frame
            world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
            human_world_poses = human.data.body_link_state_w[:, 0, 0:7]
            world_to_human_pos, world_to_human_rot = (
                human_world_poses[:, 0:3],
                human_world_poses[:, 3:7],
            )

            cmd_target_poses = torch.rand((scene.num_envs, 3), device=sim.device)
            min_init = init_cmd_pose_min
            max_init = init_cmd_pose_max
            cmd_target_poses = cmd_target_poses * (max_init - min_init) + min_init

            US_slicer.update_cmd(cmd_target_poses - US_slicer.current_x_z_x_angle_cmd)
            _world_to_ee_init_pos, _world_to_ee_init_rot = US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos, world_to_human_rot
            )
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)

            counter += 1

        if counter % 6 == 1:

            # human frame in WORLD
            human_world_poses = human.data.body_link_state_w[:, 0, 0:7]
            world_to_human_pos, world_to_human_rot = (
                human_world_poses[:, 0:3],
                human_world_poses[:, 3:7],
            )

            # target EE pose in WORLD (still in US frame)
            world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
                world_to_human_pos,
                world_to_human_rot,
                US_slicer.human_to_ee_target_pos,
                US_slicer.human_to_ee_target_quat,
            )

            # orientation: US frame -> robot frame
            world_ee_target_rot_us = matrix_from_quat(world_ee_target_quat_us)
            world_ee_target_rot_robot = torch.bmm(
                world_ee_target_rot_us,
                RotMat.transpose(1, 2),
            )
            world_ee_target_quat = quat_from_matrix(world_ee_target_rot_robot)

            # base in WORLD
            world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
            base_pos_w = world_to_base_pose[:, 0:3]
            base_quat_w = world_to_base_pose[:, 3:7]

            # current EE pose in WORLD, robot frame
            ee_pose_w_robot = robot.data.body_state_w[:, left_ee_id, 0:7]

            # current EE in BASE frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                ee_pose_w_robot[:, 0:3],
                ee_pose_w_robot[:, 3:7],
            )

            # target EE in BASE frame
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_ee_target_pos,
                world_ee_target_quat,
            )
            base_to_ee_target_pose = torch.cat(
                [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
            )

            # set command in BASE frame
            diff_ik_lw.set_command(base_to_ee_target_pose)

            # Jacobian WORLD -> BASE
            US_jacobian = robot.root_physx_view.get_jacobians()[
                :, left_ee_id - 1, :, left_full_joint_ids
            ]
            base_rotmat = matrix_from_quat(quat_inv(base_quat_w))

            US_jacobian[:, 0:3, :] = torch.bmm(base_rotmat, US_jacobian[:, 0:3, :])
            US_jacobian[:, 3:6, :] = torch.bmm(base_rotmat, US_jacobian[:, 3:6, :])

            # joint positions
            US_joint_pos = robot.data.joint_pos[:, left_full_joint_ids]

            # compute joint commands
            joint_pos_des = diff_ik_lw.compute(
                ee_pos_b,
                ee_quat_b,
                US_jacobian,
                US_joint_pos,
            )

            # apply joint targets
            robot.set_joint_position_target(
                joint_pos_des, joint_ids=left_full_joint_ids
            )

            counter1 += 1

            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            if is_rendering:
                sim.render()
            
            if counter1 > 500:

                # current EE pose in WORLD 
                ee_pose_w_robot = robot.data.body_state_w[:, left_ee_id, 0:7]
                ee_pos_w = ee_pose_w_robot[:, 0:3]
                ee_quat_w = ee_pose_w_robot[:, 3:7]
                
                # rotate EE orientation robot -> US frame (SonoGym convention)
                ee_rotmat_w = matrix_from_quat(ee_quat_w)
                us_rotmat_w = torch.bmm(ee_rotmat_w, RotMat)      # (N,3,3)
                us_quat_w = quat_from_matrix(us_rotmat_w)         # (N,4) wxyz

                # EE pose in US frame 
                US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)

                vertebra_to_US_2d_pos = torch.tensor([0.0, 0.0]).to(device)
                vertebra_2d_pos = vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
                US_target_2d_pos = vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0)
                US_target_2d_angle = goal_cmd_pose[:, 2:3] * torch.ones_like(vertebra_2d_pos[:, 0:1])
                US_target_2d = torch.cat([US_target_2d_pos, US_target_2d_angle], dim=-1)
                goal_cmd_pose = US_target_2d

                cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
                    world_to_human_pos,
                    world_to_human_rot,
                    US_ee_pose_w[:, 0:3],
                    US_ee_pose_w[:, 3:7],
                )
                cur_cmd_pose = gt_motion_generator.human_cmd_state_from_ee_pose(
                    cur_human_ee_pos, cur_human_ee_quat
                )
                counter2 = 0
                phase2_traj = []
                counter += 1


        if counter % 6 == 2:

            # get human frame (WORLD) 
            human_world_poses = human.data.body_link_state_w[:, 0, 0:7]
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # current EE pose in WORLD 
            ee_pose_w_robot = robot.data.body_state_w[:, left_ee_id, 0:7]
            ee_pos_w = ee_pose_w_robot[:, 0:3]
            ee_quat_w = ee_pose_w_robot[:, 3:7]

            # rotate EE orientation robot -> US frame (SonoGym convention)
            ee_rotmat_w = matrix_from_quat(ee_quat_w)
            us_rotmat_w = torch.bmm(ee_rotmat_w, RotMat)      # (N,3,3)
            us_quat_w = quat_from_matrix(us_rotmat_w)         # (N,4) wxyz

            # EE pose in US frame 
            US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)

            # build observation image 
            US_slicer.slice_US(
                world_to_human_pos,
                world_to_human_rot,
                US_ee_pose_w[:, 0:3],
                US_ee_pose_w[:, 3:7],
            )

            # da (N, 200, 150, 1) a (N, 1, 150, 200)
            obs_img = US_slicer.us_img_tensor.permute(0, 3, 2, 1).contiguous().float() * obs_scale

            with torch.inference_mode():
                act_raw = policy(obs_img)
                if isinstance(act_raw, (tuple, list)):
                    act_raw = act_raw[0]
                act_raw = act_raw.to(device)

            # usa azione "usata" separata dalla raw
            act_used = torch.clamp(act_raw * action_scale_left, -max_action_left, max_action_left)

            if counter2 % 60 == 0:
                print(f"[DBG][policy_in] shape={tuple(obs_img.shape)} "
                    f"min={obs_img.min().item():.3f} max={obs_img.max().item():.3f} "
                    f"mean={obs_img.mean().item():.3f} std={obs_img.std().item():.3f}", flush=True)
                print(f"[DBG][act_raw] {act[0].tolist()}  [DBG][act_used] {act_used[0].tolist()}", flush=True)
                d = (cur_cmd_pose[:,0:2] - cur_cmd_pose_prev[:,0:2])[0]
                print(f"[DBG][d_cur_xy] dx={d[0].item():.3f} dy={d[1].item():.3f}", flush=True)
                cur_cmd_pose_prev = cur_cmd_pose.detach()

            # action = [dx, dz, rot] in image/EE plane
            human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
                world_to_human_pos,
                world_to_human_rot,
                US_ee_pose_w[:, 0:3],
                US_ee_pose_w[:, 3:7],
            )
            human_to_ee_rot_mat = matrix_from_quat(human_to_ee_quat)

            dx_dz_human = (
                act_used[:, 0].unsqueeze(1) * human_to_ee_rot_mat[:, :, 0] +
                act_used[:, 1].unsqueeze(1) * human_to_ee_rot_mat[:, :, 1]
            )
            cmd = torch.cat([dx_dz_human[:, [0, 2]], act_used[:, 2:3]], dim=-1)  # (N,3): [dx_h, dz_h, rot]

            # update cmd in the slicer
            US_slicer.update_cmd(cmd)

            US_slicer.slice_US(world_to_human_pos, world_to_human_rot, ee_pos_w, us_quat_w)
            if sim_cfg["vis_us"]:
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg["vis_seg_map"]:
                US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, ee_pos_w, ee_quat_w)
            # compute desired WORLD EE pose from cmd (US frame quat)
            world_to_ee_target_pos, world_to_ee_target_quat_us = US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos, world_to_human_rot
            )

            # rotate target orientation from US -> robot frame
            world_to_ee_target_rotmat_us = matrix_from_quat(world_to_ee_target_quat_us)
            world_to_ee_target_rotmat_robot = torch.bmm(
                world_to_ee_target_rotmat_us,
                RotMat.transpose(1, 2),
            )
            world_to_ee_target_quat_robot = quat_from_matrix(world_to_ee_target_rotmat_robot)

            # base in WORLD
            world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
            base_pos_w = world_to_base_pose[:, 0:3]
            base_quat_w = world_to_base_pose[:, 3:7]

            # current EE in BASE (robot frame)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                ee_pose_w_robot[:, 0:3],
                ee_pose_w_robot[:, 3:7],
            )

            # target EE in BASE (robot frame)
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                world_to_ee_target_pos,
                world_to_ee_target_quat_robot,
            )
            base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

            # IK command
            diff_ik_lw.set_command(base_to_ee_target_pose)

            # Jacobian WORLD->BASE for chain joints
            J = robot.root_physx_view.get_jacobians()[:, left_ee_id - 1, :, left_full_joint_ids]  # (N,6,J)
            base_rotmat = matrix_from_quat(quat_inv(base_quat_w))
            J[:, 0:3, :] = torch.bmm(base_rotmat, J[:, 0:3, :])
            J[:, 3:6, :] = torch.bmm(base_rotmat, J[:, 3:6, :])

            q_chain = robot.data.joint_pos[:, left_full_joint_ids]
            q_des = diff_ik_lw.compute(ee_pos_b, ee_quat_b, J, q_chain)

            robot.set_joint_position_target(q_des, joint_ids=left_full_joint_ids)

            counter2 += 1

            if counter2 > avg_start:
                cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
                    world_to_human_pos, world_to_human_rot,
                    US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7],
                )
                cur_cmd_pose = gt_motion_generator.human_cmd_state_from_ee_pose(cur_human_ee_pos, cur_human_ee_quat)
                phase2_traj.append(cur_cmd_pose[:, 0:2].detach().clone())

                if LOG_CMD_STATS and (counter2 % LOG_EVERY == 0):
                    # errori: pos (pixel/label units) e rot (rad)
                    pos_err = torch.norm(cur_cmd_pose[:, 0:2] - goal_cmd_pose[:, 0:2], dim=-1)
                    rot_err = torch.abs(cur_cmd_pose[:, 2] - goal_cmd_pose[:, 2])
                    print(
                        f"[DBG][cmd] step={counter2} "
                        f"pos_err={float(pos_err[LOG_ENV].item()):.4f} "
                        f"rot_err={float(rot_err[LOG_ENV].item()):.4f} "
                        f"cur={_vec_stats(cur_cmd_pose)['vals']} "
                        f"goal={_vec_stats(goal_cmd_pose)['vals']}",
                        flush=True
                    )

            if counter2 >= steps2:

                phase2_traj_t = torch.stack(phase2_traj, dim=1)   # (N,T,2)
                guidance_pose_result = phase2_traj_t.mean(dim=1)  # (N,2)

                xy = phase2_traj_t.detach().cpu().numpy()
                gp = guidance_pose_result.detach().cpu().numpy()
                goal = goal_cmd_pose[:, 0:2].detach().cpu().numpy()

                fig = plt.figure()
                for e in range(xy.shape[0]):
                    plt.plot(xy[e, :, 0], xy[e, :, 1])
                plt.scatter(gp[:, 0], gp[:, 1])
                plt.scatter(goal[:, 0], goal[:, 1])
                fig.savefig(f"phase2_{counter}.png", dpi=150)
                plt.close(fig)

                counter += 1

            # step sim (questo "consuma" il blocco 2 come fase autonoma)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)
            sim.render()

            step_i += 1
            sim_time_acc += sim_dt
        
        if counter % 6 == 3:
            input("\n[PAUSE] counter==3. Premi INVIO per continuare...\n")


# ------------------------
# Main
# ------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(
        RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    )

    # load label maps
    label_map_list = []
    for f in label_map_file_list:
        label_map_list.append(nib.load(f).get_fdata())

    # load ct maps
    ct_map_list = []
    for f in ct_map_file_list:
        ct_map = nib.load(f).get_fdata()
        ct_min_max = scene_cfg["sim"]["ct_range"]
        ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
        ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
        ct_map_list.append(ct_map)

    sim.reset()
    print("[INFO] Setup complete. Running…")
    run(sim, scene, label_map_list, ct_map_list)

if __name__ == "__main__":
    main()
    simulation_app.close()