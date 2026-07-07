# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Spawn SonoGym scene and drive the left probe with a frozen guidance policy."""

from __future__ import annotations

import argparse
import os
import cProfile

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Mock simulation with frozen probe policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--reset_seconds", type=float, default=8.0, help="Reset every X seconds of sim time.")
parser.add_argument("--debug_every", type=int, default=30, help="Print debug info every N physics steps. 0 disables periodic debug.")
parser.add_argument("--control_decimation", type=int, default=2, help="Apply policy/IK command every N physics steps, matching DirectRLEnv decimation.")
parser.add_argument("--physics_dt", type=float, default=1.0/120.0, help="Physics dt. Use 1/120 to match DirectRLEnv training cfg.")
parser.add_argument("--cmd_update_mode", type=str, default="update_cmd", choices=["update_cmd", "direct"], help="Use USSlicer.update_cmd(cmd), as in guidance training, or direct current_cmd += cmd.")
parser.add_argument("--obs_scale", type=float, default=None, help="Override observation scale. If None, use scene_cfg['observation']['scale'] if present, else 0.02.")
parser.add_argument(
    "--policy",
    type=str,
    default="/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/policies/prova_policy_jit.pt",
    help="Path to exported TorchScript probe policy (.pt).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac / SonoGym imports
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_mul

from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer

import pinocchio as pin
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pink.solve_ik import solve_ik

from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import nibabel as nib
import numpy as np
import torch
import logging

# filter out joint limit warnings from Pink
class JointLimitFilter(logging.Filter):
    """Filter out joint limit warnings on the root logger."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Drop messages that match the specific joint limit pattern
        if "is out of limits" in msg and "Value" in msg:
            return False  # do not log this record
        return True       # keep all other log records
root_logger = logging.getLogger()  # this is the root logger
root_logger.addFilter(JointLimitFilter())

# -----------------------------------------------------------------------------
# Scene parameters from YAML
# -----------------------------------------------------------------------------
scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))

patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)

bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)
scale_bed = bed_cfg["scale"]

patient_ids = patient_cfg["id_list"]

human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id for p_id in patient_ids
]
human_stl_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id for p_id in patient_ids
]
human_raw_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id for p_id in patient_ids
]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg["label_res"]

robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")
EE_LINK_NAME = "left_wrist_yaw_link"

if robot_type == "g1":
    ROBOT_CFG: ArticulationCfg = G1_TOOLS_SURGERY_CFG2 # G1_TOOLS_BASE_FIX_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/G1"
    ROBOT_KEY = "g1"
elif robot_type == "h1":
    ROBOT_CFG: ArticulationCfg = H12_CFG_TOOLS_BASEFIX.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/H1"
    ROBOT_KEY = "h1"
else:
    raise ValueError(f"Unknown robot type in YAML: {robot_type!r}")

ROBOT_CFG.init_state.pos = scene_cfg[robot_type]["pos"]
q_xyzw = R.from_euler("z", scene_cfg[robot_type]["yaw"], degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
ROBOT_CFG.init_state.rot = q_wxyz

ROBOT_HEIGHT = scene_cfg[robot_type]["height"]
ROBOT_HEIGHT_IMG = scene_cfg[robot_type]["height_img"]

if robot_type == "g1":
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif robot_type == "h1":
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_body29_hand14.urdf"
else:
    raise ValueError(f"Unsupported robot_type for URDF selection: {robot_type!r}")


# -----------------------------------------------------------------------------
# Scene config
# -----------------------------------------------------------------------------
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
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
        init_state=INIT_STATE_BED
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _t2np(t: torch.Tensor):
    return t.detach().cpu().numpy()

def build_pinocchio_model(urdf_path: str):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

def build_name_to_qidx(model: pin.Model):
    name_to_qidx = {}
    for j_id, joint in enumerate(model.joints):
        name = model.names[j_id]
        if joint.nq == 1:
            name_to_qidx[name] = joint.idx_q
    return name_to_qidx

def isaac_to_pin_q(q_isaac: torch.Tensor, isaac_joint_names: list[str], model: pin.Model, name_to_qidx: dict[str, int]) -> np.ndarray:
    q_pin = np.zeros(model.nq, dtype=float)
    q_isaac_np = q_isaac.detach().cpu().numpy()
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            idx_q = name_to_qidx[jname]
            q_pin[idx_q] = q_isaac_np[i]
    return q_pin

def pin_to_isaac_q(q_pin: np.ndarray, isaac_joint_names: list[str], name_to_qidx: dict[str, int]) -> np.ndarray:
    q_isaac = np.zeros(len(isaac_joint_names), dtype=float)
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            idx_q = name_to_qidx[jname]
            q_isaac[i] = q_pin[idx_q]
    return q_isaac


# -----------------------------------------------------------------------------
# Policy wrapper
# -----------------------------------------------------------------------------
class FrozenProbePolicy:
    def __init__(self, policy_path: str, device: torch.device):
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

    @torch.no_grad()
    def act(self, obs_img: torch.Tensor) -> torch.Tensor:
        action = self.policy(obs_img.to(self.device))
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return action


# -----------------------------------------------------------------------------
# Run loop
# -----------------------------------------------------------------------------
def run(
    sim: SimulationContext,
    scene: InteractiveScene,
    label_map_list: list,
    ct_map_list: list | None = None,
):
    robot = scene[ROBOT_KEY]
    human = scene["human"]

    ik_mode = str(scene_cfg["robot"].get("ik", "differentialIK")).lower()
    if ik_mode in ["differential_ik", "differentialik", "diffik", "dls"]:
        ik_mode = "differentialIK"
    elif ik_mode in ["pink", "pinocchio"]:
        ik_mode = "pink"
    else:
        raise ValueError(
            f"Invalid robot.ik={ik_mode!r}. Use 'differentialIK' or 'pink'."
        )

    print(f"[INFO] IK mode: {ik_mode}")

    SIDE = "left" if "left" in EE_LINK_NAME else ("right" if "right" in EE_LINK_NAME else None)
    joint_pattern = rf"(waist_(pitch|roll|yaw)_joint|{SIDE}_(shoulder|elbow|wrist)_.*)" if SIDE else r".*"

    robot_entity_cfg = SceneEntityCfg(
        ROBOT_KEY,
        joint_names=[joint_pattern],
        body_names=[EE_LINK_NAME],
    )
    robot_entity_cfg.resolve(scene)

    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    # -------------------------------------------------------------------------
    # Differential IK controller
    # -------------------------------------------------------------------------
    ik_params = {"lambda_val": 0.08}
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

    # -------------------------------------------------------------------------
    # Pink / Pinocchio setup
    # -------------------------------------------------------------------------
    pin_model = None
    pin_datas = None
    pink_cfgs = None
    pink_tasks = None
    name_to_qidx = None
    isaac_joint_names = None

    if ik_mode == "pink":
        pin_model, _ = build_pinocchio_model(URDF_PATH)
        pin_datas = [pin_model.createData() for _ in range(scene.num_envs)]
        name_to_qidx = build_name_to_qidx(pin_model)
        isaac_joint_names = list(robot.data.joint_names)

        pink_cfgs = []
        pink_tasks = []

        for env_id in range(scene.num_envs):
            q0 = isaac_to_pin_q(
                robot.data.joint_pos[env_id],
                isaac_joint_names,
                pin_model,
                name_to_qidx,
            )
            pink_cfgs.append(Configuration(pin_model, pin_datas[env_id], q0))
            pink_tasks.append(
                FrameTask(
                    frame=EE_LINK_NAME,
                    position_cost=1.0,
                    orientation_cost=1.0,
                )
            )

    # -------------------------------------------------------------------------
    # US / policy setup
    # -------------------------------------------------------------------------
    label_convert_map = YAML().load(
        open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
    )
    us_cfg = YAML().load(
        open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r")
    )
    us_generative_cfg = YAML().load(
        open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
    )

    sim_cfg = scene_cfg["sim"]

    # Match the guidance task observation scaling. Prefer YAML if available.
    if args_cli.obs_scale is not None:
        obs_scale = float(args_cli.obs_scale)
    else:
        obs_scale = float(scene_cfg.get("observation", {}).get("scale", 0.02))
    if scene_cfg.get("sim", {}).get("us", None) == "net" and "observation" in scene_cfg:
        obs_scale = float(scene_cfg["observation"].get("scale_net", obs_scale))

    print(f"[CONFIG] robot_type={robot_type} IK={ik_mode} sim_dt={sim.get_physics_dt():.6f} "
          f"control_decimation={args_cli.control_decimation} physics_dt_arg={args_cli.physics_dt:.6f} cmd_update_mode={args_cli.cmd_update_mode} obs_scale={obs_scale}", flush=True)

    action_scale = torch.tensor(
        scene_cfg["motion_planning"]["action_scale"],
        device=sim.device,
        dtype=torch.float32,
    ).reshape(1, -1)

    max_action = torch.tensor(
        scene_cfg["motion_planning"]["max_action"],
        device=sim.device,
        dtype=torch.float32,
    ).reshape(1, -1)

    init_cmd_pose_min = torch.tensor(
        sim_cfg["patient_xz_init_range"][0],
        dtype=torch.float32,
        device=sim.device,
    ).reshape(1, -1).repeat(scene.num_envs, 1)

    init_cmd_pose_max = torch.tensor(
        sim_cfg["patient_xz_init_range"][1],
        dtype=torch.float32,
        device=sim.device,
    ).reshape(1, -1).repeat(scene.num_envs, 1)

    print(f"[CONFIG] action_scale={action_scale[0].detach().cpu().numpy()} "
          f"max_action={max_action[0].detach().cpu().numpy()}", flush=True)
    print(f"[CONFIG] init_cmd_min={init_cmd_pose_min[0].detach().cpu().numpy()} "
          f"init_cmd_max={init_cmd_pose_max[0].detach().cpu().numpy()}", flush=True)

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
        roll_adj=scene_cfg["motion_planning"]["US_roll_adj"],
        visualize=sim_cfg["vis_seg_map"],
        sim_mode=sim_cfg["us"],
        us_generative_cfg=us_generative_cfg,
        height=ROBOT_HEIGHT,
        height_img=ROBOT_HEIGHT_IMG,
    )

    default_policy_path = os.path.join(PACKAGE_DIR, "policies", "probe_policy_jit.pt")
    policy_path = args_cli.policy if args_cli.policy is not None else default_policy_path
    probe_policy = FrozenProbePolicy(policy_path=policy_path, device=sim.device)

    print(f"[INFO] Loaded frozen probe policy from: {policy_path}")

    # -------------------------------------------------------------------------
    # State/cache
    # -------------------------------------------------------------------------
    root_pos0 = None
    root_rot0 = None

    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = robot.data.default_joint_vel.clone() * 0.0

    sim_dt = sim.get_physics_dt()
    step_i = 0
    reset_T = float(args_cli.reset_seconds)
    sim_time_acc = 0.0

    R21 = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    RotMat_torch = (
        torch.as_tensor(R21, dtype=torch.float32, device=sim.device)
        .unsqueeze(0)
        .expand(scene.num_envs, -1, -1)
    )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_human_pose_w():
        human_world_poses = human.data.body_link_state_w[:, 0, 0:7]
        return human_world_poses[:, 0:3], human_world_poses[:, 3:7]

    def get_us_ee_pose_w_robot():
        ee_pose_w_robot = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
        return ee_pose_w_robot[:, 0:3], ee_pose_w_robot[:, 3:7]

    def robot_quat_to_us_quat(ee_quat_w: torch.Tensor) -> torch.Tensor:
        ee_rotmat_w = matrix_from_quat(ee_quat_w)
        us_rotmat_w = torch.bmm(ee_rotmat_w, RotMat_torch)
        return quat_from_matrix(us_rotmat_w)

    def us_target_to_robot_target(
        world_to_ee_target_pos: torch.Tensor,
        world_to_ee_target_quat_us: torch.Tensor,
    ):
        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]

        world_to_ee_target_rotmat_us = matrix_from_quat(world_to_ee_target_quat_us)
        world_to_ee_target_rotmat_robot = torch.bmm(
            world_to_ee_target_rotmat_us,
            RotMat_torch.transpose(1, 2),
        )
        world_to_ee_target_quat_robot = quat_from_matrix(world_to_ee_target_rotmat_robot)

        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            world_to_ee_target_quat_robot,
        )

        return base_to_ee_target_pos, base_to_ee_target_quat

    def apply_us_target_differential_ik(
        world_to_ee_target_pos: torch.Tensor,
        world_to_ee_target_quat_us: torch.Tensor,
    ):
        base_to_ee_target_pos, base_to_ee_target_quat = us_target_to_robot_target(
            world_to_ee_target_pos,
            world_to_ee_target_quat_us,
        )

        base_to_ee_target_pose = torch.cat(
            [base_to_ee_target_pos, base_to_ee_target_quat],
            dim=-1,
        )

        pose_diff_ik_controller.set_command(base_to_ee_target_pose)

        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]

        ee_pose_w_robot = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            ee_pose_w_robot[:, 0:3],
            ee_pose_w_robot[:, 3:7],
        )

        US_jacobian = robot.root_physx_view.get_jacobians()[
            :, US_ee_jacobi_idx - 1, :, robot_entity_cfg.joint_ids
        ]

        base_rotmat = matrix_from_quat(quat_inv(world_to_base_pose[:, 3:7]))

        US_jacobian[:, 0:3, :] = torch.bmm(base_rotmat, US_jacobian[:, 0:3, :])
        US_jacobian[:, 3:6, :] = torch.bmm(base_rotmat, US_jacobian[:, 3:6, :])

        US_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        joint_pos_des = pose_diff_ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            US_jacobian,
            US_joint_pos,
        )

        robot.set_joint_position_target(
            joint_pos_des,
            joint_ids=robot_entity_cfg.joint_ids,
        )

    def apply_us_target_pink(
        world_to_ee_target_pos: torch.Tensor,
        world_to_ee_target_quat_us: torch.Tensor,
    ):
        if pin_model is None or pink_cfgs is None or pink_tasks is None:
            raise RuntimeError("Pink was requested but Pink state was not initialized.")

        base_to_ee_target_pos, base_to_ee_target_quat = us_target_to_robot_target(
            world_to_ee_target_pos,
            world_to_ee_target_quat_us,
        )

        target_pos_np = base_to_ee_target_pos.detach().cpu().numpy().astype(np.float64)
        target_rot_np = (
            matrix_from_quat(base_to_ee_target_quat)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        q_all_np = robot.data.joint_pos.detach().cpu().numpy()

        q_next_np = np.zeros(
            (scene.num_envs, len(robot_entity_cfg.joint_ids)),
            dtype=np.float32,
        )

        for env_id in range(scene.num_envs):
            pink_tasks[env_id].set_target(
                pin.SE3(target_rot_np[env_id], target_pos_np[env_id])
            )

            q_pin = isaac_to_pin_q(
                robot.data.joint_pos[env_id],
                isaac_joint_names,
                pin_model,
                name_to_qidx,
            )

            pink_cfgs[env_id].update(q_pin)

            vel = solve_ik(
                pink_cfgs[env_id],
                tasks=[pink_tasks[env_id]],
                dt=sim_dt,
                solver="quadprog",
                damping=5e-2,
                safety_break=False,
            )

            pink_cfgs[env_id].integrate_inplace(vel, sim_dt)

            q_next_pin = pink_cfgs[env_id].q.copy()
            q_next_isaac = pin_to_isaac_q(
                q_next_pin,
                isaac_joint_names,
                name_to_qidx,
            )

            q_next_np[env_id, :] = q_next_isaac[robot_entity_cfg.joint_ids].astype(
                np.float32,
                copy=False,
            )

        q_next_t = torch.from_numpy(q_next_np).to(
            device=sim.device,
            dtype=robot.data.joint_pos.dtype,
        )

        robot.set_joint_position_target(
            q_next_t,
            joint_ids=torch.as_tensor(
                robot_entity_cfg.joint_ids,
                device=sim.device,
                dtype=torch.long,
            ),
        )

    def apply_us_target(
        world_to_ee_target_pos: torch.Tensor,
        world_to_ee_target_quat_us: torch.Tensor,
    ):
        if ik_mode == "pink":
            apply_us_target_pink(
                world_to_ee_target_pos,
                world_to_ee_target_quat_us,
            )
        else:
            apply_us_target_differential_ik(
                world_to_ee_target_pos,
                world_to_ee_target_quat_us,
            )

    def compute_us_tracking_error(
        world_to_ee_target_pos: torch.Tensor,
        world_to_ee_target_quat_us: torch.Tensor,
    ):
        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]

        # target US -> target robot frame
        target_rotmat_us = matrix_from_quat(world_to_ee_target_quat_us)
        target_rotmat_robot = torch.bmm(
            target_rotmat_us,
            RotMat_torch.transpose(1, 2),
        )
        target_quat_robot = quat_from_matrix(target_rotmat_robot)

        target_pos_b, target_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            target_quat_robot,
        )

        ee_pose_w_robot = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            ee_pose_w_robot[:, 0:3],
            ee_pose_w_robot[:, 3:7],
        )

        pos_err = torch.linalg.norm(target_pos_b - ee_pos_b, dim=-1)

        q_err = quat_mul(target_quat_b, quat_inv(ee_quat_b))
        ang_err = 2.0 * torch.acos(torch.clamp(torch.abs(q_err[:, 0]), 0.0, 1.0))
        ang_err_deg = ang_err * 180.0 / torch.pi

        return pos_err, ang_err_deg

    def reset_ik_state():
        pose_diff_ik_controller.reset()

        if ik_mode == "pink" and pink_cfgs is not None:
            for env_id in range(scene.num_envs):
                q_pin = isaac_to_pin_q(
                    robot.data.joint_pos[env_id],
                    isaac_joint_names,
                    pin_model,
                    name_to_qidx,
                )
                pink_cfgs[env_id].update(q_pin)

    def move_towards_target(
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 300,
    ):
        for _ in range(num_steps):
            world_to_human_pos, world_to_human_rot = get_human_pose_w()

            world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
                world_to_human_pos,
                world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            apply_us_target(
                world_ee_target_pos,
                world_ee_target_quat_us,
            )

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def do_reset(log_message: str | None = None):
        nonlocal root_pos0, root_rot0
        nonlocal default_pos, zero_vel
        nonlocal sim_time_acc

        root_state = robot.data.root_state_w.clone()

        if root_pos0 is None:
            root_pos0 = root_state[:, 0:3].clone()
            root_rot0 = root_state[:, 3:7].clone()

        default_pos = robot.data.default_joint_pos.clone()
        zero_vel = robot.data.default_joint_vel.clone() * 0.0

        root_state[:, 0:3] = root_pos0
        root_state[:, 3:7] = root_rot0
        root_state[:, 7:13] *= 0.0

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(default_pos, zero_vel)
        robot.reset()
        scene.reset()

        reset_ik_state()

        US_slicer.current_x_z_x_angle_cmd[:] = (
            init_cmd_pose_min + init_cmd_pose_max
        ) / 2.0

        cmd_target_poses = torch.rand((scene.num_envs, 3), device=sim.device)
        cmd_target_poses = (
            cmd_target_poses * (init_cmd_pose_max - init_cmd_pose_min)
            + init_cmd_pose_min
        )

        US_slicer.current_x_z_x_angle_cmd[:] = cmd_target_poses

        world_to_human_pos, world_to_human_rot = get_human_pose_w()

        world_to_ee_target_pos, world_to_ee_target_quat_us = (
            US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos,
                world_to_human_rot,
            )
        )

        human_to_ee_target_pos, human_to_ee_target_quat = subtract_frame_transforms(
            world_to_human_pos,
            world_to_human_rot,
            world_to_ee_target_pos,
            world_to_ee_target_quat_us,
        )

        US_slicer.human_to_ee_target_pos = human_to_ee_target_pos
        US_slicer.human_to_ee_target_quat = human_to_ee_target_quat

        move_towards_target(
            human_to_ee_target_pos,
            human_to_ee_target_quat,
            num_steps=300,
        )

        US_slicer.current_x_z_x_angle_cmd[:] = cmd_target_poses

        print(f"[RESET] sampled cmd target: {cmd_target_poses.detach().cpu().numpy()}")
        print(
            f"[RESET] current slicer cmd : "
            f"{US_slicer.current_x_z_x_angle_cmd.detach().cpu().numpy()}"
        )
        pos_err0, ang_err0 = compute_us_tracking_error(world_to_ee_target_pos, world_to_ee_target_quat_us)
        print(f"[RESET] after move_towards_target track_pos_err={pos_err0.mean().item():.5f}m "
              f"track_ang_err={ang_err0.mean().item():.2f}deg", flush=True)

        sim_time_acc = 0.0

        if log_message is not None:
            print(log_message)

    def update_us_command_from_probe_policy(obs_img: torch.Tensor):
        """
        Same command update logic used in roboticUSNavigationSurgeryEnv._update_us_command_from_left_policy().
        The probe policy action is interpreted as an incremental command in USSlicer command space:
            [x_cmd, z_cmd, angle_cmd]
        with x/z derived from the current probe local frame expressed in the human frame.
        """

        # Match training script: if observation is 3D, feed only the central slice to probe policy.
        if obs_img.ndim != 4:
            raise ValueError(f"Expected obs_img with shape (N, C, H, W), got {tuple(obs_img.shape)}")

        center_idx = obs_img.shape[1] // 2
        probe_obs_img = obs_img[:, center_idx:center_idx + 1, :, :].detach()

        with torch.no_grad():
            raw_policy_action = probe_policy.act(probe_obs_img)

        # Same scaling/clamping as the first script.
        scale = torch.tensor([0.5, 0.5, 0.04], device=sim.device, dtype=torch.float32).reshape(1, 3)
        amin = -torch.tensor([1.0, 1.0, 0.1], device=sim.device, dtype=torch.float32).reshape(1, 3)
        amax = torch.tensor([1.0, 1.0, 0.1], device=sim.device, dtype=torch.float32).reshape(1, 3)

        policy_action = torch.clamp(raw_policy_action * scale, min=amin, max=amax)

        if robot_type == "h1":
            policy_action = policy_action * 4.0
        if robot_type == "g1":
            policy_action = policy_action * 0.3

        # Current EE pose in WORLD, expressed in US convention.
        world_to_human_pos, world_to_human_rot = get_human_pose_w()

        ee_pos_w, ee_quat_w_robot = get_us_ee_pose_w_robot()
        us_quat_w = robot_quat_to_us_quat(ee_quat_w_robot)

        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            world_to_human_pos,
            world_to_human_rot,
            ee_pos_w,
            us_quat_w,
        )

        human_to_ee_rot_mat = matrix_from_quat(human_to_ee_quat)

        dx_dz_human = (
            policy_action[:, 0].unsqueeze(1) * human_to_ee_rot_mat[:, :, 0]
            + policy_action[:, 1].unsqueeze(1) * human_to_ee_rot_mat[:, :, 1]
        )

        cmd = torch.cat(
            [dx_dz_human[:, [0, 2]], policy_action[:, 2:3]],
            dim=-1,
        )

        # Default: match robotic_US_guidance_G1.py, which uses US_slicer.update_cmd(cmd).
        # Direct mode is kept only for A/B testing.
        prev_cmd = US_slicer.current_x_z_x_angle_cmd.detach().clone()
        unclamped_cmd = prev_cmd + cmd
        if args_cli.cmd_update_mode == "update_cmd":
            US_slicer.update_cmd(cmd)
        else:
            next_cmd_direct = torch.maximum(unclamped_cmd, init_cmd_pose_min)
            next_cmd_direct = torch.minimum(next_cmd_direct, init_cmd_pose_max)
            US_slicer.current_x_z_x_angle_cmd = next_cmd_direct
        next_cmd = US_slicer.current_x_z_x_angle_cmd.detach().clone()
        clamp_mask = (torch.abs(next_cmd - unclamped_cmd) > 1e-6)

        world_to_ee_target_pos, world_to_ee_target_quat_us = (
            US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos,
                world_to_human_rot,
            )
        )

        # Same cache update as in the training env.
        US_slicer.human_to_ee_target_pos, US_slicer.human_to_ee_target_quat = subtract_frame_transforms(
            world_to_human_pos,
            world_to_human_rot,
            world_to_ee_target_pos,
            world_to_ee_target_quat_us,
        )

        return (
            world_to_ee_target_pos,
            world_to_ee_target_quat_us,
            raw_policy_action,
            policy_action,
            cmd,
            prev_cmd,
            unclamped_cmd,
            next_cmd,
            clamp_mask,
            probe_obs_img,
        )
    # -------------------------------------------------------------------------
    # Main loop: apply policy with DirectRLEnv-like decimation
    # -------------------------------------------------------------------------
    world_to_ee_target_pos = None
    world_to_ee_target_quat_us = None
    last_raw_policy_action = None
    last_policy_action = None
    last_cmd_delta = None
    last_prev_cmd = None
    last_next_cmd = None
    last_clamp_mask = None
    last_probe_obs_img = None

    while simulation_app.is_running():

        if step_i == 0:
            do_reset(
                log_message="[INFO] Reset complete. Robot moved to a sampled initial pose."
            )

        if step_i > 0:
            should_control = (step_i % max(1, int(args_cli.control_decimation)) == 0)

            if should_control:
                world_to_human_pos, world_to_human_rot = get_human_pose_w()

                ee_pos_w, ee_quat_w_robot = get_us_ee_pose_w_robot()
                us_quat_w = robot_quat_to_us_quat(ee_quat_w_robot)

                US_slicer.slice_US(
                    world_to_human_pos,
                    world_to_human_rot,
                    ee_pos_w,
                    us_quat_w,
                )

                if sim_cfg["vis_us"]:
                    US_slicer.visualize(key="US", first_n=1)

                obs_img = (
                    US_slicer.us_img_tensor.permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                    * obs_scale
                )

                (
                    world_to_ee_target_pos,
                    world_to_ee_target_quat_us,
                    raw_policy_action,
                    policy_action,
                    cmd_delta,
                    prev_cmd,
                    unclamped_cmd,
                    next_cmd,
                    clamp_mask,
                    probe_obs_img,
                ) = update_us_command_from_probe_policy(obs_img)

                apply_us_target(
                    world_to_ee_target_pos,
                    world_to_ee_target_quat_us,
                )

                last_raw_policy_action = raw_policy_action.detach().clone()
                last_policy_action = policy_action.detach().clone()
                last_cmd_delta = cmd_delta.detach().clone()
                last_prev_cmd = prev_cmd.detach().clone()
                last_next_cmd = next_cmd.detach().clone()
                last_clamp_mask = clamp_mask.detach().clone()
                last_probe_obs_img = probe_obs_img.detach().clone()

                if args_cli.debug_every > 0 and step_i % args_cli.debug_every == 0:
                    pos_err, ang_err = compute_us_tracking_error(
                        world_to_ee_target_pos,
                        world_to_ee_target_quat_us,
                    )
                    cur_cmd = US_slicer.current_x_z_x_angle_cmd
                    dist_to_bounds = torch.minimum(cur_cmd - init_cmd_pose_min, init_cmd_pose_max - cur_cmd)
                    print(
                        f"[DBG {step_i}] IK={ik_mode} control=YES obs_full={tuple(obs_img.shape)} "
                        f"obs_policy={tuple(probe_obs_img.shape)} obs_minmax="
                        f"{probe_obs_img.min().item():.5f}/{probe_obs_img.max().item():.5f} "
                        f"obs_mean={probe_obs_img.mean().item():.5f}",
                        flush=True,
                    )
                    print(f"[DBG {step_i}] raw_action[0]={raw_policy_action[0].detach().cpu().numpy()} "
                          f"scaled_action[0]={policy_action[0].detach().cpu().numpy()} "
                          f"cmd_delta[0]={cmd_delta[0].detach().cpu().numpy()}", flush=True)
                    print(f"[DBG {step_i}] cmd_prev[0]={prev_cmd[0].detach().cpu().numpy()} "
                          f"cmd_next[0]={next_cmd[0].detach().cpu().numpy()} "
                          f"clamped_any={bool(clamp_mask.any().item())} "
                          f"dist_to_bounds_mean={dist_to_bounds.mean(dim=0).detach().cpu().numpy()}", flush=True)
                    print(f"[DBG {step_i}] track_pos_err_mean={pos_err.mean().item():.5f}m "
                          f"track_ang_err_mean={ang_err.mean().item():.2f}deg", flush=True)

        scene.write_data_to_sim()
        sim.step()
        step_i += 1
        scene.update(sim_dt)
        sim_time_acc += sim_dt

        if args_cli.debug_every > 0 and step_i % args_cli.debug_every == 0 and step_i > 0 and world_to_ee_target_pos is not None:
            pos_err, ang_err = compute_us_tracking_error(
                world_to_ee_target_pos,
                world_to_ee_target_quat_us,
            )
            print(
                f"[TRACK {ik_mode}] pos_err={pos_err.mean().item():.4f} m "
                f"ang_err={ang_err.mean().item():.2f} deg "
                f"last_cmd0={(last_next_cmd[0].detach().cpu().numpy() if last_next_cmd is not None else None)}",
                flush=True,
            )

        if reset_T > 0.0 and sim_time_acc >= reset_T:
            do_reset(
                log_message=(
                    f"[INFO] Soft reset at sim_t={step_i * sim_dt:.2f}s. "
                    f"Robot moved to a new sampled initial pose."
                )
            )
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=args_cli.physics_dt, render_interval=max(1, int(args_cli.control_decimation)))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(
        RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    )

    label_map_list = []
    for label_map_file in label_map_file_list:
        label_map = nib.load(label_map_file).get_fdata()
        label_map_list.append(label_map)

    ct_map_list = []
    for ct_map_file in ct_map_file_list:
        ct_map = nib.load(ct_map_file).get_fdata()
        ct_min_max = scene_cfg["sim"]["ct_range"]
        ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
        ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
        ct_map_list.append(ct_map)

    sim.reset()
    print("[INFO] Setup complete. Running…")

    scene.reset()
    run(sim, scene, label_map_list, ct_map_list)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")
    simulation_app.close()