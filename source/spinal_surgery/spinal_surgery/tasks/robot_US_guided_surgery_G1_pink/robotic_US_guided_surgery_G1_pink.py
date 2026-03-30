from __future__ import annotations
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    Articulation,
    RigidObject,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
import nibabel as nib
import cProfile
import time
import numpy as np
from collections.abc import Sequence
import gymnasium as gym
import pyvista as pv
import copy
import os
import sys

# Pink / Pinocchio
import pinocchio as pin
import pink
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pink.solve_ik import solve_ik

##
# Pre-defined configs
##
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
)
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, apply_delta_pose
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR, ASSETS_DATA_DIR
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTMotionGenerator,
    GTDiscreteMotionGenerator,
)
import cProfile
from gymnasium.spaces import Dict
import wandb
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

def build_pinocchio_model(urdf_path: str) -> tuple[pin.Model, list[pin.Data]]:
    # Build Pinocchio model and one Data per env (safe even if you loop sequentially).
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    return model, [model.createData() for _ in range(1)]  # placeholder, overwritten in __init__


def build_name_to_qidx(model: pin.Model) -> dict[str, int]:
    # Map Pinocchio joint name -> starting q index (only 1DoF joints handled here).
    name_to_qidx: dict[str, int] = {}
    for j_id, j in enumerate(model.joints):
        if j.nq == 1:
            name_to_qidx[model.names[j_id]] = j.idx_q
    return name_to_qidx


def isaac_to_pin_q(
    q_isaac: torch.Tensor,
    isaac_joint_names: list[str],
    model: pin.Model,
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    # Build Pinocchio q vector from Isaac joint positions (name-based mapping).
    q_pin = np.zeros(model.nq, dtype=float)
    q_np = q_isaac.detach().cpu().numpy()
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            q_pin[name_to_qidx[jname]] = float(q_np[i])
    return q_pin


def pin_to_isaac_q(
    q_pin: np.ndarray,
    isaac_joint_names: list[str],
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    # Convert Pinocchio q back to Isaac joint ordering (name-based).
    q_isaac = np.zeros(len(isaac_joint_names), dtype=float)
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            q_isaac[i] = float(q_pin[name_to_qidx[jname]])
    return q_isaac

def isaac_np_to_pin_q_fast(
    q_isaac_np: np.ndarray,
    model_nq: int,
    isaac_valid_idx: np.ndarray,
    pin_valid_qidx: np.ndarray,
) -> np.ndarray:
    # Fast Isaac->Pin q fill using precomputed indices (no name loops).
    q_pin = np.zeros((model_nq,), dtype=np.float64)
    q_pin[pin_valid_qidx] = q_isaac_np[isaac_valid_idx].astype(np.float64, copy=False)
    return q_pin


scene_cfg = YAML().load(
    open(
        f"{PACKAGE_DIR}/tasks/robot_US_guided_surgery_G1_pink/cfgs/robotic_US_guided_surgery_G1_pink.yaml",
        "r",
    )
)
# TODO: fix observation scale
if scene_cfg["sim"]["us"] == "net":
    scene_cfg["observation"]["scale"] = scene_cfg["observation"]["scale_net"]
us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
us_generative_cfg = YAML().load(
    open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
)
robot_cfg = scene_cfg["robot"]

# robot type/side from YAML (supports G1 / H1)
robot_type = str(robot_cfg.get("type", "g1")).lower()     # 'g1' or 'h1'
robot_side = str(robot_cfg.get("side", "left")).lower()   # 'left' or 'right'

if robot_type not in ["g1", "h1"]:
    raise ValueError(f"robot.type must be 'g1' or 'h1', got: {robot_type!r}")

# select base articulation config depending on robot type
if robot_type == "g1":
    base_robot_cfg: ArticulationCfg = G1_TOOLS_SURGERY_CFG
else:
    base_robot_cfg: ArticulationCfg = H12_TOOLS_SURGERY_CFG 

# pose is under 'g1:' / 'h1:' section in YAML
robot_pose_cfg = scene_cfg[robot_type]

# build a clean initial state for the selected robot (do NOT mutate the base cfg)
base_pos = (
    float(robot_pose_cfg["pos"][0]),
    float(robot_pose_cfg["pos"][1]),
    float(robot_pose_cfg["pos"][2]),
)

q_xyzw = R.from_euler("z", float(robot_pose_cfg["yaw"]), degrees=True).as_quat()
q_wxyz = (
    float(q_xyzw[3]),
    float(q_xyzw[0]),
    float(q_xyzw[1]),
    float(q_xyzw[2]),
)  # xyzw -> wxyz

# heights for USSlicer (robot-dependent)
ROBOT_HEIGHT = float(robot_pose_cfg["height"])
ROBOT_HEIGHT_IMG = float(robot_pose_cfg["height_img"])

robot_init_state = ArticulationCfg.InitialStateCfg(
    pos=base_pos,
    rot=q_wxyz,
    joint_pos=base_robot_cfg.init_state.joint_pos,
    joint_vel=base_robot_cfg.init_state.joint_vel,
)

robot_articulation_cfg = base_robot_cfg.replace(init_state=robot_init_state)

if robot_type == "g1":
    DRILL_TO_TIP_POS = np.array([0.305, 0.0, 0.0]).astype(np.float32)  # -0.135
elif robot_type == "h1":
    DRILL_TO_TIP_POS = np.array([0.328, 0.0, 0.0]).astype(np.float32)  # -0.135
q_xyzw = R.from_euler("YXZ", [-90, 0, 90], degrees=True).as_quat().astype(np.float32)
DRILL_TO_TIP_QUAT_WXYZ = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

# patient
patient_cfg = scene_cfg["patient"]

# --- H1 patient pose tweak (same as mock) ---
if robot_type == "h1":
    patient_cfg["pos"] = [
        float(patient_cfg["pos"][0]) + 0.00,
        float(patient_cfg["pos"][1]) + 0.0,
        float(patient_cfg["pos"][2]) + 0.00,
    ]
    patient_cfg["euler_yxz"] = [
        float(patient_cfg["euler_yxz"][0]) + 0.0,
        float(patient_cfg["euler_yxz"][1]) + 0.0,
        float(patient_cfg["euler_yxz"][2]) + 0.0,
    ]

quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(float(patient_cfg["pos"][0]), float(patient_cfg["pos"][1]), float(patient_cfg["pos"][2])),
    rot=(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])),
)

# bed
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=(
        float(bed_cfg["pos"][0]),
        float(bed_cfg["pos"][1]),
        float(bed_cfg["pos"][2]),
    ),
    rot=(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])),
)
scale_bed = bed_cfg["scale"]

# use stl: Totalsegmentator_dataset_v2_subset_body_contact
human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id
    for p_id in patient_cfg["id_list"]
]
human_stl_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id
    for p_id in patient_cfg["id_list"]
]
human_raw_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id
    for p_id in patient_cfg["id_list"]
]
target_anatomy = patient_cfg["target_anatomy"]
target_stl_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + str(target_anatomy)
    + ".stl"
    for p_id in patient_cfg["id_list"]
]
target_traj_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + "standard_right_traj_"
    + str(target_anatomy)[-2:]
    + ".stl"
    for p_id in patient_cfg["id_list"]
]

usd_file_list = [
    human_file + "/combined_wrapwrap/combined_wrapwrap.usd"
    for human_file in human_usd_list
]
label_map_file_list = [
    human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list
]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg["label_res"]
scale = 1 / label_res

# Viewer camera
CAMERA_EYE = (-1.145, 0.277, 1.73)
CAMERA_TARGET = (-0.366, -0.229, 1.359)

@configclass
class roboticUSGuidedSurgeryCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = scene_cfg["sim"]["episode_length"]  # 5 # 300
    action_scale = 1
    action_space = 5
    observation_space = [
        us_cfg["image_3D_thickness"] // scene_cfg["observation"]["downsample"],
        200 // scene_cfg["observation"]["downsample"],
        150 // scene_cfg["observation"]["downsample"],
    ]
    state_space = 0
    observation_scale = scene_cfg["observation"]["scale"]

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120, render_interval=decimation
    )

    robot_cfg: ArticulationCfg = robot_articulation_cfg.replace(
        prim_path="/World/envs/env_.*/Robot_US"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=4.0, replicate_physics=False
    )


class roboticUSGuidedSurgeryEnv(DirectRLEnv):
    cfg: roboticUSGuidedSurgeryCfg

    def __init__(
        self, cfg: roboticUSGuidedSurgeryCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Set viewer camera
        if self.sim.has_gui():
            self.sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

        # End-effectors on the same humanoid (US side + drill side)
        UNITREE_EE = {
            "g1": {
                "left":  {"joints": [rf"(waist_(roll|pitch|yaw)_joint|left_(shoulder|elbow|wrist)_.*)"],  "ee": ["left_wrist_yaw_link"]},
                "right": {"joints": [rf"(waist_(roll|pitch|yaw)_joint|right_(shoulder|elbow|wrist)_.*)"], "ee": ["right_wrist_yaw_link"]},
            },
            "h1": {
                "left":  {"joints": [rf"(torso_joint|left_(shoulder|elbow|wrist)_.*)"],  "ee": ["left_wrist_yaw_link"]},
                "right": {"joints": [rf"(torso_joint|right_(shoulder|elbow|wrist)_.*)"], "ee": ["right_wrist_yaw_link"]},
            },
        }

        self.robot_type = robot_type
        self.robot_side = robot_side
        other_side = "right" if self.robot_side == "left" else "left"

        # US EE (selected side)
        us_map = UNITREE_EE[self.robot_type][self.robot_side]
        self.robot_us_entity_cfg = SceneEntityCfg(
            "robot_US",
            joint_names=us_map["joints"],
            body_names=us_map["ee"],
        )
        self.robot_us_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_us_entity_cfg.body_ids[-1]

        # Drill EE (other side)
        dr_map = UNITREE_EE[self.robot_type][other_side]
        self.robot_drill_entity_cfg = SceneEntityCfg(
            "robot_US",
            joint_names=dr_map["joints"],
            body_names=dr_map["ee"],
        )
        self.robot_drill_entity_cfg.resolve(self.scene)
        self.drill_ee_jacobi_idx = self.robot_drill_entity_cfg.body_ids[-1]

        # Full joint set for stacked IK (includes waist + both arms)
        self.full_joint_ids = torch.as_tensor(
            sorted(set(self.robot_us_entity_cfg.joint_ids + self.robot_drill_entity_cfg.joint_ids)),
            device=self.sim.device, dtype=torch.long
        )

        # --
        # Pink / Pinocchio setup (per-env loop)
        # --
        if self.robot_type == "g1":
            self._pink_urdf_path = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
        else:
            self._pink_urdf_path = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_2_handless.urdf"

        self.pin_model = pin.buildModelFromUrdf(self._pink_urdf_path)
        self.pin_datas = [self.pin_model.createData() for _ in range(self.scene.num_envs)]
        self.name_to_qidx = build_name_to_qidx(self.pin_model)

        # Isaac joint names (used for name-based mapping)
        self.isaac_joint_names = list(self.robot.data.joint_names)

        # Fast index maps for Isaac<->Pinocchio (avoid name loops in hot path) 
        n_isaac = len(self.isaac_joint_names)
        self._isaac_to_pin_qidx = np.full((n_isaac,), -1, dtype=np.int32)

        for i, jname in enumerate(self.isaac_joint_names):
            qidx = self.name_to_qidx.get(jname, -1)
            self._isaac_to_pin_qidx[i] = int(qidx) if qidx != -1 else -1

        self._isaac_valid_idx = np.nonzero(self._isaac_to_pin_qidx >= 0)[0].astype(np.int32)
        self._pin_valid_qidx  = self._isaac_to_pin_qidx[self._isaac_valid_idx].astype(np.int32)

        # Cache: full joint ids list (python list is handy for slicing numpy)
        self.full_joint_ids_list = [int(i) for i in self.full_joint_ids.detach().cpu().tolist()]

        # Controlled joints for "US arm only" motions (used by _move_towards_target)
        self._us_ctrl_joint_ids_list = [int(j) for j in self.robot_us_entity_cfg.joint_ids]
        self._us_ctrl_joint_ids_t = torch.as_tensor(self._us_ctrl_joint_ids_list, device=self.sim.device, dtype=torch.long)

        # Precompute pin q-indices for the joint subsets we actually control
        self._pin_qidx_full_ctrl = self._isaac_to_pin_qidx[np.array(self.full_joint_ids_list, dtype=np.int32)]
        if np.any(self._pin_qidx_full_ctrl < 0):
            bad = np.array(self.full_joint_ids_list, dtype=np.int32)[self._pin_qidx_full_ctrl < 0]
            raise RuntimeError(f"[Pink] Some controlled joints are not mapped to Pinocchio q. Isaac joint idx: {bad.tolist()}")

        self._pin_qidx_us_ctrl = self._isaac_to_pin_qidx[np.array(self._us_ctrl_joint_ids_list, dtype=np.int32)]
        if np.any(self._pin_qidx_us_ctrl < 0):
            bad = np.array(self._us_ctrl_joint_ids_list, dtype=np.int32)[self._pin_qidx_us_ctrl < 0]
            raise RuntimeError(f"[Pink] Some US-arm joints are not mapped to Pinocchio q. Isaac joint idx: {bad.tolist()}")

        # Pink configurations (one per env)
        self.pink_cfgs: list[Configuration] = []
        for _ in range(self.scene.num_envs):
            q0 = np.zeros(self.pin_model.nq, dtype=float)
            self.pink_cfgs.append(Configuration(self.pin_model, self.pin_datas[_], q0))

        # Pink tasks (US wrist + drill wrist) in the SAME QP
        self.pink_us_task: list[FrameTask] = []
        self.pink_drill_task: list[FrameTask] = []
        us_frame = self.robot_us_entity_cfg.body_names[-1]      # e.g., left_wrist_yaw_link
        drill_frame = self.robot_drill_entity_cfg.body_names[-1]  # e.g., right_wrist_yaw_link

        assert self.pin_model.existFrame(us_frame), f"Missing frame in Pinocchio: {us_frame}"

        # Pink task frame name for the US EE (must exist in Pinocchio model)
        self._us_ee_frame_name = self.robot_us_entity_cfg.body_names[-1]

        if robot_type == "g1":
            cost_d = 1.0
        elif robot_type == "h1":
            cost_d = 2.0

        for _ in range(self.scene.num_envs):
            self.pink_us_task.append(FrameTask(frame=us_frame, position_cost=1.0, orientation_cost=1.0))
            self.pink_drill_task.append(FrameTask(frame=drill_frame, position_cost=cost_d, orientation_cost=cost_d))

        # Storage for current step targets in BASE
        self._us_target_pos_b = torch.zeros((self.scene.num_envs, 3), device=self.sim.device)
        self._us_target_quat_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device).repeat(self.scene.num_envs, 1)
        self._drill_target_pos_b = torch.zeros((self.scene.num_envs, 3), device=self.sim.device)
        self._drill_target_quat_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device).repeat(self.scene.num_envs, 1)
        
        # ---------------------------------------------------------------------
        # Arm-only joint list for pre-positioning (exclude waist/torso joints)
        # ---------------------------------------------------------------------
        UNITREE_US_ARM_ONLY = {
            "g1": {
                "left":  {"joints": [rf"(left_(shoulder|elbow|wrist)_.*)"],  "ee": ["left_wrist_yaw_link"]},
                "right": {"joints": [rf"(right_(shoulder|elbow|wrist)_.*)"], "ee": ["right_wrist_yaw_link"]},
            },
            "h1": {
                "left":  {"joints": [rf"(torso_joint|left_(shoulder|elbow|wrist)_.*)"],  "ee": ["left_wrist_yaw_link"]},
                "right": {"joints": [rf"(torso_joint|right_(shoulder|elbow|wrist)_.*)"], "ee": ["right_wrist_yaw_link"]},
            },
        }

        us_arm_map = UNITREE_US_ARM_ONLY[self.robot_type][self.robot_side]
        self.robot_us_arm_only_entity_cfg = SceneEntityCfg(
            "robot_US",
            joint_names=us_arm_map["joints"],
            body_names=us_arm_map["ee"],
        )
        self.robot_us_arm_only_entity_cfg.resolve(self.scene)

        # Isaac joint ids (arm-only) for move_towards_target
        self._us_arm_only_ctrl_joint_ids_list = [int(j) for j in self.robot_us_arm_only_entity_cfg.joint_ids]
        self._us_arm_only_ctrl_joint_ids_t = torch.as_tensor(
            self._us_arm_only_ctrl_joint_ids_list, device=self.sim.device, dtype=torch.long
        )

        # Precompute Pinocchio q indices for arm-only controlled joints
        self._pin_qidx_us_arm_only_ctrl = self._isaac_to_pin_qidx[
            np.array(self._us_arm_only_ctrl_joint_ids_list, dtype=np.int32)
        ]

        # rotation: robot EE frame -> US frame (SonoGym convention)
        self.R21 = np.array(
            [
                [0.0, 0.0, 1.0],  # x' = z
                [1.0, 0.0, 0.0],  # y' = x
                [0.0, 1.0, 0.0],  # z' = y
            ],
            dtype=float,
        )

        self.RotMat = (
            torch.as_tensor(self.R21, dtype=torch.float32, device=self.sim.device)
            .unsqueeze(0)
            .expand(self.scene.num_envs, -1, -1)
        )  # (N, 3, 3)

        # load label maps
        label_map_list = []
        for label_map_file in label_map_file_list:
            label_map = nib.load(label_map_file).get_fdata()
            label_map_list.append(label_map)
        # load ct maps
        ct_map_list = []
        for ct_map_file in ct_map_file_list:
            ct_map = nib.load(ct_map_file).get_fdata()
            ct_min_max = scene_cfg["sim"]["ct_range"]
            ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
            ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
            ct_map_list.append(ct_map)

        # construct label image slicer
        label_convert_map = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
        )

        # construct US simulator
        self.sim_cfg = scene_cfg["sim"]
        self.init_cmd_pose_min = (
            torch.tensor(
                self.sim_cfg["patient_xz_init_range"][0], device=self.sim.device
            )
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.init_cmd_pose_max = (
            torch.tensor(
                self.sim_cfg["patient_xz_init_range"][1], device=self.sim.device
            )
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        if scene_cfg["observation"]["3D"]:
            img_thickness = us_cfg["image_3D_thickness"]
        else:
            img_thickness = 1

        # down sample
        res = scene_cfg["observation"]["downsample"]
        us_cfg["image_size"] = [
            int(us_cfg["image_size"][0] / res),
            int(us_cfg["image_size"][1] / res),
        ]
        us_cfg["system_params"]["sx_E"] = us_cfg["system_params"]["sx_E"] / np.sqrt(res)
        us_cfg["system_params"]["sy_E"] = us_cfg["system_params"]["sy_E"] / np.sqrt(res)
        us_cfg["system_params"]["sx_B"] = us_cfg["system_params"]["sx_B"] / np.sqrt(res)
        us_cfg["system_params"]["sy_B"] = us_cfg["system_params"]["sy_B"] / np.sqrt(res)
        us_cfg["system_params"]["I0"] *= np.sqrt(res)
        us_cfg["E_S_ratio"] /= np.sqrt(res)
        img_thickness = max(int(img_thickness // res), 1)
        us_cfg["resolution"] = us_cfg["resolution"] * res

        self.US_slicer = USSlicer(
            us_cfg,
            label_map_list,
            ct_map_list,
            self.sim_cfg["if_use_ct"],
            human_stl_list,
            self.scene.num_envs,
            self.sim_cfg["patient_xz_range"],
            self.sim_cfg["patient_xz_init_range"][0],
            self.sim.device,
            label_convert_map,
            us_cfg["image_size"],
            us_cfg["resolution"],
            img_thickness=img_thickness,
            roll_adj=scene_cfg["motion_planning"]["US_roll_adj"],
            visualize=self.sim_cfg["vis_seg_map"],
            sim_mode=scene_cfg["sim"]["us"],
            us_generative_cfg=us_generative_cfg,
            height=ROBOT_HEIGHT,
            height_img=ROBOT_HEIGHT_IMG,
        )
        self.US_slicer.current_x_z_x_angle_cmd = (
            self.init_cmd_pose_min + self.init_cmd_pose_max
        ) / 2

        self.human_world_poses = (
            self.human.data.root_state_w
        )  # these are already the initial poses

        # construct ground truth motion generator
        motion_plan_cfg = scene_cfg["motion_planning"]
        self.vertebra_viewer = VertebraViewer(
            self.scene.num_envs,
            len(human_usd_list),
            target_stl_file_list,
            target_traj_file_list,
            self.sim_cfg["vis_us"],
            label_res,
            self.sim.device,
        )

        # change observation space to image
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.cfg.observation_space[0], self.cfg.observation_space[1], self.cfg.observation_space[2]),
        #     dtype=np.uint8,
        # )
        # drill rand
        self.rand_joint_pos_max = (
            torch.tensor(motion_plan_cfg["joint_pos_rand_max"])
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
            .to(self.sim.device)
        )

        if robot_type == "h1":
            self.rand_joint_pos_max = self.rand_joint_pos_max[:, 2:]/2  # H1 has no waist_roll


        self.cfg.observation_space[0] = self.US_slicer.img_thickness
        if scene_cfg["sim"]["us"] == "net":
            self.cfg.observation_space[0] = (
                self.cfg.observation_space[0]
                // us_generative_cfg["elevation_downsample"]
            )

        self.single_observation_space["policy"] = Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.cfg.observation_space[0],
                        self.cfg.observation_space[1],
                        self.cfg.observation_space[2],
                    ),
                    dtype=np.uint8,
                ),
                "pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "quat": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        self.observation_space = Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.scene.num_envs,
                        self.cfg.observation_space[0],
                        self.cfg.observation_space[1],
                        self.cfg.observation_space[2],
                    ),
                    dtype=np.uint8,
                ),
                "pos": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.scene.num_envs, 3),
                    dtype=np.float32,
                ),
                "quat": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.scene.num_envs, 4),
                    dtype=np.float32,
                ),
            }
        )

        self.termination_direct = True
        self.observation_mode = scene_cfg["observation"]["mode"]
        self.action_mode = scene_cfg["action"]["mode"]
        self.action_scale = (
            torch.tensor(scene_cfg["action"]["scale"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )


        # for reward
        self.safe_height = scene_cfg["reward"]["safe_height"]
        self.w_pos = scene_cfg["reward"]["w_pos"]
        self.w_angle = scene_cfg["reward"]["w_angle"]
        self.w_cost = scene_cfg["reward"]["w_cost"]
        self.w_insertion = scene_cfg["reward"]["w_insertion"]

        # action scale
        self.max_action = (
            torch.tensor(scene_cfg["action"]["max_action"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.min_action = (
            -torch.tensor(scene_cfg["action"]["max_action"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        # discrete action
        if scene_cfg["action"]["mode"] == "discrete":
            self.single_action_space = gym.spaces.Discrete(
                self.cfg.action_space * 2 + 1
            )
        else:
            self.single_action_space = gym.spaces.Box(
                low=-(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
                high=(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
                shape=(self.cfg.action_space,),
                dtype=np.float32,
            )

        # wandb.init()
        self.num_step = 0

        # reset-time debug logs (async resets -> append per-env scalar entries)
        self._reset_debug = {
            "env_id": [],
            "sim_step": [],
            "us_cmd_x": [],
            "us_cmd_z": [],
            "us_cmd_angle": [],
            "us_roll_adj": [],
            "us_pos_err": [],
            "us_ang_err_deg": [],
            "drill_rand_joint_l2": [],
            "tip_to_traj_dist": [],
            "tip_pos_along_traj": [],
        }

        #  per-step reward component logging (same style as other .pt logs) 
        self.reward_comp_names = [
            "free_progress",
            "close_bonus",
            "close_angle",
            "insertion_progress",
            "unsafe_refund",
            "unsafe_penalty",
            "dist_shaping",
            "angle_shaping",
        ]

        self.reward_comp_trajs = []  # list of (N_envs, C) tensors, appended every step
        self.drill_tip_proj2d_trajs = []
        # per-step IK commanded vs actual EE pose error logging
        # (logged after physics step, saved at reset like other .pt logs)
        self.ik_pose_err_trajs = []  # list of (N_envs, 4) tensors: [us_pos, us_ang_deg, drill_pos, drill_ang_deg]
        self.manipulability_drill = []  # list of (N_envs,) tensors, appended every step
        self.manipulability_US = []
        # infer run mode from entry script name (train.py vs play.py)
        entry = os.path.basename(sys.argv[0]).lower()
        if "play" in entry:
            self._run_mode = "play"
        elif "train" in entry:
            self._run_mode = "train"
        else:
            self._run_mode = "train"
            
        record_rel = str(scene_cfg.get("record_path", "recordings/")).lstrip("/")
        log_name = f"reset_debug_{self._run_mode}.pt"
        self._reset_debug_path = os.path.join(PACKAGE_DIR, record_rel, log_name)

    def _quat_angle_error_deg(self, q_curr, q_tgt):
        q_err = quat_mul(q_tgt, quat_inv(q_curr))      # wxyz
        ang = 2.0 * torch.acos(torch.clamp(torch.abs(q_err[:, 0]), 0.0, 1.0)) * 180.0 / torch.pi
        return ang

    def _safe_normalize(self, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # Normalize vectors safely (batch)
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)

    def _compute_tip_rp_errors(self) -> None:
        """
        Compute tip roll/pitch errors in the trajectory frame.
        - traj frame: z = traj direction, x/y = stable orthonormal basis
        - tip frame: z-axis is the drill axis (your convention)
        - yaw around z (twist) is ignored by taking only roll/pitch from R_rel using ZYX.
        """

        # traj direction in human frame (N,3)
        z_traj = self._safe_normalize(self.vertebra_viewer.traj_drct)

        # build a stable traj frame (x_traj, y_traj, z_traj)
        x0 = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device).reshape(1, 3).repeat(z_traj.shape[0], 1)
        parallel = torch.abs(torch.sum(x0 * z_traj, dim=-1)) > 0.95
        if parallel.any():
            y0 = torch.tensor([0.0, 1.0, 0.0], device=self.sim.device).reshape(1, 3).repeat(z_traj.shape[0], 1)
            x0[parallel] = y0[parallel]

        x_traj = x0 - torch.sum(x0 * z_traj, dim=-1, keepdim=True) * z_traj
        x_traj = self._safe_normalize(x_traj)
        y_traj = torch.cross(z_traj, x_traj, dim=-1)
        y_traj = self._safe_normalize(y_traj)

        # traj rotation matrix in human frame (columns are basis vectors)
        R_traj = torch.stack([x_traj, y_traj, z_traj], dim=-1)  # (N,3,3)

        # tip rotation matrix in human frame
        R_tip = matrix_from_quat(self.human_to_tip_quat)         # (N,3,3)

        # relative rotation expressed in traj frame
        R_rel = torch.bmm(R_traj.transpose(1, 2), R_tip)         # (N,3,3)

        # Extract roll/pitch from ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        # yaw is ignored, but roll/pitch are well-defined for the chosen x/y basis.
        r20 = torch.clamp(-R_rel[:, 2, 0], -1.0, 1.0)
        pitch = torch.asin(r20)                                  # rad
        roll  = torch.atan2(R_rel[:, 2, 1], R_rel[:, 2, 2])       # rad

        self.tip_pitch_err_deg = pitch * (180.0 / torch.pi)
        self.tip_roll_err_deg  = roll  * (180.0 / torch.pi)

    def _flush_reset_debug(self) -> None:
        """Persist reset debug logs to disk (variable-length lists are OK)."""

        out_dir = os.path.dirname(self._reset_debug_path)
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except PermissionError:
                # fallback for clusters / read-only roots
                self._reset_debug_path = os.path.join("/tmp", "reset_debug.pt")
                out_dir = os.path.dirname(self._reset_debug_path)
                os.makedirs(out_dir, exist_ok=True)

        torch.save(self._reset_debug, self._reset_debug_path)


    def get_US_target_pose(self):
        # compute position change
        vertebra_to_US_2d_pos = torch.tensor(
            scene_cfg["motion_planning"]["vertebra_to_US_2d_pos"]
        ).to(self.sim.device)
        rand_disturbance = (
            torch.rand((self.scene.num_envs, 2), device=self.sim.device)
            * 2
            * scene_cfg["motion_planning"]["vertebra_to_US_rand_max"]
            - scene_cfg["motion_planning"]["vertebra_to_US_rand_max"]
        )
        vertebra_2d_pos = self.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
        US_target_2d_pos = (
            vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0) + rand_disturbance
        )

        # change roll adj
        rand_dist_angle = (
            torch.rand((self.scene.num_envs, 1), device=self.sim.device)
            * 2
            * scene_cfg["motion_planning"]["US_roll_rand_max"]
            - scene_cfg["motion_planning"]["US_roll_rand_max"]
        )
        self.US_slicer.roll_adj = (
            scene_cfg["motion_planning"]["US_roll_adj"]
            * torch.ones_like(vertebra_2d_pos[:, 0:1])
            + rand_dist_angle
        )

        rand_dist_angle = (
            torch.rand((self.scene.num_envs, 1), device=self.sim.device)
            * 2
            * scene_cfg["motion_planning"]["US_roll_rand_max"]
            - scene_cfg["motion_planning"]["US_roll_rand_max"]
        )
        US_target_2d_angle = (
            1.57 * torch.ones_like(vertebra_2d_pos[:, 0:1]) + rand_dist_angle
        )

        US_target_2d = torch.cat([US_target_2d_pos, US_target_2d_angle], dim=-1)

        self.US_slicer.current_x_z_x_angle_cmd = US_target_2d
        world_to_ee_init_pos, world_to_ee_init_rot = (
            self.US_slicer.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos, self.world_to_human_rot
            )
        )

    def _setup_scene(self):
        """Configuration for a cart-pole scene."""

        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # lights
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        )
        dome_light_cfg.func("/World/Light", dome_light_cfg)

        # articulation
        # kuka US
        self.robot = Articulation(self.cfg.robot_cfg)

        # medical bad
        if scene_cfg["sim"]["vis_us"]:
            usd_folder = "usd_colored"
        else:
            usd_folder = "usd_no_contact"

        medical_bed_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bed",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/" + usd_folder + "/hospital_bed.usd",
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
        _ = RigidObject(medical_bed_cfg)

        # human (static/kinematic)
        human_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Human",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_file_list,
                random_choice=False,
                scale=(label_res, label_res, label_res),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1.0,
                    max_angular_velocity=1.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=INIT_STATE_HUMAN,
        )
        self.human = RigidObject(human_cfg)

        # assign members
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human

        self.drill_to_tip_pos = (
            torch.tensor(DRILL_TO_TIP_POS, device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.drill_to_tip_quat = (
            torch.tensor(DRILL_TO_TIP_QUAT_WXYZ, device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.tip_to_drill_pos, self.tip_to_drill_quat = subtract_frame_transforms(
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
            torch.zeros_like(self.drill_to_tip_pos).to(self.sim.device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            .to(self.sim.device)
            .repeat(self.scene.num_envs, 1),
        )

    def action_discrete_to_continuous(self, action):
        # action = 0, 1, 2,...,12
        cont_actions = torch.zeros(
            (self.scene.num_envs, self.cfg.action_space), device=self.sim.device
        )
        total_inds = torch.arange(self.scene.num_envs, device=self.sim.device)

        non_zero_inds = total_inds[action.reshape((-1,)) != 10]  # (K_n,)
        non_zero_dim = (action[non_zero_inds] // 2).to(
            torch.int
        )  # 0, 1, 2, 3, 4, 5 (K_n,)

        action_scale = self.max_action[non_zero_inds, :]  # (k_n, 6)
        action_scale = (
            action_scale[
                torch.arange(action_scale.shape[0]).to(self.sim.device), non_zero_dim
            ]
            / 2
        )  # (k_n,)
        action_sign = (action[non_zero_inds] % 2) * 2 - 1  # (k_n, 6)
        cont_actions[non_zero_inds, non_zero_dim] = action_scale * action_sign
        return cont_actions

    def _get_observations(self) -> dict:
        # get human frame
        self.human_world_poses = self.human.data.body_link_state_w[
            :, 0, 0:7
        ]  # these are already the initial pose
        
        # EE pose in WORLD, robot frame (raw from Isaac)
        ee_pose_w_robot = self.robot.data.body_state_w[
            :, self.US_ee_jacobi_idx, 0:7
        ]  # (N, 7)
        # define world to human poses
        ee_pos_w = ee_pose_w_robot[:, 0:3]
        ee_quat_w = ee_pose_w_robot[:, 3:7]

        # rotate orientation robot -> US frame (SonoGym convention)
        ee_rotmat_w = matrix_from_quat(ee_quat_w)
        us_rotmat_w = torch.bmm(ee_rotmat_w, self.RotMat)
        us_quat_w = quat_from_matrix(us_rotmat_w)

        # pose in US frame (for slicer)
        self.US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)
        self.num_step += 1

        if self.observation_mode == "US":
            self.US_slicer.slice_US(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            obs_img = (
                self.US_slicer.us_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
        elif self.observation_mode == "CT":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            obs_img = (
                self.US_slicer.ct_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )

        elif self.observation_mode == "seg":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            obs_img = (
                self.US_slicer.label_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
        else:
            raise ValueError("Invalid observation mode")

        if self.sim_cfg["vis_us"] and self.num_step % self.sim_cfg["vis_int"] == 0:
            self.US_slicer.visualize(self.observation_mode)
            
        self.drill_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_drill_entity_cfg.body_ids[-1], 0:7
        ]

        # get drill to US pose
        self.US_to_drill_pos, self.US_to_drill_quat = subtract_frame_transforms(
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )
        self.US_to_tip_pos, self.US_to_tip_quat = combine_frame_transforms(
            self.US_to_drill_pos,
            self.US_to_drill_quat,
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
        )

        observations = {
            "policy": {
                "image": obs_img,
                "pos": self.US_to_drill_pos,
                "quat": self.US_to_drill_quat,
            }
        }

        # ------------------------------------------------------------
        # Log IK commanded vs actual EE pose error (base frame)
        # ------------------------------------------------------------
        if scene_cfg.get("if_record_traj", False):
            # Actual EE poses in base frame
            self.get_US_ee_pose_b()
            self.get_drill_ee_pose_b()

            # Commanded targets are already stored in base frame:
            #   self._us_target_pos_b, self._us_target_quat_b
            #   self._drill_target_pos_b, self._drill_target_quat_b

            us_pos_err = torch.linalg.norm(self.US_ee_pos_b - self._us_target_pos_b, dim=-1)  # (N,)
            us_ang_err_deg = self._quat_angle_error_deg(self.US_ee_quat_b, self._us_target_quat_b)  # (N,)

            dr_pos_err = torch.linalg.norm(self.drill_ee_pos_b - self._drill_target_pos_b, dim=-1)  # (N,)
            dr_ang_err_deg = self._quat_angle_error_deg(self.drill_ee_quat_b, self._drill_target_quat_b)  # (N,)

            ik_err = torch.stack([us_pos_err, us_ang_err_deg, dr_pos_err, dr_ang_err_deg], dim=-1)  # (N,4)
            self.ik_pose_err_trajs.append(ik_err.detach().cpu())
    
            # manipulability idx
            self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
            # # get joint position targets
            drill_jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.drill_ee_jacobi_idx - 1, :, self.robot_drill_entity_cfg.joint_ids
            ]
            # convert Jacobian from WORLD frame to BASE frame
            base_rotmat = matrix_from_quat(quat_inv(self.world_to_base_pose[:, 3:7]))
            # rotate linear part
            drill_jacobian[:, 0:3, :] = torch.bmm(base_rotmat, drill_jacobian[:, 0:3, :])
            # rotate angular part
            drill_jacobian[:, 3:6, :] = torch.bmm(base_rotmat, drill_jacobian[:, 3:6, :])
            JJT = torch.bmm(drill_jacobian, drill_jacobian.transpose(1, 2))  # (N, 6, 6)
            eps = 1e-12
            JJT = JJT + eps * torch.eye(6, device=JJT.device, dtype=JJT.dtype).unsqueeze(0)
            det_JJT = torch.linalg.det(JJT)
            manipulability_drill = torch.sqrt(torch.clamp(det_JJT, min=0.0))
            self.manipulability_drill.append(manipulability_drill.detach().cpu())

            # # get joint position targets
            US_jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.US_ee_jacobi_idx - 1, :, self.robot_us_entity_cfg.joint_ids
            ]
            # convert Jacobian from WORLD frame to BASE frame
            base_rotmat = matrix_from_quat(quat_inv(self.world_to_base_pose[:, 3:7]))
            # rotate linear part
            US_jacobian[:, 0:3, :] = torch.bmm(base_rotmat, US_jacobian[:, 0:3, :])
            # rotate angular part
            US_jacobian[:, 3:6, :] = torch.bmm(base_rotmat, US_jacobian[:, 3:6, :])
            JJT = torch.bmm(US_jacobian, US_jacobian.transpose(1, 2))  # (N, 6, 6)
            eps = 1e-12
            JJT = JJT + eps * torch.eye(6, device=JJT.device, dtype=JJT.dtype).unsqueeze(0)
            det_JJT = torch.linalg.det(JJT)
            manipulability_US = torch.sqrt(torch.clamp(det_JJT, min=0.0))
            self.manipulability_US.append(manipulability_US.detach().cpu())

        self.check_nan()

        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # control drill robot
        self.get_drill_ee_pose_b()

        if self.action_mode == "discrete":
            actions = self.action_discrete_to_continuous(actions)
        else:
            actions = actions * self.action_scale
            actions = torch.clamp(actions, self.min_action, self.max_action)

        # apply physics constraints
        self.get_traj_to_tip_state()
        too_high = self.tip_pos_along_traj < -0.5
        actions[too_high, :] = 0

        # actions qui sono (N,5): [dx,dy,dz, droll, dpitch]  (yaw rimosso)
        actions5 = actions

        actions6 = torch.zeros((actions5.shape[0], 6), device=actions5.device, dtype=actions5.dtype)
        actions6[:, 0:5] = actions5
        actions6[:, 5] = 0.0  # yaw non attuato

        # action in ee space
        tip_to_next_tip_pos, tip_to_next_tip_quat = apply_delta_pose(
            torch.zeros_like(self.drill_ee_pos_b).to(self.scene.device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            .to(self.scene.device)
            .repeat(self.scene.num_envs, 1),
            actions6,
        )
        tip_pos_b, tip_quat_b = combine_frame_transforms(
            self.drill_ee_pos_b,
            self.drill_ee_quat_b,
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
        )
        next_tip_pos_b, next_tip_quat_b = combine_frame_transforms(
            tip_pos_b, tip_quat_b, tip_to_next_tip_pos, tip_to_next_tip_quat
        )
        # next drill_pos
        drill_next_ee_pos_b, drill_next_ee_quat_b = combine_frame_transforms(
            next_tip_pos_b,
            next_tip_quat_b,
            self.tip_to_drill_pos,
            self.tip_to_drill_quat,
        )

        # Store drill wrist target in BASE for Pink solve (per-env later in _apply_action)
        self._drill_target_pos_b = drill_next_ee_pos_b
        self._drill_target_quat_b = drill_next_ee_quat_b

    def _apply_action(self):
        self.get_drill_ee_pose_b()
        self.get_US_ee_pose_b()
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]

        self._apply_us_command()  # fills _us_target_*_b
        # drill target already cached in _pre_physics_step: _drill_target_*_b

        N = self.scene.num_envs
        dt = float(self.physics_dt)

        # Move targets to CPU ONCE (avoid per-env .cpu().numpy())
        us_t_np = self._us_target_pos_b.detach().cpu().numpy().astype(np.float64, copy=False)         # (N,3)
        us_R_np = matrix_from_quat(self._us_target_quat_b).detach().cpu().numpy().astype(np.float64)  # (N,3,3)

        dr_t_np = self._drill_target_pos_b.detach().cpu().numpy().astype(np.float64, copy=False)
        dr_R_np = matrix_from_quat(self._drill_target_quat_b).detach().cpu().numpy().astype(np.float64)

        # Current joints to CPU ONCE
        q_all_np = self.robot.data.joint_pos.detach().cpu().numpy()  # (N, n_joints)

        # Output as numpy; torch conversion once
        q_next_ctrl_np = np.zeros((N, len(self.full_joint_ids_list)), dtype=np.float32)

        for i in range(N):
            q_pin = isaac_np_to_pin_q_fast(
                q_all_np[i],
                self.pin_model.nq,
                self._isaac_valid_idx,
                self._pin_valid_qidx,
            )
            self.pink_cfgs[i].update(q_pin)

            self.pink_us_task[i].set_target(pin.SE3(us_R_np[i], us_t_np[i]))
            self.pink_drill_task[i].set_target(pin.SE3(dr_R_np[i], dr_t_np[i]))

            vel = solve_ik(
                self.pink_cfgs[i],
                tasks=[self.pink_us_task[i], self.pink_drill_task[i]],
                dt=dt,
                solver="quadprog",
                damping=0.05,
                safety_break=False,
            )
            self.pink_cfgs[i].integrate_inplace(vel, dt)

            q_next_pin = self.pink_cfgs[i].q  # (nq,)
            # Directly read ONLY controlled joints from q_next_pin (no full q_next_isaac)
            q_next_ctrl_np[i, :] = q_next_pin[self._pin_qidx_full_ctrl].astype(np.float32, copy=False)

        q_next_ctrl_t = torch.from_numpy(q_next_ctrl_np).to(
            device=self.sim.device,
            dtype=self.robot.data.joint_pos.dtype,
        )

        # Clamp (recommended; cheap)
        joint_limits = self.robot.data.joint_pos_limits
        joint_ids_t = self.full_joint_ids
        jmin = joint_limits[0, joint_ids_t, 0]
        jmax = joint_limits[0, joint_ids_t, 1]
        safety_margin = 1e-3
        q_next_ctrl_t = torch.clamp(
            q_next_ctrl_t,
            jmin.unsqueeze(0) + safety_margin,
            jmax.unsqueeze(0) - safety_margin,
        )

        self.robot.set_joint_position_target(q_next_ctrl_t, joint_ids=self.full_joint_ids)

    def _apply_us_command(self):
        self.get_US_ee_pose_b()

        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]

        # target EE pose in WORLD (orientation is in US frame)
        world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        # orientation: US frame -> robot frame (inverse of robot->US)
        world_ee_target_rotmat_us = matrix_from_quat(world_ee_target_quat_us)
        world_ee_target_rotmat_robot = torch.bmm(
            world_ee_target_rotmat_us,
            self.RotMat.transpose(1, 2),
        )
        world_ee_target_quat = quat_from_matrix(world_ee_target_rotmat_robot)

        # convert WORLD target to BASE frame for IK
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            self.world_to_base_pose[:, 0:3],
            self.world_to_base_pose[:, 3:7],
            world_ee_target_pos,
            world_ee_target_quat,
        )
        base_to_ee_target_pose = torch.cat(
            [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
        )

        # Cache US wrist target in BASE for Pink solve
        self._us_target_pos_b = base_to_ee_target_pos
        self._us_target_quat_b = base_to_ee_target_quat

    def _get_rewards(self) -> torch.Tensor:
        N = self.scene.num_envs

        # --- visualize tip (optional) ---
        if self.sim_cfg.get("vis_us", False):
            self.vertebra_viewer.update_tip_vis(self.human_to_tip_pos, self.human_to_tip_quat)


        if robot_type == "g1":
            w_i = 0.8
            w_cb = 1
            w_ds = 1
            w_u = 1
            w_o = 1
        elif robot_type == "h1":   
            w_i = 1
            w_cb = 1.2
            w_ds = 2
            w_u = 0
            w_o = 1

        # =========================================================================
        # Regions / masks  (frustum safe_close)
        # =========================================================================
        free_region = self.tip_pos_along_traj <= -self.safe_height
        safety_critical = ~free_region

        s = self.tip_pos_along_traj
        s0 = -self.safe_height
        s1 = self.vertebra_viewer.traj_half_length 

        # progress in [0,1] lungo l'inserzione (solo in safety_critical, ma va bene definirlo per tutti)
        den = (s1 - s0).clamp(min=1e-6)
        t = ((s - s0) / den).clamp(0.0, 1.0)

        # tolleranza extra: 2mm -> 0mm (cono che si stringe)
        extra = -0.002 * t
        r_allowed = self.vertebra_viewer.traj_radius + extra

        # longitudinal zones
        past_goal = s >= s1
        before_goal = ~past_goal

        # -----------------------------
        # Longitudinal overshoot region
        # -----------------------------
        overshoot = safety_critical & past_goal          # oltre half-length (solo quando conta)

        # close: dentro al cono (zona "accettabile" che si stringe)
        close = self.tip_to_traj_dist < r_allowed * 2
        close_rad = self.tip_to_traj_dist < r_allowed + 0.002

        # safe_close: close + vincolo longitudinale (come prima)
        safe_close = safety_critical & close_rad & before_goal

        # unsafe: tutto il resto in safety_critical
        unsafe = (safety_critical & (~safe_close)) 

        # last-step bools
        last_safe_close = self.last_safe_close.to(torch.bool)
        last_unsafe = self.last_unsafe.to(torch.bool)
        last_close = self.last_close.to(torch.bool)
        last_overshoot = self.last_overshoot.to(torch.bool)

        # =========================================================================
        # 1) FREE terms (only in free_region)
        # =========================================================================
        free_progress = torch.zeros(N, device=self.sim.device)
        free_progress[free_region] = 1 * self.w_pos * (
            torch.abs(self.last_tip_pos_along_traj[free_region] + self.safe_height)
            - torch.abs(self.tip_pos_along_traj[free_region] + self.safe_height)
        )

        # =========================================================================
        # 2) CLOSE terms (safe_close + any_close)  -- versione PRECEDENTE
        # =========================================================================
        # monotone insertion tracking
        self.max_tip_pos_along_traj = torch.maximum(self.tip_pos_along_traj, self.max_tip_pos_along_traj)

        insertion_progress = torch.zeros(N, device=self.sim.device)
        
        insertion_progress[safe_close] = w_i * self.w_insertion * (
            torch.abs(self.last_max_tip_pos_along_traj[safe_close] - self.vertebra_viewer.traj_half_length[safe_close])
            - torch.abs(self.max_tip_pos_along_traj[safe_close] - self.vertebra_viewer.traj_half_length[safe_close])
        )
        """
        insertion_progress[safe_close] = self.w_insertion * (
            torch.abs(self.last_tip_pos_along_traj[safe_close] - self.vertebra_viewer.traj_half_length[safe_close])
            - torch.abs(self.tip_pos_along_traj[safe_close] - self.vertebra_viewer.traj_half_length[safe_close])
        )
        """
        # accumulate insertion (kept for compatibility; even if you don't use unsafe_penalty/refund)
        self.total_insertion[safe_close] += (
            torch.abs(self.last_max_tip_pos_along_traj[safe_close] - self.vertebra_viewer.traj_half_length[safe_close])
            - torch.abs(self.max_tip_pos_along_traj[safe_close] - self.vertebra_viewer.traj_half_length[safe_close])
        )


        # any_close = last_close OR close
        any_close = last_close | close

        # --- close bonus PRECEDENTE: step-difference (può essere negativo) ---
        close_bonus = torch.zeros(N, device=self.sim.device)
        close_bonus[any_close] = w_cb * self.w_insertion * (
            self.last_tip_to_traj_dist[any_close] - self.tip_to_traj_dist[any_close]
        )

        angle_small = torch.abs(self.traj_to_tip_sin) < torch.ones_like(self.traj_to_tip_sin) * 0.08  # 0.08 g1

        any_close_angle = any_close | angle_small

        close_angle = torch.zeros(N, device=self.sim.device)
        close_angle[any_close_angle] = 0.15 * self.w_insertion * (   #0.15
            self.last_traj_to_tip_sin[any_close_angle] - self.traj_to_tip_sin[any_close_angle]
        )

        # =========================================================================
        # 3) UNSAFE terms (left as-is; set to 0 if you are not using them)
        # =========================================================================
        last_safe_close_now_unsafe = last_safe_close & unsafe
        last_unsafe_now_safe_close = last_unsafe & safe_close
        last_os_now_chill = last_overshoot & ~overshoot
        last_chill_now_os = ~last_overshoot & overshoot

        """
        penalty_mass = torch.zeros(N, device=self.sim.device)
        penalty_mass[last_safe_close_now_unsafe] = self.total_insertion[last_safe_close_now_unsafe]
        unsafe_penalty = -self.w_cost * penalty_mass * w_u

        unsafe_refund = torch.zeros(N, device=self.sim.device)
        unsafe_refund[last_unsafe_now_safe_close] = 1 * self.w_cost * self.total_insertion[last_unsafe_now_safe_close] * w_u
        """
        unsafe_penalty = torch.zeros(N, device=self.sim.device)#
        unsafe_refund = torch.zeros(N, device=self.sim.device)#


        # --- Overshoot penalty/refund (make scaling w_o apply to ALL envs) ---

        # seg_len can be scalar or per-env; make it (N,) so indexing + scaling is consistent
        seg_len = (s1 - s0)
        if seg_len.ndim == 0:
            seg_len = seg_len * torch.ones((N,), device=self.sim.device)

        unsafe_penalty[last_safe_close_now_unsafe] = -self.w_cost * seg_len[last_safe_close_now_unsafe] * w_u#
        unsafe_refund[last_unsafe_now_safe_close] = self.w_cost * seg_len[last_unsafe_now_safe_close] * w_u#

        os_penalty = torch.zeros(N, device=self.sim.device)
        os_penalty[last_chill_now_os] = -5.0 * seg_len[last_chill_now_os] * w_o

        overshoot_ref = torch.zeros(N, device=self.sim.device)
        overshoot_ref[last_os_now_chill] = +5.0 * seg_len[last_os_now_chill] * w_o

        # keep semantics clean
        unsafe_penalty = unsafe_penalty + os_penalty
        unsafe_refund  = unsafe_refund  + overshoot_ref

        # =========================================================================
        # 4) GLOBAL shaping terms
        # =========================================================================
        dist_shaping = self.w_pos * (self.last_tip_to_traj_dist - self.tip_to_traj_dist) * w_ds
        angle_shaping = self.w_angle * (torch.abs(self.last_traj_to_tip_sin) - torch.abs(self.traj_to_tip_sin))

        # =========================================================================
        # Total reward
        # =========================================================================
        reward = (
            free_progress
            + close_bonus
            + close_angle
            + insertion_progress
            + unsafe_refund
            + unsafe_penalty
            + dist_shaping
            + angle_shaping
        )

        # =========================================================================
        # Metrics / extras
        # =========================================================================
        self.total_dist = torch.sqrt(
            (torch.abs(self.tip_pos_along_traj - self.vertebra_viewer.traj_half_length) ** 2) + self.tip_to_traj_dist
        )
        self.angle = torch.asin(self.traj_to_tip_sin) * 180.0 / torch.pi
        self.insert_err = torch.abs(self.tip_pos_along_traj - self.vertebra_viewer.traj_half_length)
        self.inserted = (self.tip_pos_along_traj > -self.vertebra_viewer.traj_half_length).reshape((-1,))

        self.total_rewards += reward
        self.extras["cost"] = unsafe.to(torch.float32)
        self.total_costs += self.extras["cost"]

        ones = torch.ones((N,), device=self.sim.device)
        self.extras["traj_drct"] = self.vertebra_viewer.traj_drct
        self.extras["human_to_tip_pos"] = self.human_to_tip_pos
        self.extras["human_to_tip_quat"] = self.human_to_tip_quat
        self.extras["safe_height"] = self.safe_height * ones
        self.extras["traj_half_length"] = self.vertebra_viewer.traj_half_length
        self.extras["traj_radius"] = self.vertebra_viewer.traj_radius
        self.extras["tip_to_traj_dist"] = self.tip_to_traj_dist
        self.extras["traj_to_tip_sin"] = self.traj_to_tip_sin
        self.extras["human_to_traj_pos"] = self.vertebra_viewer.human_to_traj_pos
        self.extras["tip_pos_along_traj"] = self.tip_pos_along_traj

        #  per-step trajectory + reward-components logging 
        if scene_cfg.get("if_record_traj", False):
            if not hasattr(self, "reward_comp_trajs"):
                self.reward_comp_trajs = []
            if not hasattr(self, "tip_pos_along_traj_trajs"):
                self.tip_pos_along_traj_trajs = []
            if not hasattr(self, "tip_to_traj_dist_trajs"):
                self.tip_to_traj_dist_trajs = []
            if not hasattr(self, "tip_roll_err_deg_trajs"):
                self.tip_roll_err_deg_trajs = []
            if not hasattr(self, "tip_pitch_err_deg_trajs"):
                self.tip_pitch_err_deg_trajs = []
            if not hasattr(self, "drill_tip_proj2d_trajs"):
                self.drill_tip_proj2d_trajs = []

            self.tip_pos_along_traj_trajs.append(self.tip_pos_along_traj)
            self.tip_to_traj_dist_trajs.append(self.tip_to_traj_dist)
            self.tip_roll_err_deg_trajs.append(self.tip_roll_err_deg)
            self.tip_pitch_err_deg_trajs.append(self.tip_pitch_err_deg)
            self.drill_tip_proj2d_trajs.append(self.drill_tip_proj2d)

            comps = torch.stack(
                [
                    free_progress,
                    close_bonus,
                    close_angle,
                    insertion_progress,
                    unsafe_refund,
                    unsafe_penalty,
                    dist_shaping,
                    angle_shaping,
                ],
                dim=-1,
            )
            self.reward_comp_trajs.append(comps.detach().cpu())

        # =========================================================================
        # Update last-step buffers
        # =========================================================================
        self.last_tip_pos_along_traj = self.tip_pos_along_traj.clone()
        self.last_tip_to_traj_dist = self.tip_to_traj_dist.clone()
        self.last_traj_to_tip_sin = self.traj_to_tip_sin.clone()
        self.last_unsafe = unsafe.clone()
        self.last_safe_close = safe_close.clone()
        self.last_close = close.clone()
        self.last_max_tip_pos_along_traj = self.max_tip_pos_along_traj.clone()
        self.last_tip_pos_along_traj = self.tip_pos_along_traj.clone()
        self.last_overshoot = overshoot.clone() 

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time limit
        if self.termination_direct:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros_like(self.episode_length_buf)

        out_of_bounds = torch.zeros_like(self.US_slicer.no_collide)
        return out_of_bounds, time_out

    def _move_towards_target(
        self,
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 200,
    ):
        """Move ONLY the US arm EE towards a target pose expressed in the HUMAN frame (US convention).
        """
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Convenience
        N = self.scene.num_envs
        joint_ids_t = self._us_arm_only_ctrl_joint_ids_t
        ctrl_ids_list = self._us_arm_only_ctrl_joint_ids_list
        n_ctrl = len(ctrl_ids_list)

        # Create per-env Pink tasks for the US EE frame (same frame name)
        if not hasattr(self, "_pink_us_only_tasks"):
            self._pink_us_only_tasks = []
            for _ in range(N):
                self._pink_us_only_tasks.append(
                    FrameTask(frame=self._us_ee_frame_name, position_cost=1.0, orientation_cost=1.0)
                )

        dt = float(self.physics_dt)

        for _ in range(num_steps):
            self._sim_step_counter += 1

            # 1) HUMAN pose in WORLD
            self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
            self.world_to_human_pos = self.human_world_poses[:, 0:3]
            self.world_to_human_rot = self.human_world_poses[:, 3:7]

            # 2) Target EE in WORLD (orientation in US frame)
            world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
                self.world_to_human_pos,
                self.world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # US frame -> robot EE frame (undo robot->US)
            world_ee_target_rot_us = matrix_from_quat(world_ee_target_quat_us)
            world_ee_target_rot_robot = torch.bmm(world_ee_target_rot_us, self.RotMat.transpose(1, 2))
            world_ee_target_quat = quat_from_matrix(world_ee_target_rot_robot)

            # 3) Convert WORLD target -> BASE target
            world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
            base_pos_w  = world_to_base_pose[:, 0:3]
            base_quat_w = world_to_base_pose[:, 3:7]

            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_ee_target_pos,
                world_ee_target_quat,
            )

            # Precompute targets to CPU once per step
            t_all_np = base_to_ee_target_pos.detach().cpu().numpy().astype(np.float64, copy=False)
            R_all_np = matrix_from_quat(base_to_ee_target_quat).detach().cpu().numpy().astype(np.float64)

            q_all_np = self.robot.data.joint_pos.detach().cpu().numpy()

            arm_q_np = np.zeros((N, n_ctrl), dtype=np.float32)

            for env_id in range(N):
                task = self._pink_us_only_tasks[env_id]
                task.set_target(pin.SE3(R_all_np[env_id], t_all_np[env_id]))

                q_pin = isaac_np_to_pin_q_fast(
                    q_all_np[env_id], self.pin_model.nq, self._isaac_valid_idx, self._pin_valid_qidx
                )
                self.pink_cfgs[env_id].update(q_pin)

                vel = solve_ik(
                    self.pink_cfgs[env_id],
                    tasks=[task],
                    dt=dt,
                    solver="quadprog",
                    damping=0.05,
                    safety_break=False,
                )

                self.pink_cfgs[env_id].integrate_inplace(vel, dt)

                q_next_pin = self.pink_cfgs[env_id].q
                arm_q_np[env_id, :] = q_next_pin[self._pin_qidx_us_arm_only_ctrl].astype(np.float32, copy=False)

            arm_q_t = torch.from_numpy(arm_q_np).to(device=self.sim.device, dtype=self.robot.data.joint_pos.dtype)

            # Clamp to joint limits (arm-only)
            joint_limits = self.robot.data.joint_pos_limits
            joint_min = joint_limits[0, joint_ids_t, 0]
            joint_max = joint_limits[0, joint_ids_t, 1]
            safety_margin = 1e-2
            arm_q_t = torch.clamp(arm_q_t, joint_min.unsqueeze(0) + safety_margin, joint_max.unsqueeze(0) - safety_margin)

            self.robot.set_joint_position_target(arm_q_t, joint_ids=joint_ids_t)

            # 7) Step sim
            self.scene.write_data_to_sim()
            self.sim.step(render=False)

            if (self._sim_step_counter % self.cfg.sim.render_interval == 0) and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

    def get_US_ee_pose_b(self):
        # get ee pose in base frame
        self.US_root_pose_w = self.robot.data.root_state_w[:, 0:7]

        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_us_entity_cfg.body_ids[-1], 0:7
        ]
        # compute frame in root frame
        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            self.US_root_pose_w[:, 0:3],
            self.US_root_pose_w[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )

    def get_drill_ee_pose_b(self):
        self.drill_root_pose_w = self.robot.data.root_state_w[:, 0:7]

        self.drill_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_drill_entity_cfg.body_ids[-1], 0:7
        ]
        # compute frame in root frame
        self.drill_ee_pos_b, self.drill_ee_quat_b = subtract_frame_transforms(
            self.drill_root_pose_w[:, 0:3],
            self.drill_root_pose_w[:, 3:7],
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )

    def get_traj_to_tip_state(self):
        # Assicurati di avere questi aggiornati (se non lo sono già)
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos = self.human_world_poses[:, 0:3]
        self.world_to_human_rot = self.human_world_poses[:, 3:7]

        self.drill_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_drill_entity_cfg.body_ids[-1], 0:7
        ]

        # HUMAN -> DRILL_EE
        self.human_to_ee_pos, self.human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )

        # HUMAN -> TIP = (HUMAN->EE) o (EE->TIP)
        self.human_to_tip_pos, self.human_to_tip_quat = combine_frame_transforms(
            self.human_to_ee_pos,
            self.human_to_ee_quat,
            self.drill_to_tip_pos,
            self.drill_to_tip_quat,
        )

        self.tip_pos_along_traj, self.tip_to_traj_dist, self.traj_to_tip_sin = (
            self.vertebra_viewer.compute_tip_in_traj(
                self.human_to_tip_pos, self.human_to_tip_quat
            )
        )

        self._compute_tip_rp_errors()

        #  drill tip projection in 2D plane perpendicular to PROBE TARGET axis 
        # probe target axis = z of target orientation (in human frame)
        R_probe_tgt = matrix_from_quat(self.US_slicer.human_to_ee_target_quat)   # (N,3,3)
        z_axis = self._safe_normalize(R_probe_tgt[:, :, 2])                     # (N,3)

        x0 = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device).reshape(1, 3).repeat(z_axis.shape[0], 1)
        parallel = torch.abs(torch.sum(x0 * z_axis, dim=-1)) > 0.95
        if parallel.any():
            x0[parallel] = torch.tensor([0.0, 1.0, 0.0], device=self.sim.device).reshape(1, 3).repeat(parallel.sum(), 1)

        x_axis = x0 - torch.sum(x0 * z_axis, dim=-1, keepdim=True) * z_axis
        x_axis = self._safe_normalize(x_axis)
        y_axis = self._safe_normalize(torch.cross(z_axis, x_axis, dim=-1))

        # origin = probe TARGET position in human frame
        p = self.human_to_tip_pos - self.US_slicer.human_to_ee_target_pos        # (N,3)
        proj_x = torch.sum(p * x_axis, dim=-1)
        proj_y = torch.sum(p * y_axis, dim=-1)
        self.drill_tip_proj2d = torch.stack([proj_x, proj_y], dim=-1)            # (N,2)

    def reset_controllers(self):
        # Reset Pink configs to match current Isaac state (per-env)
        q_all = self.robot.data.joint_pos
        for i in range(self.scene.num_envs):
            q_pin = isaac_to_pin_q(q_all[i], self.isaac_joint_names, self.pin_model, self.name_to_qidx)
            self.pink_cfgs[i].update(q_pin)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.sim_cfg.get("vis_us", False) and hasattr(self, "vertebra_viewer"):
            env_ids_list = [int(e) for e in env_ids]
            if 0 in env_ids_list:
                self.vertebra_viewer.us_frame_idx = 0

                curr_dir = self.vertebra_viewer.us_record_dir
                base_dir = curr_dir.rstrip("/")

                if "_" in base_dir and base_dir.split("_")[-1].isdigit():
                    prefix = "_".join(base_dir.split("_")[:-1])
                    ep_idx = int(base_dir.split("_")[-1]) + 1
                else:
                    prefix = base_dir
                    ep_idx = 1

                self.vertebra_viewer.us_record_dir = f"{prefix}_{ep_idx}"
                os.makedirs(self.vertebra_viewer.us_record_dir, exist_ok=True)

        if hasattr(self, "total_rewards"):
            wandb.log({"total_rewards": self.total_rewards.mean().item()})

        # reconstruct random maps
        self.US_slicer.construct_T_maps()
        self.US_slicer.construct_Vl_maps()

        #  reset single articulation joint state
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        # randomize drill-chain joints only (keep US chain at default)
        rand_joint_pos = (
            torch.rand(
                (self.scene.num_envs, len(self.robot_drill_entity_cfg.joint_ids)),
                device=self.sim.device,
            )
            * 2
            - 1
        )
        # if your YAML gives per-joint max, keep this; otherwise set a scalar
        rand_joint_pos = rand_joint_pos * self.rand_joint_pos_max

        drill_ids = torch.as_tensor(
            self.robot_drill_entity_cfg.joint_ids,
            device=self.sim.device,
            dtype=torch.long,
        )
        joint_pos[:, drill_ids] = joint_pos[:, drill_ids] + rand_joint_pos

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()

        # single target application (union of both chains)
        self.robot.set_joint_position_target(
            joint_pos[:, self.full_joint_ids], joint_ids=self.full_joint_ids
        )

        self.reset_controllers()

        # inverse kinematics?
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        # get human frame
        self.human_world_poses = self.human.data.body_link_state_w[
            :, 0, 0:7
        ]  # these are already the initial poses
        # define world to human poses
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]

        # compute 2d target poses
        # compute ultrasound target pose
        self.get_US_target_pose()
        # compute joint positions
        # set joint positions
        self._move_towards_target(
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        #  reset debug: final US tracking error (after pre-positioning) 
        # current EE in base frame
        self.get_US_ee_pose_b()

        # target EE pose in WORLD (US frame orientation), then convert to base frame
        world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        # US frame -> robot EE frame
        world_ee_target_rotmat_us = matrix_from_quat(world_ee_target_quat_us)
        world_ee_target_rotmat_robot = torch.bmm(
            world_ee_target_rotmat_us,
            self.RotMat.transpose(1, 2),
        )
        world_ee_target_quat = quat_from_matrix(world_ee_target_rotmat_robot)

        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            self.world_to_base_pose[:, 0:3],
            self.world_to_base_pose[:, 3:7],
            world_ee_target_pos,
            world_ee_target_quat,
        )

        us_pos_err = torch.norm(base_to_ee_target_pos - self.US_ee_pos_b, dim=-1)
        us_ang_err_deg = self._quat_angle_error_deg(self.US_ee_quat_b, base_to_ee_target_quat)

        # log current US cmd (x, z, angle) and roll_adj
        us_cmd = self.US_slicer.current_x_z_x_angle_cmd
        us_roll_adj = self.US_slicer.roll_adj

        # log drill randomization magnitude (only for the drill chain)
        drill_rand_joint_l2 = torch.norm(rand_joint_pos, dim=-1)

        self.get_drill_ee_pose_b()
        self.get_traj_to_tip_state()

        # log trajectory-related initial state after reset
        tip_to_traj_dist = self.tip_to_traj_dist.detach()
        tip_pos_along_traj = self.tip_pos_along_traj.detach()

        self.last_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)

        self.max_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)
        self.last_max_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)
        self.last_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)   
        

        # if hasattr(self, "total_rewards") and torch.abs(self.total_rewards.mean()) > 0:
        #     wandb.log({"total_reward": self.total_rewards.mean().item()})
        #     wandb.log({"total_insertion": self.total_insertion.mean().item()})
        #     wandb.log({"last_sin": self.last_traj_to_tip_sin.mean().item()})
        #     wandb.log({"last_dist": self.last_tip_to_traj_dist.mean().item()})
        #     wandb.log({"last_angle": self.angle.mean().item()})
        #     wandb.log({"last_total_dist": self.total_dist.mean().item()})
        #     wandb.log({"last_insert_err": self.insert_err[self.inserted].mean().item()})

        self.last_tip_to_traj_dist = copy.deepcopy(self.tip_to_traj_dist)
        self.last_total_dist = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_traj_to_tip_sin = copy.deepcopy(self.traj_to_tip_sin)
        self.last_unsafe = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_safe_close = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_insertion = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_traj_pos_along_traj_safe_close = torch.ones(
            self.scene.num_envs, device=self.sim.device
        ) * (-self.safe_height)


        # if hasattr(self, "total_costs") and torch.abs(self.total_rewards.mean()) > 0:
        #     wandb.log({"total_cost": self.total_costs.mean().item()})
        self.total_rewards = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.total_costs = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_close = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.last_overshoot = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.sim.device)

        # record information
        ones = torch.ones((self.scene.num_envs,), device=self.sim.device)
        self.extras["traj_drct"] = self.vertebra_viewer.traj_drct
        self.extras["human_to_tip_pos"] = self.human_to_tip_pos
        self.extras["human_to_tip_quat"] = self.human_to_tip_quat
        self.extras["safe_height"] = self.safe_height * ones
        self.extras["traj_half_length"] = self.vertebra_viewer.traj_half_length
        self.extras["traj_radius"] = self.vertebra_viewer.traj_radius
        self.extras["tip_to_traj_dist"] = self.tip_to_traj_dist
        self.extras["traj_to_tip_sin"] = self.traj_to_tip_sin
        self.extras["human_to_traj_pos"] = self.vertebra_viewer.human_to_traj_pos
        self.extras["tip_pos_along_traj"] = self.tip_pos_along_traj
        self.extras["cost"] = torch.zeros(self.scene.num_envs, device=self.sim.device)      

        #  append reset debug per env id (async-safe) 
        for _eid in env_ids:
            i = int(_eid)
            self._reset_debug["env_id"].append(i)
            self._reset_debug["sim_step"].append(int(self._sim_step_counter))
            self._reset_debug["us_cmd_x"].append(float(us_cmd[i, 0].detach().cpu()))
            self._reset_debug["us_cmd_z"].append(float(us_cmd[i, 1].detach().cpu()))
            self._reset_debug["us_cmd_angle"].append(float(us_cmd[i, 2].detach().cpu()))
            self._reset_debug["us_roll_adj"].append(float(us_roll_adj[i, 0].detach().cpu()))
            self._reset_debug["us_pos_err"].append(float(us_pos_err[i].detach().cpu()))
            self._reset_debug["us_ang_err_deg"].append(float(us_ang_err_deg[i].detach().cpu()))
            self._reset_debug["drill_rand_joint_l2"].append(float(drill_rand_joint_l2[i].detach().cpu()))
            self._reset_debug["tip_to_traj_dist"].append(float(tip_to_traj_dist[i].detach().cpu()))
            self._reset_debug["tip_pos_along_traj"].append(float(tip_pos_along_traj[i].detach().cpu()))

        # persist frequently: reset events are sparse vs sim steps
        self._flush_reset_debug()


        if scene_cfg["if_record_traj"]:
            record_path = PACKAGE_DIR + scene_cfg["record_path"]
            if hasattr(self, "tip_pos_along_traj_trajs"):
                if not os.path.exists(record_path):
                    os.makedirs(record_path)
                # Stack over time: [N_envs, T]  (time is dim=1)
                self.tip_pos_along_traj_trajs = torch.stack(self.tip_pos_along_traj_trajs, dim=1)
                torch.save(self.tip_pos_along_traj_trajs, record_path + "tip_pos_along_traj.pt")

                self.tip_to_traj_dist_trajs = torch.stack(self.tip_to_traj_dist_trajs, dim=1)
                torch.save(self.tip_to_traj_dist_trajs, record_path + "tip_to_traj_dist.pt")

                self.tip_roll_err_deg_trajs = torch.stack(self.tip_roll_err_deg_trajs, dim=1)
                torch.save(self.tip_roll_err_deg_trajs, record_path + "tip_roll_err_deg.pt")

                self.tip_pitch_err_deg_trajs = torch.stack(self.tip_pitch_err_deg_trajs, dim=1)
                torch.save(self.tip_pitch_err_deg_trajs, record_path + "tip_pitch_err_deg.pt")

            self.tip_pos_along_traj_trajs = [self.tip_pos_along_traj]
            self.tip_to_traj_dist_trajs = [self.tip_to_traj_dist]
            self.tip_roll_err_deg_trajs = [self.tip_roll_err_deg]
            self.tip_pitch_err_deg_trajs = [self.tip_pitch_err_deg]

            # Save reward components: (N_envs, T, C)
            if hasattr(self, "reward_comp_trajs") and len(self.reward_comp_trajs) > 0:
                reward_comp_tensor = torch.stack(self.reward_comp_trajs, dim=1)  # (N, T, C)
                torch.save(
                    {
                        "components": reward_comp_tensor,
                        "names": self.reward_comp_names,
                    },
                    record_path + "reward_components.pt",
                )
            self.reward_comp_trajs = []
            # Save drill tip projection in probe-target plane: (N_envs, T, 2)
            if hasattr(self, "drill_tip_proj2d_trajs") and len(self.drill_tip_proj2d_trajs) > 0:
                drill_tip_proj2d_trajs = torch.stack(self.drill_tip_proj2d_trajs, dim=1)  # (N,T,2)
                torch.save(drill_tip_proj2d_trajs, record_path + "drill_tip_proj2d.pt")
            
            self.drill_tip_proj2d_trajs = [self.drill_tip_proj2d]
            # Save IK pose tracking error: (N_envs, T, 4)
            # channels: [us_pos_err, us_ang_err_deg, drill_pos_err, drill_ang_err_deg]
            if hasattr(self, "ik_pose_err_trajs") and len(self.ik_pose_err_trajs) > 0:
                ik_pose_err_tensor = torch.stack(self.ik_pose_err_trajs, dim=1)  # (N, T, 4)
                torch.save(
                    {
                        "errors": ik_pose_err_tensor,
                        "names": ["us_pos_err", "us_ang_err_deg", "drill_pos_err", "drill_ang_err_deg"],
                    },
                    record_path + "ik_pose_error.pt",
                )
            self.ik_pose_err_trajs = []

            if hasattr(self, "manipulability_drill") and len(self.manipulability_drill) > 0:
                manipulability_tensor_drill = torch.stack(self.manipulability_drill, dim=1)  # (N, T)
                torch.save(manipulability_tensor_drill, record_path + "yoshikawa_manipulability_drill.pt")
            self.manipulability_drill = []
            if hasattr(self, "manipulability_US") and len(self.manipulability_US) > 0:
                manipulability_tensor_US = torch.stack(self.manipulability_US, dim=1)  # (N, T)
                torch.save(manipulability_tensor_US, record_path + "yoshikawa_manipulability_us.pt")
            self.manipulability_US = []


    def check_nan(self):
        if torch.isnan(self.US_ee_pos_b).any() or torch.isnan(self.US_ee_quat_b).any():
            print("US_ee_pos_b", self.US_ee_pos_b)
            print("US_ee_quat_b", self.US_ee_quat_b)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.drill_ee_pos_b).any()
            or torch.isnan(self.drill_ee_quat_b).any()
        ):
            print("drill_ee_pos_b", self.drill_ee_pos_b)
            print("drill_ee_quat_b", self.drill_ee_quat_b)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.human_to_tip_pos).any()
            or torch.isnan(self.human_to_tip_quat).any()
        ):
            print("human_to_tip_pos", self.human_to_tip_pos)
            print("human_to_tip_quat", self.human_to_tip_quat)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.tip_pos_along_traj).any()
            or torch.isnan(self.tip_to_traj_dist).any()
            or torch.isnan(self.traj_to_tip_sin).any()
        ):
            print("tip_pos_along_traj", self.tip_pos_along_traj)
            print("tip_to_traj_dist", self.tip_to_traj_dist)
            print("traj_to_tip_sin", self.traj_to_tip_sin)
            raise ValueError("nan value detected")
        if torch.isnan(self.total_rewards).any() or torch.isnan(self.total_costs).any():
            print("total_rewards", self.total_rewards)
            print("total_costs", self.total_costs)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.last_tip_pos_along_traj).any()
            or torch.isnan(self.last_tip_to_traj_dist).any()
            or torch.isnan(self.last_traj_to_tip_sin).any()
        ):
            print("last_tip_pos_along_traj", self.last_tip_pos_along_traj)
            print("last_tip_to_traj_dist", self.last_tip_to_traj_dist)
            print("last_traj_to_tip_sin", self.last_traj_to_tip_sin)
            raise ValueError("nan value detected")
        if torch.isnan(self.US_slicer.ct_img_tensor).any():
            print("ct_img_tensor", self.US_slicer.human_to_ee_target_pos)
            raise ValueError("nan value detected")
        if torch.isnan(self.US_slicer.label_img_tensor).any():
            print("label_img_tensor", self.US_slicer.human_to_ee_target_pos)
            raise ValueError("nan value detected")
        if (
            torch.isnan(self.US_to_drill_pos).any()
            or torch.isnan(self.US_to_drill_quat).any()
        ):
            print("US_to_drill_pos", self.US_to_drill_pos)
            print("US_to_drill_quat", self.US_to_drill_quat)
            raise ValueError("nan value detected")
