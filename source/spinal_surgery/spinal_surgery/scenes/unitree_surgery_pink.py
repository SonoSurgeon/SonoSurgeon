# guided_surgery_mock_bimanual_pink_qp_g1h1.py
# Copyright...
# SPDX-License-Identifier: BSD-3-Clause

"""Mock bimanual IK for SonoGym guided surgery with Unitree G1/H1 using Pink (single QP, two FrameTasks).

- Left arm: tracks a US-derived target pose (via USSlicer.human_to_ee_target_*) using LEFT_EE_NAME.
- Right arm: tracks a moving placeholder TIP target (base frame), converted to desired RIGHT wrist pose via fixed wrist->tip offset.
- One Pink solve_ik call per step, with TWO FrameTasks in the SAME QP.

Launch Isaac Sim first.

Usage:
  ./isaaclab.sh -p path/to/guided_surgery_mock_bimanual_pink_qp_g1h1.py --num_envs 1
"""

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="SonoGym mock guided surgery (bimanual Pink IK, G1/H1 switch).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--reset_seconds",
    type=float,
    default=5.0,
    help="Soft reset every X seconds of *sim time* (<=0 disables).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
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
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_mul,
)

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import nibabel as nib
import os
import cProfile
import matplotlib.pyplot as plt
import copy

# Pink / Pinocchio
import pinocchio as pin
import pink
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pink.solve_ik import solve_ik

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
# YAML scene config
# -----------------------------------------------------------------------------

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))
sim_cfg = scene_cfg["sim"]
motion_plan_cfg = scene_cfg["motion_planning"]
patient_cfg = scene_cfg["patient"]
target_anatomy = patient_cfg["target_anatomy"]

# patient pose
patient_cfg = scene_cfg["patient"]

# bed pose
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)
scale_bed = bed_cfg["scale"]

# datasets (human USD/labels/CT)
patient_ids = patient_cfg["id_list"]
human_usd_list = [f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id for p_id in patient_ids]
human_stl_list = [f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id for p_id in patient_ids]
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
scale = 1.0 / float(label_res)

# robot selector
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")  # default: g1

if robot_type == "h1":
    # esempio: sposta il paziente (modifica questi offset come ti serve)
    patient_cfg["pos"] = [
        float(patient_cfg["pos"][0]) + 0.0,   # x
        float(patient_cfg["pos"][1]) + 0.0,   # y
        float(patient_cfg["pos"][2]) + 0.0,   # z
    ]
    patient_cfg["euler_yxz"] = [    
        float(patient_cfg["euler_yxz"][0]) + 0.0,   #
        float(patient_cfg["euler_yxz"][1]) + 0.0,   #
        float(patient_cfg["euler_yxz"][2]) + 0.0,   #
    ]

quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)


LEFT_EE_NAME = "left_wrist_yaw_link"
RIGHT_EE_NAME = "right_wrist_yaw_link"

if robot_type == "g1":
    ROBOT_CFG: ArticulationCfg = G1_TOOLS_SURGERY_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/G1"
    ROBOT_KEY = "g1"
    # TODO: set the URDF you actually use for Pink
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif robot_type == "h1":
    ROBOT_CFG: ArticulationCfg = H12_TOOLS_SURGERY_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/H1"
    ROBOT_KEY = "h1"
    # TODO: set the URDF you actually use for Pink
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_2_handless.urdf"
else:
    raise ValueError(f"Unknown robot type in YAML: {robot_type!r} (expected 'g1' or 'h1').")

# Override init pose + orientation from YAML robot section
pos_init = scene_cfg[robot_type]["pos"]
ROBOT_CFG.init_state.pos = pos_init
q_xyzw = R.from_euler("z", scene_cfg[robot_type]["yaw"], degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # xyzw → wxyz
ROBOT_CFG.init_state.rot = q_wxyz

ROBOT_HEIGHT = scene_cfg[robot_type]["height"]
ROBOT_HEIGHT_IMG = scene_cfg[robot_type]["height_img"]

# Right tool: wrist -> tip
if robot_type == "g1":
    DRILL_TO_TIP_POS = np.array([0.305, 0.0, 0.0]).astype(np.float32)  # -0.135
elif robot_type == "h1":
    DRILL_TO_TIP_POS = np.array([0.328, 0.0, 0.0]).astype(np.float32)  # -0.135
q_xyzw = R.from_euler("YXZ", [-90, 0, 90], degrees=True).as_quat().astype(np.float32)
DRILL_TO_TIP_QUAT = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

WRIST_TO_TIP_POS = torch.tensor(DRILL_TO_TIP_POS, dtype=torch.float32)  # (3,)
WRIST_TO_TIP_QUAT_WXYZ = torch.tensor(DRILL_TO_TIP_QUAT, dtype=torch.float32)  # già wxyz

TIP_ALONG_Z_M = 0.15   # meters along trajectory direction

IK_ENABLE = True
PLOT_ENABLE = True

def _t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def build_pinocchio_model(urdf_path: str) -> tuple[pin.Model, pin.Data]:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

def build_name_to_qidx(model: pin.Model) -> dict[str, int]:
    name_to_qidx: dict[str, int] = {}
    for j_id, joint in enumerate(model.joints):
        if joint.nq == 1:
            name_to_qidx[model.names[j_id]] = joint.idx_q
    return name_to_qidx

def isaac_to_pin_q(
    q_isaac: torch.Tensor,
    isaac_joint_names: list[str],
    model: pin.Model,
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    q_pin = np.zeros(model.nq, dtype=float)
    q_isaac_np = q_isaac.detach().cpu().numpy()
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            q_pin[name_to_qidx[jname]] = float(q_isaac_np[i])
    return q_pin

def pin_to_isaac_q(
    q_pin: np.ndarray,
    isaac_joint_names: list[str],
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    q_isaac = np.zeros(len(isaac_joint_names), dtype=float)
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            q_isaac[i] = float(q_pin[name_to_qidx[jname]])
    return q_isaac

def quat_angle_error_deg(q_curr_wxyz: torch.Tensor, q_tgt_wxyz: torch.Tensor) -> torch.Tensor:
    """Return angular distance between two quaternions [w,x,y,z] in degrees.
    Uses 2*acos(|dot(q1,q2)|). Output shape: (N,)
    """
    # Normalize for safety
    q1 = q_curr_wxyz / torch.linalg.norm(q_curr_wxyz, dim=-1, keepdim=True).clamp_min(1e-9)
    q2 = q_tgt_wxyz / torch.linalg.norm(q_tgt_wxyz, dim=-1, keepdim=True).clamp_min(1e-9)

    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot)
    return ang * (180.0 / torch.pi)

def debug_print_waist_right_joints(
    robot: Articulation,
    right_entity_cfg: SceneEntityCfg,
    left_entity_cfg: SceneEntityCfg,
    every: int = 10,
):
    # right_entity_cfg.joint_ids are exactly the joints matched by right_joint_pattern
    joint_ids = right_entity_cfg.joint_ids + left_entity_cfg.joint_ids

    # joint names in the same order as joint_ids
    # (Articulation exposes joint names; try .joint_names / .data.joint_names depending on IsaacLab version)
    try:
        all_names = robot.joint_names
    except AttributeError:
        all_names = robot.data.joint_names  # fallback

    names = [all_names[j] for j in joint_ids]

    # Try to fetch per-joint position bounds (lower/upper) from IsaacLab, handling version differences.
    def _get_joint_pos_limits():
        # Most common in IsaacLab: (num_envs, num_joints, 2) or (num_joints, 2)
        for attr in ("joint_pos_limits", "joint_limits", "soft_joint_pos_limits"):
            if hasattr(robot.data, attr):
                lim = getattr(robot.data, attr)
                if lim is not None:
                    return lim
        return None

    def _print(step_i: int):
        if step_i % every != 0:
            return

        q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()

        lim = _get_joint_pos_limits()
        if lim is None:
            bounds = None
        else:
            # Support both shapes: (E, J, 2) and (J, 2)
            try:
                bounds = lim[0, joint_ids].detach().cpu().numpy()
            except Exception:
                bounds = lim[joint_ids].detach().cpu().numpy()

        print("\n[DEBUG] env0 waist+right joints:")
        if bounds is None:
            for n, qi in zip(names, q):
                print(f"  {n:40s}  q={qi:+.2f}   bounds=[?, ?]")
        else:
            for n, qi, (lo, hi) in zip(names, q, bounds):
                print(f"  {n:40s}  q={qi:+.2f}   bounds=[{lo:+.2f}, {hi:+.2f}]")

    return _print

def yoshikawa_manip_from_J(J: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Yoshikawa manipulability w = prod(singular_values(J)).
    Uses log-space to avoid underflow. Returns shape (N,).
    """
    # Work in float64 for numerical stability
    J64 = J.to(dtype=torch.float64)
    s = torch.linalg.svdvals(J64)  # (N,6) if J is (N,6,nJ)
    # Avoid log(0)
    s = torch.clamp(s, min=eps)
    logw = torch.sum(torch.log(s), dim=-1)  # (N,)
    w = torch.exp(logw)                     # (N,)
    return w.to(dtype=J.dtype)

def make_us_cfg_for_scene(raw_us_cfg: dict, downsample: int) -> dict:
    us_cfg = copy.deepcopy(raw_us_cfg)
    res = int(downsample)
    if res > 1:
        us_cfg["image_size"] = [int(us_cfg["image_size"][0]/res), int(us_cfg["image_size"][1]/res)]
        us_cfg["system_params"]["sx_E"] /= np.sqrt(res)
        us_cfg["system_params"]["sy_E"] /= np.sqrt(res)
        us_cfg["system_params"]["sx_B"] /= np.sqrt(res)
        us_cfg["system_params"]["sy_B"] /= np.sqrt(res)
        us_cfg["system_params"]["I0"] *= np.sqrt(res)
        us_cfg["E_S_ratio"] /= np.sqrt(res)
        us_cfg["resolution"] *= res
    return us_cfg

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


def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    robot_name = "g1" if robot_type == "g1" else "h1"
    robot: Articulation = scene[robot_name]
    human: RigidObject = scene["human"]


    def build_default_pos_from_cfg(robot: Articulation, robot_cfg: ArticulationCfg) -> torch.Tensor:
        """Return (N, n_joints) tensor using robot_cfg.init_state.joint_pos mapped by joint name."""
        N = scene.num_envs
        joint_names = list(robot.data.joint_names)
        nJ = len(joint_names)

        # start from current to preserve joints not specified in cfg
        q = robot.data.joint_pos.clone()  # (N, nJ)

        jp = robot_cfg.init_state.joint_pos  # dict: name -> value
        if isinstance(jp, dict):
            for j, name in enumerate(joint_names):
                if name in jp:
                    q[:, j] = float(jp[name])
        else:
            # if somebody configured it as list/array already
            q[:, :] = torch.as_tensor(jp, device=q.device, dtype=q.dtype).view(1, -1).repeat(N, 1)

        return q


    # Joint regex for each arm
    left_joint_pattern = rf"(torso_joint|waist_(pitch|roll|yaw)_joint|left_(shoulder|elbow|wrist)_.*)"
    right_joint_pattern = rf"(torso_joint|waist_(pitch|roll|yaw)_joint|right_(shoulder|elbow|wrist)_.*)"

    # Resolve EE and joint ids (for PD application)
    left_entity_cfg = SceneEntityCfg(robot_name, joint_names=[left_joint_pattern], body_names=[LEFT_EE_NAME])
    left_entity_cfg.resolve(scene)
    left_ee_id = left_entity_cfg.body_ids[-1]

    right_entity_cfg = SceneEntityCfg(robot_name, joint_names=[right_joint_pattern], body_names=[RIGHT_EE_NAME])
    right_entity_cfg.resolve(scene)
    right_ee_id = right_entity_cfg.body_ids[-1]

    # -------------------------------------------------------------------------
    # Manipulability logging (Yoshikawa) for drill (right) and US (left)
    # -------------------------------------------------------------------------

    # Keep the same naming used in your snippet (entity cfg aliases)
    robot_drill_entity_cfg = right_entity_cfg
    robot_us_entity_cfg = left_entity_cfg

    # Jacobian indices: user snippet expects "*_jacobi_idx - 1"
    # In IsaacLab/PhysX jacobians, the first index is typically the root.
    # A practical mapping is to offset body_id by +1 and then subtract 1 in the slice.
    drill_ee_jacobi_idx = int(right_ee_id) + 1
    US_ee_jacobi_idx = int(left_ee_id) + 1

    # Buffers (store CPU tensors, one per step)
    manipulability_drill_hist: list[torch.Tensor] = []
    manipulability_us_hist: list[torch.Tensor] = []

    # -------------------------------------------------------------------------
    # Arm-only (probe) joint subset for pre-positioning (exclude waist/torso)
    # -------------------------------------------------------------------------
    left_arm_only_pattern = rf"(torso_joint|left_(shoulder|elbow|wrist)_.*)"
    left_arm_only_entity_cfg = SceneEntityCfg(
        robot_name,
        joint_names=[left_arm_only_pattern],
        body_names=[LEFT_EE_NAME],
    )
    left_arm_only_entity_cfg.resolve(scene)

    left_arm_only_joint_ids_list = [int(j) for j in left_arm_only_entity_cfg.joint_ids]
    left_arm_only_joint_ids_t = torch.tensor(left_arm_only_joint_ids_list, device=sim.device, dtype=torch.long)

    full_joint_ids_list = sorted(set(left_entity_cfg.joint_ids + right_entity_cfg.joint_ids))
    full_joint_ids = torch.tensor(full_joint_ids_list, device=sim.device, dtype=torch.long)

    # Precompute fixed transforms for wrist <-> tip (batched)
    wrist_to_tip_pos = WRIST_TO_TIP_POS.to(sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
    wrist_to_tip_quat = WRIST_TO_TIP_QUAT_WXYZ.to(sim.device).unsqueeze(0).repeat(scene.num_envs, 1)

    # Inverse transform tip -> wrist
    tip_to_wrist_pos, tip_to_wrist_quat = subtract_frame_transforms(
        wrist_to_tip_pos,
        wrist_to_tip_quat,
        torch.zeros_like(wrist_to_tip_pos),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device).repeat(scene.num_envs, 1),
    )

    def _make_frame_vis(path: str, scale: float, color_rgb: tuple[float, float, float]) -> VisualizationMarkers:

        return VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=path,
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(scale, scale, scale),
                    ),
                },
            )
        )

    #vis_left_wrist  = _make_frame_vis("/Visuals/frames/left_wrist",  0.06, (0.10, 0.80, 1.00))  # azzurro
    #vis_right_wrist = _make_frame_vis("/Visuals/frames/right_wrist", 0.06, (1.00, 0.40, 0.10))  # arancio
    #vis_tip_target  = _make_frame_vis("/Visuals/frames/tip_target",  0.08, (0.20, 1.00, 0.20))  # verde
    #vis_right_tip   = _make_frame_vis("/Visuals/frames/right_tip",   0.08, (1.00, 1.00, 0.20))  # giallo

    # USSlicer
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r"))
    raw = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
    DOWNSAMPLE = 4
    us_cfg = make_us_cfg_for_scene(raw, downsample=DOWNSAMPLE)
    sim_cfg = scene_cfg["sim"]

    US_slicer = USSlicer(
        us_cfg,
        label_map_list,
        ct_map_list,
        sim_cfg["if_use_ct"],
        human_stl_list,
        scene.num_envs,
        sim_cfg["patient_xz_range"],
        sim_cfg["patient_xz_init"],
        sim.device,
        label_convert_map,
        us_cfg["image_size"],
        us_cfg["resolution"],
        height=ROBOT_HEIGHT,
        height_img=ROBOT_HEIGHT_IMG,
        visualize=sim_cfg["vis_seg_map"],
    )

    vertebra_viewer = VertebraViewer(
        scene.num_envs,
        len(patient_ids),
        target_stl_file_list,
        target_traj_file_list,
        if_vis=True,
        res=label_res,
        device=sim.device,
    )

    # LEFT command target on patient (human frame, from YAML)
    init_cmd_pose_min = torch.tensor(sim_cfg["patient_xz_init_range"][0], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)
    init_cmd_pose_max = torch.tensor(sim_cfg["patient_xz_init_range"][1], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)
    patient_xz_target = torch.tensor(sim_cfg["patient_xz_target"], device=sim.device).reshape((1, -1)).repeat(scene.num_envs, 1)
    US_slicer.current_x_z_x_angle_cmd = (init_cmd_pose_min + init_cmd_pose_max) / 2

    # Pink / Pinocchio
    model, data = build_pinocchio_model(URDF_PATH)
    name_to_qidx = build_name_to_qidx(model)
    joint_names = list(robot.data.joint_names)

    configuration: Configuration | None = None
    left_task: FrameTask | None = None
    right_task: FrameTask | None = None

    # -------------------------------------------------------------------------
    # Pre-positioning: run first PRE_STEPS with LEFT arm only to reach US target
    # -------------------------------------------------------------------------
    PRE_STEPS = 100
    pre_steps_left = 0  # set at reset

    pre_task: FrameTask | None = None

    # Soft reset caches
    # Soft reset caches (USE SURGERY init joints, not robot.data.default_joint_pos)
    default_pos = build_default_pos_from_cfg(robot, ROBOT_CFG)
    zero_vel = torch.zeros_like(robot.data.joint_vel)
    root_pos0 = None
    root_rot0 = None

    # Right TIP target state (base frame)
    tip_tgt_pos_b = None
    tip_tgt_rot_b = None
    tip_tgt_quat_b = None

    # Plot buffers
    if PLOT_ENABLE:
        t_hist = []
        l_pos_err_hist = []
        l_ang_err_hist = []
        r_pos_err_hist = []
        r_ang_err_hist = []
        PLOT_EVERY = 20

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle("EE Tracking Errors (env 0) - components")
        plt.ion()
        plt.show()

    # Constant EE->US rotation (same as your DLS scripts)
    R21 = np.array([[0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]], dtype=float)
    RotMat_torch = torch.as_tensor(R21, dtype=torch.float32, device=sim.device).unsqueeze(0).expand(scene.num_envs, -1, -1)

    sim_dt = sim.get_physics_dt()
    step_i = 0
    sim_time_acc = 0.0
    reset_T = float(args_cli.reset_seconds)

    # ----------------------------
    # Full joint set for a single set_joint_position_target call
    # Note: left includes waist joints; right is arm-only (as per your current patterns)
    full_joint_ids_list = sorted(set(left_entity_cfg.joint_ids + right_entity_cfg.joint_ids))
    full_joint_ids = torch.tensor(full_joint_ids_list, device=sim.device, dtype=torch.long)


    print_wr = debug_print_waist_right_joints(robot, right_entity_cfg, left_entity_cfg,  every=60)  # ogni 60 step


    while simulation_app.is_running():
        if step_i % sim_cfg["episode_length"] == 0:
            # reset robot joint state
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()

            robot.set_joint_position_target(default_pos[:, full_joint_ids], joint_ids=full_joint_ids)

            # cache root for soft reset
            root_state = robot.data.root_state_w.clone()
            root_pos0 = root_state[:, 0:3].clone()
            root_rot0 = root_state[:, 3:7].clone()

            # Human pose
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # same as robotic_US_guided_surgery.get_US_target_pose()
            vertebra_to_US_2d_pos = torch.tensor(
                motion_plan_cfg["vertebra_to_US_2d_pos"], device=sim.device, dtype=torch.float32
            ).reshape(1, 2)

            vertebra_2d_pos = vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]  # (N,2)
            US_target_2d_pos = vertebra_2d_pos + vertebra_to_US_2d_pos

            US_slicer.roll_adj = torch.full(
                (scene.num_envs, 1),
                float(motion_plan_cfg["US_roll_adj"]),
                device=sim.device,
                dtype=torch.float32,
            )

            US_target_2d_angle = torch.full(
                (scene.num_envs, 1),
                1.57,
                device=sim.device,
                dtype=torch.float32,
            )

            US_slicer.current_x_z_x_angle_cmd = torch.cat(
                [US_target_2d_pos, US_target_2d_angle], dim=-1
            )
            US_cmd_hold = US_slicer.current_x_z_x_angle_cmd.clone()

            # update internal target pose buffers
            US_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_rot)

            # Base pose in WORLD
            base_w = robot.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            
            # Left EE pose in WORLD
            left_ee_w = robot.data.body_state_w[:, left_ee_id, 0:7]
            right_ee_w = robot.data.body_state_w[:, right_ee_id, 0:7]
            
            # Left EE pose in BASE
            left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                left_ee_w[:, 0:3],
                left_ee_w[:, 3:7],
            )
            # Right EE pose in BASE
            right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                right_ee_w[:, 0:3],
                right_ee_w[:, 3:7],
            )
            
            # --- decide drill-axis sign once (keep same hemisphere as current tip) ---
            right_tip_pos_w0, right_tip_quat_w0 = combine_frame_transforms(
                right_ee_w[:, 0:3], right_ee_w[:, 3:7],
                wrist_to_tip_pos, wrist_to_tip_quat
            )
            human_tip_pos0, human_tip_quat0 = subtract_frame_transforms(
                world_to_human_pos, world_to_human_rot,
                right_tip_pos_w0, right_tip_quat_w0
            )
            tip_z0_h = matrix_from_quat(human_tip_quat0)[:, :, 2]  # current tip z in HUMAN
            dot0 = torch.sum(tip_z0_h * vertebra_viewer.traj_drct, dim=-1, keepdim=True)
            traj_sign = torch.where(dot0 >= 0.0, torch.ones_like(dot0), -torch.ones_like(dot0))  # (N,1)

            # init Pink config (env0)
            q0_isaac = robot.data.joint_pos[0]
            q0_pin = isaac_to_pin_q(q0_isaac, joint_names, model, name_to_qidx)
            configuration = Configuration(model, data, q0_pin)

            left_task = FrameTask(frame=LEFT_EE_NAME, position_cost=1.0, orientation_cost=1.0)
            right_task = FrameTask(frame=RIGHT_EE_NAME, position_cost=1.0, orientation_cost=0.2)

            # Pre-positioning task (left EE only)
            pre_task = FrameTask(frame=LEFT_EE_NAME, position_cost=10.0, orientation_cost=10.0)
            pre_steps_left = PRE_STEPS

        # ---------------------------------------------------------------------
        # Pre-positioning phase: first PRE_STEPS steps -> move probe to target
        # ---------------------------------------------------------------------
        if IK_ENABLE and pre_steps_left > 0:
            # human pose in WORLD
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # base in WORLD
            base_w = robot.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]

            # Keep command fixed during pre-positioning (same as your env reset)
            US_slicer.current_x_z_x_angle_cmd = US_cmd_hold
            US_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_rot)

            # Target EE in HUMAN (orientation in US frame)
            human_ee_target_pos = US_slicer.human_to_ee_target_pos.to(sim.device)
            human_ee_target_quat = US_slicer.human_to_ee_target_quat.to(sim.device)  # wxyz

            # HUMAN -> WORLD (orientation still in US frame)
            world_left_pos, world_left_quat_us = combine_frame_transforms(
                world_to_human_pos, world_to_human_rot,
                human_ee_target_pos, human_ee_target_quat
            )

            # US frame -> robot EE frame (undo EE->US)
            world_left_rot_us = matrix_from_quat(world_left_quat_us)
            world_left_rot_robot = torch.bmm(world_left_rot_us, RotMat_torch.transpose(1, 2))
            world_left_quat = quat_from_matrix(world_left_rot_robot)

            # WORLD -> BASE (Pink "world" = base)
            base_left_pos, base_left_quat = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                world_left_pos, world_left_quat
            )

            # Pink target (env 0): BASE frame
            left_R_b = _t2np(matrix_from_quat(base_left_quat)[0]).astype(float)
            left_t_b = _t2np(base_left_pos[0]).astype(float)
            pre_task.set_target(pin.SE3(left_R_b, left_t_b))

            # Update Pink configuration from current Isaac joints (env0)
            q_curr_isaac = robot.data.joint_pos[0]
            q_curr_pin = isaac_to_pin_q(q_curr_isaac, joint_names, model, name_to_qidx)
            configuration.update(q_curr_pin)

            # Solve IK with LEFT task only
            vel = solve_ik(
                configuration,
                tasks=[pre_task],
                dt=sim_dt,
                solver="quadprog",
                damping=5e-2,
                safety_break=False,
            )
            configuration.integrate_inplace(vel, sim_dt)
            q_next_pin = configuration.q.copy()

            # Map back to Isaac ordering, then apply ONLY left-arm joints
            q_next_isaac = pin_to_isaac_q(q_next_pin, joint_names, name_to_qidx)

            # Build target vector for left-arm-only subset
            arm_q = q_next_isaac[left_arm_only_joint_ids_list]
            arm_q_t = torch.from_numpy(arm_q).to(sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)

            # Optional clamp to joint limits (recommended)
            jl = robot.data.joint_pos_limits
            jmin = jl[0, left_arm_only_joint_ids_t, 0]
            jmax = jl[0, left_arm_only_joint_ids_t, 1]
            safety_margin = 1e-5
            arm_q_t = torch.clamp(arm_q_t, jmin.unsqueeze(0) + safety_margin, jmax.unsqueeze(0) - safety_margin)

            robot.set_joint_position_target(arm_q_t, joint_ids=left_arm_only_joint_ids_t)

            pre_steps_left -= 1

            # Skip the bimanual IK block for this step
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            step_i += 1
            sim_time_acc += sim_dt
            continue

            # ---------------------------------------------------------
            # Visualize frames (WORLD)
            # ---------------------------------------------------------
        
        if step_i % 10 == 0:
            idx = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
            # 1) LEFT wrist (world)
            #vis_left_wrist.visualize(left_ee_w[:, 0:3], left_ee_w[:, 3:7], marker_indices=idx)
            # 2) RIGHT wrist (world)
            #vis_right_wrist.visualize(right_ee_w[:, 0:3], right_ee_w[:, 3:7], marker_indices=idx)
            # 3) TIP target (world) = base->world (tip target in base)
            tip_tgt_pos_w_vis, tip_tgt_quat_w_vis = combine_frame_transforms(
                base_pos_w, base_quat_w, tip_tgt_pos_b, tip_tgt_quat_b
            )
            #vis_tip_target.visualize(tip_tgt_pos_w_vis, tip_tgt_quat_w_vis, marker_indices=idx)

            # 4) Current RIGHT tip (world) = right_wrist->tip (world)
            right_tip_pos_w, right_tip_quat_w = combine_frame_transforms(
                right_ee_w[:, 0:3], right_ee_w[:, 3:7],
                wrist_to_tip_pos, wrist_to_tip_quat
            )
            #vis_right_tip.visualize(right_tip_pos_w, right_tip_quat_w, marker_indices=idx)

        

        if IK_ENABLE:

            # human pose in WORLD
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # base in WORLD
            base_w = robot.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]

            # current wrists in WORLD
            left_ee_w = robot.data.body_state_w[:, left_ee_id, 0:7]
            right_ee_w = robot.data.body_state_w[:, right_ee_id, 0:7]
            left_wrist_pos_w, left_wrist_quat_w = left_ee_w[:, 0:3], left_ee_w[:, 3:7]

            # Build US orientation from LEFT wrist
            left_wrist_rotmat_w = matrix_from_quat(left_wrist_quat_w)
            left_us_rotmat_w = torch.bmm(left_wrist_rotmat_w, RotMat_torch)
            left_us_quat_w = quat_from_matrix(left_us_rotmat_w)
            left_us_pos_w = left_wrist_pos_w

            US_slicer.current_x_z_x_angle_cmd = US_cmd_hold
            US_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_rot)

            # Slice US
            US_slicer.slice_US(world_to_human_pos, world_to_human_rot, left_us_pos_w, left_us_quat_w)
            if sim_cfg.get("vis_us", False):
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg.get("vis_seg_map", False):
                US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, left_us_pos_w, left_us_quat_w)

            # ---------------------------------------------------------
            # LEFT target EE pose: take it from USSlicer (HUMAN frame)
            # ---------------------------------------------------------
            human_ee_target_pos = US_slicer.human_to_ee_target_pos.to(sim.device)        # (N,3)
            human_ee_target_quat = US_slicer.human_to_ee_target_quat.to(sim.device)     # (N,4) wxyz

            # HUMAN -> WORLD (orientation is still US-frame)
            world_left_pos, world_left_quat_us = combine_frame_transforms(
                world_to_human_pos, world_to_human_rot,
                human_ee_target_pos, human_ee_target_quat
            )

            # US -> robot orientation (undo EE->US)
            world_left_rot_us = matrix_from_quat(world_left_quat_us)
            world_left_rot_robot = torch.bmm(world_left_rot_us, RotMat_torch.transpose(1, 2))
            world_left_quat = quat_from_matrix(world_left_rot_robot)

            base_left_pos, base_left_quat = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                world_left_pos, world_left_quat
            )

            tip_tgt_pos_h = vertebra_viewer.human_to_traj_pos.clone() 

            # desired tip orientation in HUMAN: align tip z-axis with trajectory direction (with stable sign)
            z_h = - vertebra_viewer.traj_drct * traj_sign  # (N,3)

            up = torch.tensor([0.0, 1.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
            up_alt = torch.tensor([1.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
            is_parallel = (torch.abs(torch.sum(up * z_h, dim=-1, keepdim=True)) > 0.95)
            up = torch.where(is_parallel, up_alt, up)

            x_h = torch.cross(up, z_h, dim=-1)
            x_h = x_h / torch.linalg.norm(x_h, dim=-1, keepdim=True).clamp_min(1e-9)
            y_h = torch.cross(z_h, x_h, dim=-1)

            R_tip_h = torch.stack([x_h, y_h, z_h], dim=-1)   # columns = axes in HUMAN
            tip_tgt_quat_h = quat_from_matrix(R_tip_h)       # (N,4) wxyz

            # 180° around Y in HUMAN frame
            q_flip_y = torch.tensor([0.0, 0.0, 1.0, 0.0], device=sim.device, dtype=torch.float32)  # wxyz
            q_flip_y = q_flip_y.unsqueeze(0).repeat(scene.num_envs, 1)

            # apply as local rotation: R_new = R_old * Ry(pi)
            tip_tgt_quat_h = quat_mul(tip_tgt_quat_h, q_flip_y)

            # ---------------------------------------------------------
            # Offset the tip target position along its local +Z axis
            # (computed from the target orientation in HUMAN frame)
            # ---------------------------------------------------------
            R_tip_h = matrix_from_quat(tip_tgt_quat_h)          # (N,3,3)
            z_axis_h = R_tip_h[:, :, 2]                         # (N,3) local +Z expressed in HUMAN frame

            tip_tgt_pos_h = tip_tgt_pos_h + float(TIP_ALONG_Z_M) * z_axis_h

            # HUMAN -> WORLD
            tip_tgt_pos_w, tip_tgt_quat_w = combine_frame_transforms(
                world_to_human_pos, world_to_human_rot,
                tip_tgt_pos_h, tip_tgt_quat_h
            )

            # WORLD -> BASE  (tip target in BASE)
            tip_tgt_pos_b, tip_tgt_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                tip_tgt_pos_w, tip_tgt_quat_w
            )

            # TIP -> WRIST in BASE using fixed offset (tip_to_wrist_*)
            wrist_tgt_pos_b, wrist_tgt_quat_b = combine_frame_transforms(
                tip_tgt_pos_b, tip_tgt_quat_b,
                tip_to_wrist_pos, tip_to_wrist_quat
            )

            # For plotting/metrics in WORLD, reconstruct wrist target in WORLD
            world_right_pos, world_right_quat = combine_frame_transforms(
                base_pos_w, base_quat_w,
                wrist_tgt_pos_b, wrist_tgt_quat_b
            )

            # Pink target is in BASE (Pink-world = base)
            base_right_pos, base_right_quat = wrist_tgt_pos_b, wrist_tgt_quat_b

            # --- Pink targets: usa BASE frame (Pink-world = base) ---
            left_R_b  = _t2np(matrix_from_quat(base_left_quat)[0]).astype(float)
            left_t_b  = _t2np(base_left_pos[0]).astype(float)

            right_R_b = _t2np(matrix_from_quat(base_right_quat)[0]).astype(float)
            right_t_b = _t2np(base_right_pos[0]).astype(float)

            left_task.set_target(pin.SE3(left_R_b, left_t_b))
            right_task.set_target(pin.SE3(right_R_b, right_t_b))

            # update Pink configuration from current Isaac joints (env0)
            q_curr_isaac = robot.data.joint_pos[0]
            q_curr_pin = isaac_to_pin_q(q_curr_isaac, joint_names, model, name_to_qidx)
            configuration.update(q_curr_pin)

            # solve QP
            vel = solve_ik(
                configuration,
                tasks=[left_task, right_task],
                dt=sim_dt,
                solver="quadprog",
                damping=5e-2,
                safety_break=False,
            )
            configuration.integrate_inplace(vel, sim_dt)
            q_next_pin = configuration.q.copy()

            # map back to Isaac joint ordering
            q_next_isaac = pin_to_isaac_q(q_next_pin, joint_names, name_to_qidx)  # (n_joints,)
            q_next_ctrl = q_next_isaac[full_joint_ids_list]                       # (n_ctrl,)
            q_next_ctrl_t = torch.from_numpy(q_next_ctrl).to(sim.device, dtype=torch.float32).unsqueeze(0)
            q_next_ctrl_t = q_next_ctrl_t.repeat(scene.num_envs, 1)

            # apply PD targets once
            robot.set_joint_position_target(q_next_ctrl_t, joint_ids=full_joint_ids)


        # ---------------------------------------------------------
        # Visualize frames (WORLD)
        # ---------------------------------------------------------
        if step_i % 10 == 0:
            idx = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
            # 1) LEFT wrist (world)
            #vis_left_wrist.visualize(left_ee_w[:, 0:3], left_ee_w[:, 3:7], marker_indices=idx)

            # 2) RIGHT wrist (world)
            #vis_right_wrist.visualize(right_ee_w[:, 0:3], right_ee_w[:, 3:7], marker_indices=idx)

            # 3) TIP target (world) = base->world (tip target in base)
            tip_tgt_pos_w_vis, tip_tgt_quat_w_vis = combine_frame_transforms(
                base_pos_w, base_quat_w, tip_tgt_pos_b, tip_tgt_quat_b
            )
            #vis_tip_target.visualize(tip_tgt_pos_w_vis, tip_tgt_quat_w_vis, marker_indices=idx)

            # 4) Current RIGHT tip (world) = right_wrist->tip (world)
            right_tip_pos_w, right_tip_quat_w = combine_frame_transforms(
                right_ee_w[:, 0:3], right_ee_w[:, 3:7],
                wrist_to_tip_pos, wrist_to_tip_quat
            )
            #vis_right_tip.visualize(right_tip_pos_w, right_tip_quat_w, marker_indices=idx)
        
            # ---------------------------------------------------------
            # Plot tracking errors (components) - env 0
            # ---------------------------------------------------------
            if PLOT_ENABLE:
                # current EE in WORLD
                left_pos_w_cur = left_ee_w[:, 0:3]
                left_quat_w_cur = left_ee_w[:, 3:7]
                right_pos_w_cur = right_ee_w[:, 0:3]
                right_quat_w_cur = right_ee_w[:, 3:7]

                # pos error vectors in WORLD
                left_pos_err_vec = (left_pos_w_cur - world_left_pos)      # (N,3)
                right_pos_err_vec = (right_pos_w_cur - world_right_pos)   # (N,3)

                # orientation error: q_err = q_tgt * inv(q_cur)
                ql_err = quat_mul(world_left_quat, quat_inv(left_quat_w_cur))
                qr_err = quat_mul(world_right_quat, quat_inv(right_quat_w_cur))

                l_roll, l_pitch, l_yaw = euler_xyz_from_quat(ql_err)
                r_roll, r_pitch, r_yaw = euler_xyz_from_quat(qr_err)

                left_ang_err_vec_deg = torch.stack(
                    [torch.rad2deg(l_roll), torch.rad2deg(l_pitch), torch.rad2deg(l_yaw)], dim=-1
                )
                right_ang_err_vec_deg = torch.stack(
                    [torch.rad2deg(r_roll), torch.rad2deg(r_pitch), torch.rad2deg(r_yaw)], dim=-1
                )

                t_hist.append(sim_time_acc)
                l_pos_err_hist.append(left_pos_err_vec[0].detach().cpu().numpy())
                r_pos_err_hist.append(right_pos_err_vec[0].detach().cpu().numpy())
                l_ang_err_hist.append(left_ang_err_vec_deg[0].detach().cpu().numpy())
                r_ang_err_hist.append(right_ang_err_vec_deg[0].detach().cpu().numpy())

                if (step_i % PLOT_EVERY) == 0 and len(t_hist) > 2:
                    ax00, ax01 = axes[0, 0], axes[0, 1]
                    ax10, ax11 = axes[1, 0], axes[1, 1]
                    ax00.cla(); ax01.cla(); ax10.cla(); ax11.cla()

                    lpos = np.stack(l_pos_err_hist, axis=0)
                    ax00.plot(t_hist, lpos[:, 0], label="ex")
                    ax00.plot(t_hist, lpos[:, 1], label="ey")
                    ax00.plot(t_hist, lpos[:, 2], label="ez")
                    ax00.set_title("Left pos err [m] (WORLD)")
                    ax00.set_xlabel("time [s]"); ax00.set_ylabel("m")
                    ax00.legend()
                    ax00.grid(True, which="both"); ax00.minorticks_on()

                    lang = np.stack(l_ang_err_hist, axis=0)
                    ax01.plot(t_hist, lang[:, 0], label="roll")
                    ax01.plot(t_hist, lang[:, 1], label="pitch")
                    ax01.plot(t_hist, lang[:, 2], label="yaw")
                    ax01.set_title("Left ang err [deg] (WORLD)")
                    ax01.set_xlabel("time [s]"); ax01.set_ylabel("deg")
                    ax01.legend()
                    ax01.grid(True, which="both"); ax01.minorticks_on()

                    rpos = np.stack(r_pos_err_hist, axis=0)
                    ax10.plot(t_hist, rpos[:, 0], label="ex")
                    ax10.plot(t_hist, rpos[:, 1], label="ey")
                    ax10.plot(t_hist, rpos[:, 2], label="ez")
                    ax10.set_title("Right pos err [m] (WORLD)")
                    ax10.set_xlabel("time [s]"); ax10.set_ylabel("m")
                    ax10.legend()
                    ax10.grid(True, which="both"); ax10.minorticks_on()

                    rang = np.stack(r_ang_err_hist, axis=0)
                    ax11.plot(t_hist, rang[:, 0], label="roll")
                    ax11.plot(t_hist, rang[:, 1], label="pitch")
                    ax11.plot(t_hist, rang[:, 2], label="yaw")
                    ax11.set_title("Right ang err [deg] (WORLD)")
                    ax11.set_xlabel("time [s]"); ax11.set_ylabel("deg")
                    ax11.legend()
                    ax11.grid(True, which="both"); ax11.minorticks_on()

                    fig.tight_layout()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
        
        print_wr(step_i)
        # ---------------------------------------------------------
        # End-of-episode print: final manipulability values
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # Manipulability (AFTER physics update): BASE frame
        # ---------------------------------------------------------
        try:
            jacobians = robot.root_physx_view.get_jacobians()  # updated after scene.update
        except Exception as e:
            jacobians = None
            if step_i % 100 == 0:
                print(f"[MANIP] get_jacobians() failed: {e}")

        if jacobians is not None:
            # WORLD -> BASE rotation
            world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
            base_rotmat = matrix_from_quat(quat_inv(world_to_base_pose[:, 3:7]))  # (N,3,3)

            # IMPORTANT: index selection
            # In most IsaacLab builds: jacobians[:, body_id, :, dof_id]
            # So use right_ee_id / left_ee_id directly (no +1/-1).
            drill_jac = jacobians[:, int(right_ee_id), :, robot_drill_entity_cfg.joint_ids]  # (N,6,nJ)
            us_jac    = jacobians[:, int(left_ee_id),  :, robot_us_entity_cfg.joint_ids]     # (N,6,nJ)

            # Rotate WORLD -> BASE for both linear and angular parts
            drill_jac[:, 0:3, :] = torch.bmm(base_rotmat, drill_jac[:, 0:3, :])
            drill_jac[:, 3:6, :] = torch.bmm(base_rotmat, drill_jac[:, 3:6, :])

            us_jac[:, 0:3, :] = torch.bmm(base_rotmat, us_jac[:, 0:3, :])
            us_jac[:, 3:6, :] = torch.bmm(base_rotmat, us_jac[:, 3:6, :])

            manipulability_drill = yoshikawa_manip_from_J(drill_jac, eps=1e-12)
            manipulability_US    = yoshikawa_manip_from_J(us_jac,    eps=1e-12)

            manipulability_drill_hist.append(manipulability_drill.detach().cpu())
            manipulability_us_hist.append(manipulability_US.detach().cpu())

            # Debug rapido: se ancora 0, controlla la norma del Jacobiano
            if step_i % 200 == 0:
                nd = float(torch.linalg.norm(drill_jac[0]).item())
                nu = float(torch.linalg.norm(us_jac[0]).item())
                print(f"[MANIP DBG] step={step_i} | ||J_drill||={nd:.3e}, ||J_US||={nu:.3e}")

        if ((step_i + 1) % int(sim_cfg["episode_length"])) == 0:
            if len(manipulability_drill_hist) > 0 and len(manipulability_us_hist) > 0:
                mD = manipulability_drill_hist[-1]
                mU = manipulability_us_hist[-1]
                print(
                    f"[MANIP END] step={step_i} | "
                    f"drill(env0)={float(mD[0]):.6e}, US(env0)={float(mU[0]):.6e} | "
                    f"drill(mean)={float(mD.mean()):.6e}, US(mean)={float(mU.mean()):.6e}"
                )
            else:
                print(f"[MANIP END] step={step_i} | no samples.")

        # ----------------------------
        # SCENE UPDATE (mandatory)
        # ----------------------------
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_i += 1
        sim_time_acc += sim_dt


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(
        RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    )

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

    sim.reset()
    print("[INFO] Setup complete. Running…")
    run(sim, scene, label_map_list, ct_map_list)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")
    simulation_app.close()