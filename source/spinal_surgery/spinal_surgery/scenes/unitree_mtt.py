# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Spawn SonoGym scene with Unitree G1 or H1 (Pink/Differential IK test of move-towards-target).
Launch Isaac Sim first.
"""

import argparse
from isaaclab.app import AppLauncher

# --- CLI
parser = argparse.ArgumentParser(description="Pink IK test with fixed target in human frame (G1 / H1).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument(
    "--ik_steps",
    type=int,
    default=100,
    help="Number of IK steps towards the target before holding the pose.",
)
parser.add_argument(
    "--ik",
    type=str,
    default="pink",
    choices=["pink", "dik"],
    help="IK controller to use: 'pink' (Pink/Pinocchio) or 'dik' (Differential IK).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab imports
import time
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_inv,
    matrix_from_quat,
    quat_from_matrix,
    combine_frame_transforms,
)

# --- SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import G1_TOOLS_BASE_FIX_CFG
from spinal_surgery.assets.unitreeH1 import H12_CFG_TOOLS_BASEFIX
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer

import pinocchio as pin

# --- Pink imports --------------------------------------------------------
import pink
from pink.configuration import Configuration
from pink.tasks import FrameTask, PostureTask
from pink.solve_ik import solve_ik
# -------------------------------------------------------------------------

# --- Other libs
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import cProfile
import nibabel as nib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

########### Scene parameters from YAML (bed + patient + robot) ##############

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))

# patient pose
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg["pos"],
    rot=(quat[3], quat[0], quat[1], quat[2]),
)

# bed pose
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg["pos"],
    rot=(quat[3], quat[0], quat[1], quat[2]),
)
scale_bed = bed_cfg["scale"]

# datasets (human USD/labels/CT)
patient_ids = patient_cfg["id_list"]

human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id
    for p_id in patient_ids
]
human_stl_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id
    for p_id in patient_ids
]
human_raw_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id for p_id in patient_ids
]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg["label_res"]
scale = 1 / label_res

# robot + sim cfg
robot_cfg = scene_cfg["robot"]
sim_cfg = scene_cfg["sim"]

ROBOT_TYPE = robot_cfg.get("type", "g1").lower()
if ROBOT_TYPE not in ("g1", "h1"):
    raise ValueError(f"Unsupported robot.type in YAML: {ROBOT_TYPE}")

# Select pose sub-config and base articulation cfg depending on robot type
if ROBOT_TYPE == "g1":
    robot_pose_cfg = scene_cfg["g1"]
    ROBOT_SCENE_CFG: ArticulationCfg = G1_TOOLS_BASE_FIX_CFG.copy()
    ROBOT_SCENE_CFG.prim_path = "/World/envs/env_.*/G1"
    EE_LINK_NAME = "left_wrist_yaw_link"  # parent link used to define the virtual EE (G1)
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif ROBOT_TYPE == "h1":
    robot_pose_cfg = scene_cfg["h1"]
    ROBOT_SCENE_CFG: ArticulationCfg = H12_CFG_TOOLS_BASEFIX.copy()
    ROBOT_SCENE_CFG.prim_path = "/World/envs/env_.*/H1"
    # NOTE: adjust this link name if H1 has a different EE link name in your USD
    EE_LINK_NAME = "left_wrist_yaw_link"
    # NOTE: adjust URDF path if different in your project
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1.urdf"

# override init pose + orientation from YAML
ROBOT_SCENE_CFG.init_state.pos = robot_pose_cfg["pos"]

q_xyzw = R.from_euler("z", robot_pose_cfg["yaw"], degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # xyzw → wxyz
ROBOT_SCENE_CFG.init_state.rot = q_wxyz

# IK controller settings
IK_ENABLE = True

# initial command pose range
cmd_pose_min = sim_cfg["patient_xz_init_range"][0]
cmd_pose_max = sim_cfg["patient_xz_init_range"][1]


# Scene config
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal SonoGym scene + Unitree G1/H1 (idle)."""

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

    # medical bed
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

    # human
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

    # robot (Unitree G1 or H1)
    robot = ROBOT_SCENE_CFG


# Helpers: torch<->numpy bridges
def _t2np(t: torch.Tensor):
    """Detach Torch tensor to CPU numpy array."""
    return t.detach().cpu().numpy()


def isaac_to_scipy_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert IsaacLab [w, x, y, z] quaternion to SciPy [x, y, z, w]."""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


# === Pink / Pinocchio helpers ==============================================

def build_pinocchio_model(urdf_path: str):
    """Build Pinocchio model and data from URDF path."""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def build_name_to_qidx(model: pin.Model):
    """Build a mapping from joint name to configuration index in Pinocchio q.

    Only 1-DoF joints are considered; floating base is ignored.
    """
    name_to_qidx = {}
    for j_id, joint in enumerate(model.joints):
        name = model.names[j_id]
        if joint.nq == 1:
            name_to_qidx[name] = joint.idx_q
    return name_to_qidx


def isaac_to_pin_q(
    q_isaac: torch.Tensor,
    isaac_joint_names: list[str],
    model: pin.Model,
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    """Map IsaacLab joint positions to Pinocchio configuration vector q.

    Assumes fixed-base URDF: q has only joint DOFs, no free-flyer.
    """
    q_pin = np.zeros(model.nq, dtype=float)
    q_isaac_np = q_isaac.detach().cpu().numpy()
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            idx_q = name_to_qidx[jname]
            q_pin[idx_q] = q_isaac_np[i]
    return q_pin


def pin_to_isaac_q(
    q_pin: np.ndarray,
    isaac_joint_names: list[str],
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    """Map Pinocchio configuration q back to IsaacLab joint vector ordering."""
    q_isaac = np.zeros(len(isaac_joint_names), dtype=float)
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            idx_q = name_to_qidx[jname]
            q_isaac[i] = q_pin[idx_q]
    return q_isaac


# Run loop
def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list | None = None):

    robot = scene["robot"]
    human = scene["human"]

    ik_mode = args_cli.ik.lower()

    # Pink / Pinocchio IK setup (once)
    model, data = build_pinocchio_model(URDF_PATH)
    name_to_qidx = build_name_to_qidx(model)

    # One Configuration per environment (used only if ik_mode == "pink")
    configurations: list[Configuration] = []

    # Differential IK controller (used only if ik_mode == "dik")
    pose_diff_ik_controller = None
    if ik_mode == "dik":
        ik_params = {"lambda_val": 1e-1}
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

    # get US probe index
    SIDE = "left" if "left" in EE_LINK_NAME else ("right" if "right" in EE_LINK_NAME else None)
    joint_pattern = rf"(waist_(pitch|roll|yaw)_joint|{SIDE}_(shoulder|elbow|wrist)_.*)" if SIDE else r".*"
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[joint_pattern], body_names=[EE_LINK_NAME])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    # resolve both wrist links (optional debug – link names assumed same across robots)
    robot_ee_both_cfg = SceneEntityCfg(
        "robot",
        joint_names=[],
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
    )
    try:
        robot_ee_both_cfg.resolve(scene)
        left_wrist_id, right_wrist_id = robot_ee_both_cfg.body_ids
    except Exception:
        left_wrist_id = right_wrist_id = None

    # construct label image slicer
    label_convert_map = YAML().load(
        open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
    )

    # construct US simulator, using robot-specific height from YAML
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
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
        visualize=sim_cfg["vis_seg_map"],
        height=robot_pose_cfg["height"],
        height_img=robot_pose_cfg["height_img"],
    )

    # episode length in steps (taken from YAML)
    episode_len = sim_cfg["episode_length"]

    # joint names and default pose (valid for whole simulation)
    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = robot.data.default_joint_vel.clone() * 0.0

    # time and step counters
    sim_dt = sim.get_physics_dt()
    step_i = 0

    # IK steps (after these, we keep last pose)
    ik_steps = int(args_cli.ik_steps)
    ik_step_counter = 0

    # number of IK steps for the first phase (lifted target)
    mid_phase_steps = max(1, ik_steps // 2)

    # --- Error logging (for plotting vs time) ---
    pos_err_hist = []  # position error (EE frame, env 0)
    ang_err_hist = []  # orientation error (EE frame, env 0, deg)
    time_hist = []     # simulation time [s]

    # Global simulation time (never reset across episodes)
    global_time = 0.0

    # Times when IK stops applying commands (for plotting vertical lines)
    ik_stop_times = []

    # Track previous IK active state to detect transitions
    ik_active_prev = False

    # create figure for live plot of errors
    plt.ion()
    fig, (ax_pos, ax_ang) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(f"IK tracking error in EE frame ({ik_mode}, {ROBOT_TYPE.upper()})")

    # Figure for target vs reached points in patient (human) frame, X–Z plane
    # Left: scatter in cmd units, right: color-env table
    fig_plane, (ax_plane, ax_plane_table) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    ax_plane.set_title("Targets (x) and reached points (dot) in patient plane (cmd units)")
    ax_plane.set_xlabel("cmd x")
    ax_plane.set_ylabel("cmd z")
    ax_plane.grid(True)
    ax_plane.set_aspect("equal", adjustable="box")

    ax_plane_table.axis("off")

    # Buffers storing all points across episodes (for visualization)
    patient_targets_all = []   # list of (num_envs, 2) arrays [x,z]
    patient_reached_all = []   # list of (num_envs, 2) arrays [x,z]
    patient_env_ids_all = []   # list of (num_envs,) arrays with env indices

    # Pink task placeholders (shared across envs, only for Pink)
    ee_task: FrameTask | None = None
    ik_tasks = []

    # --- USER-DEFINED TARGET IN HUMAN/US FRAME ---------------------------
    init_cmd_pose_min = (
        torch.tensor(cmd_pose_min, device=sim.device)
        .reshape((1, -1))
        .repeat(scene.num_envs, 1)
    )
    init_cmd_pose_max = (
        torch.tensor(cmd_pose_max, device=sim.device)
        .reshape((1, -1))
        .repeat(scene.num_envs, 1)
    )

    US_slicer.current_x_z_x_angle_cmd = (
        init_cmd_pose_min + init_cmd_pose_max
    ) / 2

    # Store the last sampled command poses (for plotting in cmd units)
    last_cmd_target_poses: torch.Tensor | None = None

    while simulation_app.is_running():
        # ---------------------------------------------------------------
        # Decide if IK is active at this step (before any reset logic)
        # ---------------------------------------------------------------
        ik_active = IK_ENABLE and (step_i % episode_len) != 0 and ik_step_counter < ik_steps

        # ---------------------------------------------------------------
        # Episodic reset: whenever step_i % episode_len == 0
        # ---------------------------------------------------------------
        if step_i % episode_len == 0:
            # reset articulation + scene
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()   # let the scene reposition the human

            # ---------------- Pink: init one Pinocchio Configuration per env
            if ik_mode == "pink":
                configurations = []
                for env_id in range(scene.num_envs):
                    q0_isaac_i = robot.data.joint_pos[env_id]  # (n_joints,)
                    q0_pin_i = isaac_to_pin_q(q0_isaac_i, joint_names, model, name_to_qidx)
                    configurations.append(Configuration(model, data, q0_pin_i))

                ee_task = FrameTask(
                    frame=EE_LINK_NAME,
                    position_cost=1.0,
                    orientation_cost=1.0,
                )
                ik_tasks = [ee_task]

            # ---------------- DIK: reset controller and neutral command
            if ik_mode == "dik":
                base_w = robot.data.root_link_state_w[:, 0:7]
                base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]

                ee_parent_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]
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
                pose_diff_ik_controller.set_command(ik_commands_pose, ee_pos_b, ee_quat_b)

            # -------------- Human pose (WORLD) --------------
            human_world_poses = human.data.body_link_state_w[:, 0, 0:7]
            world_to_human_pos, world_to_human_rot = (
                human_world_poses[:, 0:3],
                human_world_poses[:, 3:7],
            )

            # -------------- Random target in patient frame (common to both) --------------
            cmd_target_poses = torch.rand(
                (scene.num_envs, 3), device=sim.device
            )
            min_init = init_cmd_pose_min
            max_init = init_cmd_pose_max
            cmd_target_poses = cmd_target_poses * (max_init - min_init) + min_init

            US_slicer.update_cmd(
                cmd_target_poses - US_slicer.current_x_z_x_angle_cmd
            )
            
            # Save cmd pose for plotting (cmd units on axes)
            last_cmd_target_poses = cmd_target_poses.clone().detach()

            world_to_ee_init_pos, world_to_ee_init_rot = (
                US_slicer.compute_world_ee_pose_from_cmd(
                    world_to_human_pos, world_to_human_rot
                )
            )

            # EE target (contact) in human frame
            human_ee_target_pos = US_slicer.human_to_ee_target_pos
            human_ee_target_quat = US_slicer.human_to_ee_target_quat

            print(f"[INFO] New episode started at step {step_i} (ik_mode={ik_mode}, robot={ROBOT_TYPE})")

            # reset IK step counter for this episode
            ik_step_counter = 0

            # At reset step we do not want IK active
            ik_active = False

        # ---------------------------------------------------------------
        # IK step: apply action only for ik_steps steps each episode
        # ---------------------------------------------------------------
        if ik_active:
            ik_step_counter += 1

            # Human pose (WORLD)
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # WORLD poses (BASE + EE-parent)
            base_w = robot.data.root_link_state_w[:, 0:7]
            ee_parent_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]

            R21 = np.array(
                [
                    [0.0, 0.0, 1.0],  # x' = z
                    [1.0, 0.0, 0.0],  # y' = x
                    [0.0, 1.0, 0.0],  # z' = y
                ],
                dtype=float,
            )

            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            wrist_pos_w, wrist_quat_w = ee_parent_w[:, 0:3], ee_parent_w[:, 3:7]

            ee_pos_w = wrist_pos_w
            ee_quat_w = wrist_quat_w  # orientation of the virtual EE (robot frame)

            # Rotation EE -> US in torch (for the US slicer)
            RotMat_torch = (
                torch.as_tensor(R21, dtype=torch.float32, device=sim.device)
                .unsqueeze(0)
                .expand(scene.num_envs, -1, -1)
            )  # (N, 3, 3)

            # Current EE rotation in WORLD as matrix (robot frame)
            ee_rotmat_w = matrix_from_quat(ee_quat_w)  # (N, 3, 3)

            # Map current EE orientation to US frame for the slicer
            us_rotmat_w = torch.matmul(ee_rotmat_w, RotMat_torch)
            ee_quat_w_us = quat_from_matrix(us_rotmat_w)  # (N, 4)

            # Update US image given current EE pose (world) and human pose
            US_slicer.slice_US(
                world_to_human_pos,
                world_to_human_rot,
                ee_pos_w,
                ee_quat_w_us,
            )
            if sim_cfg["vis_us"]:
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg["vis_seg_map"]:
                US_slicer.update_plotter(
                    world_to_human_pos,
                    world_to_human_rot,
                    ee_pos_w,
                    ee_quat_w_us,
                )

            # ------------------------------------------------------------------
            # COMMON TARGET: HUMAN FRAME → WORLD (US) → ROBOT EE FRAME
            #   Phase 1 (first mid_phase_steps): go to lifted pose (z + 10 cm)
            #   Phase 2 (remaining steps): go to classical contact pose
            # ------------------------------------------------------------------
            # Contact target in WORLD from human-frame target
            world_to_ee_target_pos_contact, world_to_ee_target_quat_us = combine_frame_transforms(
                world_to_human_pos,
                world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # Apply 10 cm lift in WORLD.z only during the first phase
            if ik_step_counter <= mid_phase_steps:
                world_to_ee_target_pos = world_to_ee_target_pos_contact.clone()
                world_to_ee_target_pos[:, 2] += 0.10  # +10 cm in world z
            else:
                world_to_ee_target_pos = world_to_ee_target_pos_contact

            # Convert target orientation from US frame back to robot EE frame
            world_to_ee_target_rot_us = matrix_from_quat(world_to_ee_target_quat_us)
            world_to_ee_target_rot_robot = torch.matmul(
                world_to_ee_target_rot_us,
                RotMat_torch.transpose(1, 2),
            )
            world_to_ee_target_quat = quat_from_matrix(world_to_ee_target_rot_robot)

            # Convert target EE pose to BASE frame
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_to_ee_target_pos,
                world_to_ee_target_quat,
            )

            # ------------------------------------------------------------------
            # IK BRANCHES
            # ------------------------------------------------------------------
            if ik_mode == "pink":
                # Pink IK: one IK solve per env via Pinocchio
                base_to_ee_target_pos_np = _t2np(base_to_ee_target_pos)  # (N, 3)
                base_to_ee_target_rot_np = _t2np(
                    matrix_from_quat(base_to_ee_target_quat)
                )  # (N, 3, 3)

                num_envs = scene.num_envs
                num_arm_joints = len(robot_entity_cfg.joint_ids)
                arm_q_all = np.zeros((num_envs, num_arm_joints), dtype=float)

                for env_id in range(num_envs):
                    T_target_i = pin.SE3(
                        base_to_ee_target_rot_np[env_id],
                        base_to_ee_target_pos_np[env_id],
                    )
                    ee_task.set_target(T_target_i)

                    q_curr_isaac_i = robot.data.joint_pos[env_id]
                    q_curr_pin_i = isaac_to_pin_q(
                        q_curr_isaac_i, joint_names, model, name_to_qidx
                    )

                    configurations[env_id].update(q_curr_pin_i)
                    vel_i = solve_ik(
                        configurations[env_id],
                        tasks=ik_tasks,
                        dt=sim_dt,
                        solver="quadprog",
                        damping=1e-2,
                        safety_break=False,
                    )
                    configurations[env_id].integrate_inplace(vel_i, sim_dt)
                    q_next_pin_i = configurations[env_id].q.copy()

                    q_next_isaac_i = pin_to_isaac_q(
                        q_next_pin_i, joint_names, name_to_qidx
                    )

                    arm_q_all[env_id, :] = q_next_isaac_i[robot_entity_cfg.joint_ids]

                arm_q_t = torch.from_numpy(arm_q_all).float().to(sim.device)
                robot.set_joint_position_target(
                    arm_q_t,
                    joint_ids=torch.tensor(
                        robot_entity_cfg.joint_ids,
                        device=sim.device,
                        dtype=torch.long,
                    ),
                )

            elif ik_mode == "dik":
                # Differential IK: use base-frame Jacobian
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    base_pos_w,
                    base_quat_w,
                    ee_parent_w[:, 0:3],
                    ee_parent_w[:, 3:7],
                )

                base_to_ee_target_pose = torch.cat(
                    [base_to_ee_target_pos, base_to_ee_target_quat],
                    dim=-1,
                )

                pose_diff_ik_controller.set_command(base_to_ee_target_pose)

                US_jacobian = robot.root_physx_view.get_jacobians()[
                    :, US_ee_jacobi_idx - 1, :, robot_entity_cfg.joint_ids
                ]

                base_RotMat = matrix_from_quat(quat_inv(base_quat_w))  # [N,3,3]
                US_jacobian[:, 0:3, :] = torch.bmm(
                    base_RotMat, US_jacobian[:, 0:3, :]
                )
                US_jacobian[:, 3:6, :] = torch.bmm(
                    base_RotMat, US_jacobian[:, 3:6, :]
                )

                US_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

                joint_pos_des = pose_diff_ik_controller.compute(
                    ee_pos_b,
                    ee_quat_b,
                    US_jacobian,
                    US_joint_pos,
                )

                robot.set_joint_position_target(
                    joint_pos_des,
                    joint_ids=torch.tensor(
                        robot_entity_cfg.joint_ids,
                        device=sim.device,
                        dtype=torch.long,
                    ),
                )

        # ------------------------------------------------------------------
        # DEBUG + LOGGING (ERRORS IN EE FRAME, env 0)
        # ------------------------------------------------------------------
        if "world_to_ee_target_pos" in locals():
            ee_state_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]
            ee_pos_w_t = ee_state_w[:, 0:3]
            ee_quat_w_t = ee_state_w[:, 3:7]

            err_pos_ee_t, err_quat_ee_t = subtract_frame_transforms(
                ee_pos_w_t,
                ee_quat_w_t,
                world_to_ee_target_pos,
                world_to_ee_target_quat,
            )

            # Error for env 0 (for detailed print)
            pos_err_vec = err_pos_ee_t[0].detach().cpu().numpy()
            err_quat_wxyz = err_quat_ee_t[0].detach().cpu().numpy()
            err_quat_xyzw = isaac_to_scipy_quat(err_quat_wxyz)
            euler_err_vec = R.from_quat(err_quat_xyzw).as_euler("xyz", degrees=True)

            pos_err_norm = np.linalg.norm(pos_err_vec)
            ang_err_norm = np.linalg.norm(euler_err_vec)

            # Max error over all envs
            pos_err_all = err_pos_ee_t.detach().cpu().numpy()          # (N, 3)
            pos_norms_all = np.linalg.norm(pos_err_all, axis=1)        # (N,)
            pos_err_max = pos_norms_all.max()
            idx_max_pos = int(np.argmax(pos_norms_all))

            err_quat_all = err_quat_ee_t.detach().cpu().numpy()        # (N, 4) wxyz
            w_all = np.clip(np.abs(err_quat_all[:, 0]), 0.0, 1.0)
            ang_rad_all = 2.0 * np.arccos(w_all)
            ang_deg_all = np.degrees(ang_rad_all)
            ang_err_max = ang_deg_all.max()
            idx_max_ang = int(np.argmax(ang_deg_all))

            t_now = global_time
            time_hist.append(t_now)
            pos_err_hist.append(pos_err_vec)
            ang_err_hist.append(euler_err_vec)

            # distance probe-body (env 0)
            human_pos = human.data.root_state_w[:, 0:3]
            tip_pos = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:3]
            dist = torch.norm(human_pos - tip_pos, dim=-1)

            if step_i % 60 == 0:
                target_pos_w_dbg = world_to_ee_target_pos[0].detach().cpu().numpy()
                target_quat_dbg = world_to_ee_target_quat[0].detach().cpu().numpy()
                target_euler_dbg = R.from_quat(target_quat_dbg).as_euler(
                    "xyz", degrees=True
                )

                ee_pos_w_dbg = ee_pos_w_t[0].detach().cpu().numpy()
                ee_quat_dbg = ee_quat_w_t[0].detach().cpu().numpy()
                ee_euler_dbg = R.from_quat(ee_quat_dbg).as_euler("xyz", degrees=True)

                print("\n[DBG] ---- EE pose (WORLD) ----")
                print(f"[MODE] ik_mode                     = {ik_mode}")
                print(f"[ROB] robot_type                   = {ROBOT_TYPE}")
                print(f"[CMD] target_pos_w                 = {target_pos_w_dbg}")
                print(f"[CMD] target_euler_w               = {target_euler_dbg}")
                print(f"[SIM] ee_pos_w (env 0)             = {ee_pos_w_dbg}")
                print(f"[SIM] ee_euler_w (env 0)           = {ee_euler_dbg}")
                print(f"[ERR] pos_err_vec (EE, env 0)      = {pos_err_vec} m")
                print(f"[ERR] ang_err_vec (EE, env 0)      = {euler_err_vec} deg")
                print(f"[ERR] ||pos_err|| (env 0)          = {pos_err_norm:.4f} m")
                print(f"[ERR] ||ang_err|| (env 0)          = {ang_err_norm:.2f} deg")
                print(f"[ERR] max ||pos_err|| over envs    = {pos_err_max:.4f} m (env {idx_max_pos})")
                print(f"[ERR] max ang_err over envs        = {ang_err_max:.2f} deg (env {idx_max_ang})")
                print(f"distance probe-body (env 0)        = {dist.to('cpu').numpy()[0]:.4f} m")

        # ---------------------------------------------------------------
        # PLOTTING: errors vs time
        # ---------------------------------------------------------------
        if step_i % 10 == 0 and len(time_hist) > 0:
            pos_arr = np.stack(pos_err_hist, axis=0)
            ang_arr = np.stack(ang_err_hist, axis=0)

            ax_pos.clear()
            ax_pos.plot(time_hist, pos_arr[:, 0], label="ex (EE)")
            ax_pos.plot(time_hist, pos_arr[:, 1], label="ey (EE)")
            ax_pos.plot(time_hist, pos_arr[:, 2], label="ez (EE)")
            ax_pos.set_ylabel("Pos err [m] (EE frame)")
            ax_pos.grid(True)

            ax_ang.clear()
            ax_ang.plot(time_hist, ang_arr[:, 0], label="eroll (EE)")
            ax_ang.plot(time_hist, ang_arr[:, 1], label="epitch (EE)")
            ax_ang.plot(time_hist, ang_arr[:, 2], label="eyaw (EE)")
            ax_ang.set_xlabel("Time [s]")
            ax_ang.set_ylabel("Ang err [deg] (EE frame)")
            ax_ang.grid(True)

            for t_stop in ik_stop_times:
                ax_pos.axvline(t_stop, linestyle="--", linewidth=1.0)
                ax_ang.axvline(t_stop, linestyle="--", linewidth=1.0)

            ax_pos.legend()
            ax_ang.legend()

            plt.pause(0.001)

        # ---------------------------------------------------------------
        # SCENE UPDATE (always executed)
        # ---------------------------------------------------------------
        scene.write_data_to_sim()
        sim.step()

        # Update global simulation time
        global_time += sim_dt

        # -----------------------------------------------------------
        # Transition: IK was active at previous step and is now inactive
        # -> log target vs reached EE positions in patient frame X–Z.
        # -----------------------------------------------------------
        if ik_active_prev and not ik_active:
            ik_stop_times.append(global_time)

            if last_cmd_target_poses is not None:
                # Human pose in WORLD
                human_world_poses = human.data.root_state_w
                world_to_human_pos = human_world_poses[:, 0:3]
                world_to_human_rot = human_world_poses[:, 3:7]

                # Current EE pose in WORLD (for all envs)
                ee_state_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]
                ee_pos_w_t = ee_state_w[:, 0:3]
                ee_quat_w_t = ee_state_w[:, 3:7]

                # Express current EE pose in HUMAN (patient) frame
                human_to_ee_pos, _ = subtract_frame_transforms(
                    world_to_human_pos,
                    world_to_human_rot,
                    ee_pos_w_t,
                    ee_quat_w_t,
                )

                # Target EE pose is already in human frame: human_ee_target_pos
                # Error in human frame [m]
                delta_pos_human = human_to_ee_pos - human_ee_target_pos  # (N, 3)

                # Convert to cmd units using label_res (voxel size [m])
                # We assume patient_xz_* ranges are in voxel units.
                delta_cmd_xz = _t2np(delta_pos_human[:, [0, 2]] / label_res)  # (N, 2)

                # Target cmd coordinates (x,z) from sampled cmd pose
                target_cmd_xz = _t2np(last_cmd_target_poses[:, [0, 1]])       # (N, 2)

                # Reached cmd coordinates = target cmd + error expressed in cmd units
                reached_cmd_xz = target_cmd_xz + delta_cmd_xz                 # (N, 2)

                # Append to history (episode-wise)
                patient_targets_all.append(target_cmd_xz)
                patient_reached_all.append(reached_cmd_xz)
                patient_env_ids_all.append(np.arange(scene.num_envs))

                # Concatenate all episodes
                all_targets = np.concatenate(patient_targets_all, axis=0)
                all_reached = np.concatenate(patient_reached_all, axis=0)
                all_envs = np.concatenate(patient_env_ids_all, axis=0)

                # Build colormap per environment id
                num_envs = scene.num_envs
                cmap = plt.get_cmap("tab20")
                # Normalize env indices into [0, 1]
                colors = cmap((all_envs % num_envs) / max(1, num_envs - 1))

                # Update scatter plot (cmd units on axes)
                ax_plane.clear()
                ax_plane.set_title("Targets (x) and reached (dot) in patient plane (cmd units)")
                ax_plane.set_xlabel("cmd x")
                ax_plane.set_ylabel("cmd z")
                ax_plane.grid(True)
                ax_plane.set_aspect("equal", adjustable="box")

                # Draw command range rectangle (patient_xz_range) ----------------
                # Extract the cmd range used for sampling
                (x_min, z_min, _) = sim_cfg["patient_xz_range"][0]
                (x_max, z_max, _) = sim_cfg["patient_xz_range"][1]

                # Draw rectangle edges
                rect_x = [x_min, x_max, x_max, x_min, x_min]
                rect_z = [z_min, z_min, z_max, z_max, z_min]

                ax_plane.plot(
                    rect_x,
                    rect_z,
                    linestyle="--",
                    linewidth=1.5,
                    color="black",
                )
                # ----------------------------------------------------------------

                # Targets: crosses
                ax_plane.scatter(
                    all_targets[:, 0],
                    all_targets[:, 1],
                    marker="x",
                    s=30,
                    c=colors,
                )

                # Reached: dots
                ax_plane.scatter(
                    all_reached[:, 0],
                    all_reached[:, 1],
                    marker="o",
                    s=15,
                    c=colors,
                    alpha=0.8,
                )

                # ------------- Color–env table on the side -----------------
                ax_plane_table.clear()
                ax_plane_table.axis("off")

                # One color per env id
                env_ids = np.arange(num_envs)
                env_colors = cmap(
                    (env_ids % num_envs) / max(1, num_envs - 1)
                )

                # Simple table: one column "env", colored background
                cell_text = [[str(e)] for e in env_ids]
                cell_colours = [[env_colors[i]] for i in range(num_envs)]

                table = ax_plane_table.table(
                    cellText=cell_text,
                    cellColours=cell_colours,
                    colLabels=["env"],
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.2)
                # -----------------------------------------------------------

                plt.pause(0.001)

        # Update previous IK state
        ik_active_prev = ik_active

        step_i += 1
        scene.update(sim_dt)


# Main
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(
        RobotSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=4.0,
            replicate_physics=False,
        )
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
    print(f"[INFO] Setup complete. Running with robot={ROBOT_TYPE}…")

    scene.reset()
    run(sim, scene, label_map_list, ct_map_list)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")

    # close sim app
    simulation_app.close()