# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Spawn SonoGym scene with Unitree G1 (no actions).
Launch Isaac Sim first.
"""

import argparse
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--reset_seconds", type=float, default=8.0, help="Reset every X seconds of *sim time* (<=0 disables).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    subtract_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
)

# SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.kinematics.gt_motion_generator import GTDiscreteMotionGenerator

import pinocchio as pin
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pink.solve_ik import solve_ik

from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import cProfile
import nibabel as nib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Scene parameters from YAML
# -----------------------------------------------------------------------------

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))

# Patient pose
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg["pos"],
    rot=(quat[3], quat[0], quat[1], quat[2]),
)

# Bed pose
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg["pos"],
    rot=(quat[3], quat[0], quat[1], quat[2]),
)
scale_bed = bed_cfg["scale"]

# Datasets
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
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id
    for p_id in patient_ids
]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg["label_res"]
scale = 1 / label_res

# Robot
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")
EE_LINK_NAME = "left_wrist_yaw_link"

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

ROBOT_CFG.init_state.pos = scene_cfg[robot_type]["pos"]
q_xyzw = R.from_euler("z", scene_cfg[robot_type]["yaw"], degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
ROBOT_CFG.init_state.rot = q_wxyz

ROBOT_HEIGHT = scene_cfg[robot_type]["height"]
ROBOT_HEIGHT_IMG = scene_cfg[robot_type]["height_img"]

IK_ENABLE = True

if robot_type == "g1":
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif robot_type == "h1":
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_body29_hand14.urdf"
else:
    raise ValueError(f"Unsupported robot_type for URDF selection: {robot_type!r}")

# Motion planning config
motion_plan_cfg = scene_cfg["motion_planning"]
use_vertebra_goal = motion_plan_cfg["use_vertebra_goal"]
target_anatomy = patient_cfg["target_anatomy"]

target_stl_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id + "/" + str(target_anatomy) + ".stl"
    for p_id in patient_ids
]

target_traj_file_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/"
    + p_id
    + "/"
    + "standard_right_traj_"
    + str(target_anatomy)[-2:]
    + ".stl"
    for p_id in patient_ids
]


# -----------------------------------------------------------------------------
# Scene config
# -----------------------------------------------------------------------------

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal SonoGym scene + Unitree robot."""

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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _t2np(t: torch.Tensor):
    return t.detach().cpu().numpy()

def isaac_to_scipy_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert IsaacLab [w, x, y, z] quaternion to SciPy [x, y, z, w]."""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)

def build_pinocchio_model(urdf_path: str):
    """Build Pinocchio model and data from URDF path."""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

def build_name_to_qidx(model: pin.Model):
    """Build a mapping from joint name to configuration index in Pinocchio q."""
    name_to_qidx = {}
    for j_id, joint in enumerate(model.joints):
        name = model.names[j_id]
        if joint.nq == 1:
            name_to_qidx[name] = joint.idx_q
    return name_to_qidx

def isaac_to_pin_q(q_isaac: torch.Tensor, isaac_joint_names: list[str], model: pin.Model, name_to_qidx: dict[str, int]) -> np.ndarray:
    """Map IsaacLab joint positions to Pinocchio configuration vector q."""
    q_pin = np.zeros(model.nq, dtype=float)
    q_isaac_np = q_isaac.detach().cpu().numpy()
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            idx_q = name_to_qidx[jname]
            q_pin[idx_q] = q_isaac_np[i]
    return q_pin

def pin_to_isaac_q(q_pin: np.ndarray, isaac_joint_names: list[str], name_to_qidx: dict[str, int]) -> np.ndarray:
    """Map Pinocchio configuration q back to IsaacLab joint vector ordering."""
    q_isaac = np.zeros(len(isaac_joint_names), dtype=float)
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            idx_q = name_to_qidx[jname]
            q_isaac[i] = q_pin[idx_q]
    return q_isaac


# -----------------------------------------------------------------------------
# Run loop
# -----------------------------------------------------------------------------

def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list | None = None):

    robot = scene[ROBOT_KEY]
    human = scene["human"]

    model, data = build_pinocchio_model(URDF_PATH)
    name_to_qidx = build_name_to_qidx(model)
    configuration: Configuration | None = None

    SIDE = "left" if "left" in EE_LINK_NAME else ("right" if "right" in EE_LINK_NAME else None)
    joint_pattern = rf"(waist_(pitch|roll|yaw)_joint|{SIDE}_(shoulder|elbow|wrist)_.*)" if SIDE else r".*"
    robot_entity_cfg = SceneEntityCfg(ROBOT_KEY, joint_names=[joint_pattern], body_names=[EE_LINK_NAME])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r"))
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
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
        roll_adj=scene_cfg["motion_planning"]["US_roll_adj"],
        visualize=sim_cfg["vis_seg_map"],
        height=ROBOT_HEIGHT,
        height_img=ROBOT_HEIGHT_IMG,
    )


    vertebra_viewer = VertebraViewer(
        scene.num_envs,
        len(human_usd_list),
        target_stl_file_list,
        target_traj_file_list,
        False,
        label_res,
        sim.device,
    )

    goal_pose_ph = motion_plan_cfg["patient_xz_goal"]  # [x, z, roll] in human frame (y is ignored)

    gt_motion_generator = GTDiscreteMotionGenerator(
        goal_cmd_pose=goal_pose_ph,
        scale=torch.tensor(motion_plan_cfg["scale"], device=sim.device),
        num_envs=scene.num_envs,
        surface_map_list=US_slicer.surface_map_list,
        surface_normal_list=US_slicer.surface_normal_list,
        label_res=label_res,
        US_height=US_slicer.height,
    )


    vertebra_to_us_2d_pos = torch.tensor(
        motion_plan_cfg["vertebra_to_US_2d_pos"],
        device=sim.device,
        dtype=torch.float32,
    )

    def get_us_target_pose_current() -> torch.Tensor:
        vertebra_2d_pos = vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
        us_target_2d_pos = vertebra_2d_pos + vertebra_to_us_2d_pos.unsqueeze(0)
        us_target_2d_angle = 1.57 * torch.ones_like(vertebra_2d_pos[:, 0:1])
        return torch.cat([us_target_2d_pos, us_target_2d_angle], dim=-1)

    root_pos0 = None
    root_rot0 = None
    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = robot.data.default_joint_vel.clone() * 0.0

    sim_dt = sim.get_physics_dt()
    step_i = 0
    reset_T = float(args_cli.reset_seconds)
    sim_time_acc = 0.0

    pos_err_hist = []
    ang_err_hist = []
    time_hist = []

    plt.ion()
    fig, (ax_pos, ax_ang) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("EE tracking errors")

    ee_task: FrameTask | None = None
    ik_tasks = []

    while simulation_app.is_running():
        if step_i == 0:
            root_state = robot.data.root_state_w.clone()
            root_pos0 = root_state[:, 0:3].clone()
            root_rot0 = root_state[:, 3:7].clone()

            joint_names = list(robot.data.joint_names)
            default_pos = robot.data.default_joint_pos.clone()
            zero_vel = robot.data.default_joint_vel.clone() * 0.0

            robot.reset()
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            scene.reset()

            q0_isaac = robot.data.joint_pos[0]
            q0_pin = isaac_to_pin_q(q0_isaac, joint_names, model, name_to_qidx)
            configuration = Configuration(model, data, q0_pin)

            ee_task = FrameTask(
                frame=EE_LINK_NAME,
                position_cost=1.0,
                orientation_cost=1.0,
            )
            ik_tasks = [ee_task]

        if IK_ENABLE and step_i > 0:
            # Human pose
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # Base + wrist in world
            base_w = robot.data.root_link_state_w[:, 0:7]
            ee_parent_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]

            R21 = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )

            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            wrist_pos_w, wrist_quat_w = ee_parent_w[:, 0:3], ee_parent_w[:, 3:7]

            ee_pos_w = wrist_pos_w
            ee_quat_w = wrist_quat_w

            RotMat_torch = torch.as_tensor(
                R21, dtype=torch.float32, device=sim.device
            ).unsqueeze(0).expand(scene.num_envs, -1, -1)

            ee_rotmat_w = matrix_from_quat(ee_quat_w)
            us_rotmat_w = torch.matmul(ee_rotmat_w, RotMat_torch)
            ee_quat_w_us = quat_from_matrix(us_rotmat_w)

            # Compute vertebra-based target exactly once from the current anatomy state
            goal_cmd_pose_current = get_us_target_pose_current()

            # Write the target command directly into the slicer state
            US_slicer.current_x_z_x_angle_cmd = goal_cmd_pose_current

            world_to_ee_target_pos, world_to_ee_target_quat_us = US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos,
                world_to_human_rot,
            )

            # Update US image
            US_slicer.slice_US(
                world_to_human_pos,
                world_to_human_rot,
                ee_pos_w,
                ee_quat_w_us,
            )

            if sim_cfg["vis_us"]:
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg["vis_seg_map"]:
                US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, ee_pos_w, ee_quat_w_us)

            # Compute EE target in world (US frame)
            world_to_ee_target_pos, world_to_ee_target_quat_us = US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos,
                world_to_human_rot,
            )

            # Convert orientation from US frame back to robot EE frame
            world_to_ee_target_mat = matrix_from_quat(world_to_ee_target_quat_us)
            world_to_ee_target_rot = torch.matmul(world_to_ee_target_mat, RotMat_torch.transpose(1, 2))
            world_to_ee_target_quat = quat_from_matrix(world_to_ee_target_rot)

            # Convert target to base frame
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_to_ee_target_pos,
                world_to_ee_target_quat,
            )

            target_pos_np = _t2np(base_to_ee_target_pos)[0].astype(float)
            R_target = _t2np(matrix_from_quat(base_to_ee_target_quat))[0].astype(float)
            T_target = pin.SE3(R_target, target_pos_np)
            ee_task.set_target(T_target)

            q_curr_isaac = robot.data.joint_pos[0]
            q_curr_pin = isaac_to_pin_q(q_curr_isaac, joint_names, model, name_to_qidx)
            configuration.update(q_curr_pin)

            vel = solve_ik(
                configuration,
                tasks=ik_tasks,
                dt=sim_dt,
                solver="quadprog",
                damping=1e-2,
                safety_break=False,
            )
            configuration.integrate_inplace(vel, sim_dt)
            q_next_pin = configuration.q.copy()

            q_next_isaac = pin_to_isaac_q(q_next_pin, joint_names, name_to_qidx)
            arm_q_np = q_next_isaac[robot_entity_cfg.joint_ids]
            arm_q_t = torch.from_numpy(arm_q_np).float().to(sim.device).unsqueeze(0)

            robot.set_joint_position_target(
                arm_q_t,
                joint_ids=torch.tensor(robot_entity_cfg.joint_ids, device=sim.device, dtype=torch.long),
            )

            # Debug
            ee_state_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]
            ee_pos_w_t = ee_state_w[:, 0:3]
            ee_quat_w_t = ee_state_w[:, 3:7]

            err_pos_ee_t, err_quat_ee_t = subtract_frame_transforms(
                ee_pos_w_t,
                ee_quat_w_t,
                world_to_ee_target_pos,
                world_to_ee_target_quat,
            )

            pos_err_vec = err_pos_ee_t[0].detach().cpu().numpy()
            err_quat_wxyz = err_quat_ee_t[0].detach().cpu().numpy()
            err_quat_xyzw = isaac_to_scipy_quat(err_quat_wxyz)
            euler_err_vec = R.from_quat(err_quat_xyzw).as_euler("xyz", degrees=True)

            pos_err_norm = np.linalg.norm(pos_err_vec)
            ang_err_norm = np.linalg.norm(euler_err_vec)

            t_now = step_i * sim_dt
            time_hist.append(t_now)
            pos_err_hist.append(pos_err_vec)
            ang_err_hist.append(euler_err_vec)

            human_pos = human.data.root_state_w[:, 0:3]
            tip_pos = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:3]
            dist = torch.norm(human_pos - tip_pos, dim=-1)

            cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
                world_to_human_pos,
                world_to_human_rot,
                ee_pos_w,
                ee_quat_w_us,
            )

            cur_cmd_pose = gt_motion_generator.human_cmd_state_from_ee_pose(
                cur_human_ee_pos, cur_human_ee_quat
            )

            if step_i % 10 == 0 and len(time_hist) > 0:
                pos_arr = np.stack(pos_err_hist, axis=0)
                ang_arr = np.stack(ang_err_hist, axis=0)

                ax_pos.clear()
                ax_pos.plot(time_hist, pos_arr[:, 0], label="ex (EE)")
                ax_pos.plot(time_hist, pos_arr[:, 1], label="ey (EE)")
                ax_pos.plot(time_hist, pos_arr[:, 2], label="ez (EE)")
                ax_pos.set_ylabel("Pos err [m] (EE frame)")
                ax_pos.grid(True)
                ax_pos.legend()

                ax_ang.clear()
                ax_ang.plot(time_hist, ang_arr[:, 0], label="eroll (EE)")
                ax_ang.plot(time_hist, ang_arr[:, 1], label="epitch (EE)")
                ax_ang.plot(time_hist, ang_arr[:, 2], label="eyaw (EE)")
                ax_ang.set_xlabel("Time [s]")
                ax_ang.set_ylabel("Ang err [deg] (EE frame)")
                ax_ang.grid(True)
                ax_ang.legend()

                plt.pause(0.001)

            if step_i % 60 == 0:
                target_pos_w_dbg = world_to_ee_target_pos[0].detach().cpu().numpy()
                target_quat_dbg = world_to_ee_target_quat[0].detach().cpu().numpy()
                target_euler_dbg = R.from_quat(target_quat_dbg).as_euler("xyz", degrees=True)

                ee_pos_w_dbg = ee_pos_w_t[0].detach().cpu().numpy()
                ee_quat_dbg = ee_quat_w_t[0].detach().cpu().numpy()
                ee_euler_dbg = R.from_quat(ee_quat_dbg).as_euler("xyz", degrees=True)

                print("\n[DBG] ---- EE pose (WORLD) ----")
                print(f"[CMD] target_pos_w        = {target_pos_w_dbg}")
                print(f"[CMD] target_euler_w      = {target_euler_dbg}")
                print(f"[SIM] ee_pos_w            = {ee_pos_w_dbg}")
                print(f"[SIM] ee_euler_w          = {ee_euler_dbg}")
                print(f"[ERR] pos_err_vec (EE)    = {pos_err_vec} m")
                print(f"[ERR] ang_err_vec (EE)    = {euler_err_vec} deg")
                print(f"[ERR] ||pos_err||         = {pos_err_norm:.4f} m")
                print(f"[ERR] ||ang_err||         = {ang_err_norm:.2f} deg")
                print(f"distance probe-body       = {dist.to('cpu').numpy()[0]:.4f} m")
                print(f"current_cmd_pose          = {cur_cmd_pose.detach().cpu().numpy()[0]}")

        scene.write_data_to_sim()
        sim.step()
        step_i += 1
        scene.update(sim_dt)
        sim_time_acc += sim_dt

        if reset_T > 0.0 and sim_time_acc >= reset_T:
            root_state = robot.data.root_state_w.clone()
            root_state[:, 0:3] = root_pos0
            root_state[:, 3:7] = root_rot0
            root_state[:, 7:13] *= 0.0
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()
            sim_time_acc = 0.0
            print(f"[INFO] Soft reset at sim_t={step_i * sim_dt:.2f}s")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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