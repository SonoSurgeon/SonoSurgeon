# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Spawn SonoGym scene with Unitree G1 (Pink IK controller, parallel envs).
Launch Isaac Sim first.
"""

import argparse
from isaaclab.app import AppLauncher

# --- CLI
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--reset_seconds", type=float, default=8.0,
                    help="Reset every X seconds of *sim time* (<=0 disables).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab imports
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

# --- SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *  
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer  # (unused but kept)
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer    # (unused but kept)
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner  # (unused but kept)

# --- Pink / Pinocchio IK imports -------------------------------------------
import pinocchio as pin
import pink
from pink.configuration import Configuration
from pink.tasks import FrameTask, PostureTask
from pink.solve_ik import solve_ik
# ---------------------------------------------------------------------------

# --- Other libs
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import cProfile
import nibabel as nib
import numpy as np
import os
import torch
import time
import logging

class JointLimitFilter(logging.Filter):
    """Filter out joint limit warnings on the root logger."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Drop messages that match the specific joint limit pattern
        if "is out of limits" in msg:# and "Value" in msg:
            return False  # do not log this record
        return True       # keep all other log records

root_logger = logging.getLogger()  # this is the root logger
root_logger.addFilter(JointLimitFilter())


########### Scene parameters from YAML (bed + patient) ##############

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))

# patient pose
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

# use stl: Totalsegmentator_dataset_v2_subset_body_contact
human_usd_list = [
    f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id
    for p_id in patient_ids
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
scale = 1 / label_res

# robot
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")  # default: g1 if not specified

# Common EE link name for both robots 
EE_LINK_NAME = "left_wrist_yaw_link"

if robot_type == "g1":
    # Use G1 config
    ROBOT_CFG: ArticulationCfg = G1_TOOLS_BASE_FIX_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/G1"
    ROBOT_KEY = "g1"
elif robot_type == "h1":
    # Use H1-2 config without hands 
    ROBOT_CFG: ArticulationCfg = H12_CFG_TOOLS_BASEFIX.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/H1"
    ROBOT_KEY = "h1"
else:
    raise ValueError(f"Unknown robot type in YAML: {robot_type!r} (expected 'g1' or 'h1').")

# Override init pose + orientation from YAML robot section 
ROBOT_CFG.init_state.pos = scene_cfg[robot_type]["pos"]
q_xyzw = R.from_euler("z", scene_cfg[robot_type]["yaw"], degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # xyzw → wxyz
ROBOT_CFG.init_state.rot = q_wxyz

ROBOT_HEIGHT = scene_cfg[robot_type]["height"]
ROBOT_HEIGHT_IMG = scene_cfg[robot_type]["height_img"]

# IK controller settings
IK_ENABLE = True

# NOTE: adjust the H1 URDF path to match your local asset layout.
if robot_type == "g1":
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif robot_type == "h1":
    # TODO: replace this with the correct H1 URDF path if different
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_body29_hand14.urdf"       # STILL NEED THIS ONE
else:
    raise ValueError(f"Unsupported robot_type for URDF selection: {robot_type!r}")

# Scene config
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal SonoGym scene + Unitree G1 (idle)."""

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

    # robot: defined depending on robot_type
    if robot_type == "g1":
        g1 = ROBOT_CFG
    else:
        h1 = ROBOT_CFG

# Helpers: torch<->numpy bridges

def _t2np(t: torch.Tensor):
    """Convert a torch tensor to numpy (detach + cpu)."""
    return t.detach().cpu().numpy()


# Pink / Pinocchio helpers

def build_pinocchio_model(urdf_path: str):
    """Build Pinocchio model and data from URDF path."""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def build_name_to_qidx(model: pin.Model):
    """Build a mapping from joint name to configuration index in Pinocchio q.

    We only consider 1-DoF joints (revolute, prismatic). Floating base is ignored.
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
def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):

    robot = scene[ROBOT_KEY]
    human = scene["human"]

    # Pink / Pinocchio IK setup (once)
    model, data = build_pinocchio_model(URDF_PATH)
    name_to_qidx = build_name_to_qidx(model)

    configuration: Configuration | None = None
    ee_task: FrameTask | None = None

    # get US probe index
    SIDE = "left" if "left" in EE_LINK_NAME else ("right" if "right" in EE_LINK_NAME else None)
    joint_pattern = rf"(waist_(pitch|roll|yaw)_joint|{SIDE}_(shoulder|elbow|wrist)_.*)" if SIDE else r".*"
    robot_entity_cfg = SceneEntityCfg(ROBOT_KEY, joint_names=[joint_pattern], body_names=[EE_LINK_NAME])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1] 

    # resolve both wrist links (if needed for debugging)
    robot_ee_both_cfg = SceneEntityCfg(ROBOT_KEY, joint_names=[], body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"])
    robot_ee_both_cfg.resolve(scene)
    left_wrist_id, right_wrist_id = robot_ee_both_cfg.body_ids

    # Construct label image slicer conversion map
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r"))

    # Construct US simulator
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
        visualize=sim_cfg["vis_seg_map"],
        height_img=ROBOT_HEIGHT_IMG,
        height=ROBOT_HEIGHT,
    )

    # scene reset pose
    root_pos0 = None
    root_rot0 = None
    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = robot.data.default_joint_vel.clone() * 0.0

    # time
    sim_dt = sim.get_physics_dt()
    step_i = 0
    reset_T = float(args_cli.reset_seconds)
    sim_time_acc = 0.0
    clock = time.time()
    resets = 0

    while simulation_app.is_running():
        if step_i == 0:
            # cache for env reset
            root_state = robot.data.root_state_w.clone()
            root_pos0 = root_state[:, 0:3].clone()
            root_rot0 = root_state[:, 3:7].clone()
            human_root_pose = human.data.body_link_state_w[:, 0, 0:7]

            # joint names and default poses
            joint_names = list(robot.data.joint_names)
            default_pos = robot.data.default_joint_pos.clone()
            zero_vel = robot.data.default_joint_vel.clone() * 0.0

            # Reset articulation and apply new state
            robot.reset()
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            scene.reset()

            # Initialize Pink configuration and task using env 0 as reference
            q0_isaac = robot.data.joint_pos[0]  # (n_joints,)
            q0_pin = isaac_to_pin_q(q0_isaac, joint_names, model, name_to_qidx)
            configuration = Configuration(model, data, q0_pin)

            ee_task = FrameTask(
                frame=EE_LINK_NAME,
                position_cost=1.0,
                orientation_cost=1.0,
            )


        if IK_ENABLE and step_i > 1:
            # Human pose
            human_world_poses = human.data.body_link_state_w[:, 0, 0:7]
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # WORLD poses (BASE + EE-parent)
            base_w = robot.data.root_link_state_w[:, 0:7]
            ee_parent_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]

            # Rotation EE -> US in world
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
            ee_quat_w = wrist_quat_w  # orientation of virtual EE = wrist

            # Build rotation matrix tensor
            RotMat_torch = (
                torch.as_tensor(R21, dtype=torch.float32, device=sim.device)
                .unsqueeze(0)
                .expand(scene.num_envs, -1, -1)
            )  # (N, 3, 3)

            # Current EE rotation in WORLD as matrix
            ee_rotmat_w = matrix_from_quat(ee_quat_w)  # (N, 3, 3)

            # Assuming RotMat maps EE frame -> US frame: R_W^US = R_W^EE * R_EE^US
            us_rotmat_w = torch.matmul(ee_rotmat_w, RotMat_torch)

            # Back to quaternion for the slicer
            ee_quat_w_us = quat_from_matrix(us_rotmat_w)  # (N, 4)

            # Sample random command in US frame
            rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device) * 2.0 - 1.0
            rand_x_z_angle[:, 2] = rand_x_z_angle[:, 2] / 10.0
            US_slicer.update_cmd(rand_x_z_angle)

            # Update US image given current EE pose (world) and human pose
            US_slicer.slice_US(world_to_human_pos, world_to_human_rot, ee_pos_w, ee_quat_w_us)
            if sim_cfg["vis_us"]:
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg["vis_seg_map"]:
                US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, ee_pos_w, ee_quat_w_us)

            # Compute target EE pose in WORLD from the command (US frame)
            world_to_ee_target_pos, world_to_ee_target_quat = US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos,
                world_to_human_rot,
            )

            # Convert target orientation from US frame back to EE frame for IK
            world_to_ee_target_mat = matrix_from_quat(world_to_ee_target_quat)
            world_to_ee_target_rot = torch.matmul(
                world_to_ee_target_mat, RotMat_torch.transpose(1, 2)
            )
            world_to_ee_target_quat_ee = quat_from_matrix(world_to_ee_target_rot)

            # Convert target EE pose to BASE frame
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_to_ee_target_pos,
                world_to_ee_target_quat_ee,
            )

            # Pink IK solve for each env in parallel loop (Isaac envs)
            arm_q_list = []
            for env_id in range(scene.num_envs):
                # Target pose in BASE frame for this env
                target_pos_np = _t2np(base_to_ee_target_pos[env_id : env_id + 1])[0]
                R_target = _t2np(matrix_from_quat(base_to_ee_target_quat[env_id : env_id + 1]))[0]

                T_target = pin.SE3(R_target, target_pos_np)

                # Set Pink target
                ee_task.set_target(T_target)

                # Current joint state in this env
                q_curr_isaac = robot.data.joint_pos[env_id]
                q_curr_pin = isaac_to_pin_q(q_curr_isaac, joint_names, model, name_to_qidx)

                configuration.update(q_curr_pin)

                # Solve IK
                vel = solve_ik(
                    configuration,
                    tasks=[ee_task],
                    dt=sim_dt,
                    solver="quadprog",
                    damping=1e-3,
                    safety_break=False,
                )
                

                configuration.integrate_inplace(vel, sim_dt)
                q_next_pin = configuration.q.copy()
                q_next_isaac = pin_to_isaac_q(q_next_pin, joint_names, name_to_qidx)

                # Extract only controlled joints (arm) to command PD
                arm_q_np = q_next_isaac[robot_entity_cfg.joint_ids]
                arm_q_list.append(arm_q_np)

            # Build (N, n_ctrl_joints) tensor
            arm_q_np_stacked = np.stack(arm_q_list, axis=0)
            arm_q_t = torch.from_numpy(arm_q_np_stacked).float().to(sim.device)

            joint_limits = robot.data.joint_pos_limits  # shape: (num_envs, num_joints, 2)

            joint_ids_t = torch.tensor(
                robot_entity_cfg.joint_ids,
                device=sim.device,
                dtype=torch.long,
            )

            # We assume limits are the same across envs, so we can take env 0
            joint_min = joint_limits[0, joint_ids_t, 0]  # (n_ctrl_joints,)
            joint_max = joint_limits[0, joint_ids_t, 1]  # (n_ctrl_joints,)

            # Broadcast to (num_envs, n_ctrl_joints)
            joint_min = joint_min.unsqueeze(0).expand_as(arm_q_t)
            joint_max = joint_max.unsqueeze(0).expand_as(arm_q_t)

            safety_margin = 1e-3  

            joint_min = joint_min + safety_margin
            joint_max = joint_max - safety_margin

            # Clamp IK output
            arm_q_t = torch.clamp(arm_q_t, joint_min, joint_max)

            robot.set_joint_position_target(
                arm_q_t,
                joint_ids=torch.tensor(
                    robot_entity_cfg.joint_ids,
                    device=sim.device,
                    dtype=torch.long,
                ),
            )

            # Debug print (single env, e.g. env 0)
            if step_i % 60 == 0:
                # Compute per-environment position error in WORLD frame
                target_pos_w = world_to_ee_target_pos  # (N, 3), torch

                # Current EE positions in WORLD (N, 3)
                ee_state_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]
                ee_pos_w = ee_state_w[:, 0:3]  # (N, 3), torch

                # Position error per env
                pos_err_vec = target_pos_w - ee_pos_w          # (N, 3)
                pos_err = torch.linalg.norm(pos_err_vec, dim=1)  # (N,)

                # Aggregate statistics
                mean_err = pos_err.mean().item()
                max_err = pos_err.max().item()

                print(
                    f"[DBG] EE position error over {scene.num_envs} envs: "
                    f"mean = {mean_err:.4f} m, max = {max_err:.4f} m"
                )

        # SCENE UPDATE
        scene.write_data_to_sim()
        sim.step()
        step_i += 1
        scene.update(sim_dt)
        sim_time_acc += sim_dt

        # Soft reset
        if reset_T > 0.0 and sim_time_acc >= reset_T:
            root_state = robot.data.root_state_w.clone()
            root_state[:, 0:3] = root_pos0
            root_state[:, 3:7] = root_rot0
            root_state[:, 7:13] *= 0.0
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()
            resets =+ 1
            sim_time_acc = 0.0
            print(f"[INFO] Soft reset at sim_t={step_i * sim_dt:.2f}s")
            print(f"[INFO] clock dt: {time.time() - clock:.2f}s")

        
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
    print("[INFO] Setup complete. Running…")

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