# guided_surgery_mock_bimanual_switch_g1h1.py
# Copyright...
# SPDX-License-Identifier: BSD-3-Clause

"""Mock bimanual IK for SonoGym guided surgery with Unitree G1/H1.
- Left arm: tracks a fixed point on the patient's body (human frame offset)
- Right arm: placeholder pose (edit later)
- One DifferentialIKController object, reused for both arms each step

Launch Isaac Sim first.

Usage:
  ./isaaclab.sh -p path/to/guided_surgery_mock_bimanual_switch_g1h1.py --num_envs 1
"""

import argparse
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="SonoGym mock guided surgery (bimanual IK, G1/H1 switch).")
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
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    quat_mul,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *  
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import cProfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

########### Scene parameters from YAML (bed + patient + robot) ##############

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))
sim_cfg = scene_cfg["sim"]
motion_plan_cfg = scene_cfg["motion_planning"]
patient_cfg = scene_cfg["patient"]
target_anatomy = patient_cfg["target_anatomy"]
# PATIENT & BED CFG 

# patient pose
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)

# bed poseq_xyzw = R.from_euler("YXZ", [-90, 0, 90], degrees=True).as_quat().astype(np.float32)
bed_cfg = scene_cfg["bed"]
quat = R.from_euler("xyz", bed_cfg["euler_xyz"], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg["pos"], rot=(quat[3], quat[0], quat[1], quat[2])
)
scale_bed = bed_cfg["scale"]

# datasets (human USD/labels/CT)
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
scale = 1 / label_res

# robot
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")  # default: g1 if not specified

# Common EE link name for both robots
LEFT_EE_NAME = "left_wrist_yaw_link"
RIGHT_EE_NAME = "right_wrist_yaw_link"

if robot_type == "g1":
    # Use G1 config
    ROBOT_CFG: ArticulationCfg = G1_TOOLS_BASE_FIX_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/G1"
    ROBOT_KEY = "g1"
elif robot_type == "h1":
    # Use H1-2 config 
    ROBOT_CFG: ArticulationCfg = H12_TOOLS_SURGERY_CFG.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/H1"
    ROBOT_KEY = "h1"
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

DRILL_TO_TIP_POS = np.array([0.305, 0.0, 0.0]).astype(np.float32)  # -0.135
q_xyzw = R.from_euler("YXZ", [-90, 180, 90], degrees=True).as_quat().astype(np.float32)
DRILL_TO_TIP_QUAT = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

# Tip offset from the controlled wrist link (right_wrist_yaw_link).
# Convention: right_wrist -> tip (expressed in wrist frame)
WRIST_TO_TIP_POS = torch.tensor(DRILL_TO_TIP_POS, dtype=torch.float32)  # (3,)
WRIST_TO_TIP_QUAT_XYZW = torch.tensor(DRILL_TO_TIP_QUAT, dtype=torch.float32)  # (4,) xyzw
WRIST_TO_TIP_QUAT_WXYZ = torch.stack(
    [WRIST_TO_TIP_QUAT_XYZW[3], WRIST_TO_TIP_QUAT_XYZW[0], WRIST_TO_TIP_QUAT_XYZW[1], WRIST_TO_TIP_QUAT_XYZW[2]]
)  # (4,) wxyz

TIP_ALONG_TRAJ = 0.0

IK_ENABLE = True
PLOT_ENABLE = False

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

# --- DEBUG: print waist + right arm joints (env 0) ---
def debug_print_waist_right_joints(robot: Articulation, right_entity_cfg: SceneEntityCfg, every: int = 60):
    # right_entity_cfg.joint_ids are exactly the joints matched by right_joint_pattern
    joint_ids = right_entity_cfg.joint_ids

    # joint names in the same order as joint_ids
    # (Articulation exposes joint names; try .joint_names / .data.joint_names depending on IsaacLab version)
    try:
        all_names = robot.joint_names
    except AttributeError:
        all_names = robot.data.joint_names  # fallback

    names = [all_names[j] for j in joint_ids]

    def _print(step_i: int):
        if step_i % every != 0:
            return
        q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        dq = robot.data.joint_vel[0, joint_ids].detach().cpu().numpy()

        print("\n[DEBUG] env0 waist+right joints:")
        for n, qi, dqi in zip(names, q, dq):
            print(f"  {n:40s}  q={qi:+.5f}   dq={dqi:+.5f}")

    return _print

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal SonoGym scene + Unitree robot (G1 or H1)."""

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

def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    # Select correct articulation key in the scene
    robot_name = "g1" if robot_type == "g1" else "h1"

    # Joint regex for each arm
    left_joint_pattern = rf"(left_(shoulder|elbow|wrist)_.*)"
    right_joint_pattern = rf"(torso_joint|waist_(pitch|roll|yaw)_joint|right_(shoulder|elbow|wrist)_.*)"

    robot: Articulation = scene[robot_name]
    human: RigidObject = scene["human"]

    # Resolve left EE entity
    left_entity_cfg = SceneEntityCfg(
        robot_name,
        joint_names=[left_joint_pattern],
        body_names=[LEFT_EE_NAME],
    )
    left_entity_cfg.resolve(scene)
    left_ee_id = left_entity_cfg.body_ids[-1]

    # Resolve right EE entity
    right_entity_cfg = SceneEntityCfg(
        robot_name,
        joint_names=[right_joint_pattern],
        body_names=[RIGHT_EE_NAME],
    )
    right_entity_cfg.resolve(scene)
    right_ee_id = right_entity_cfg.body_ids[-1]

    # One IK controller object, reused for both arms
    ik_param_l = {"lambda_val": 1e-1}
    ik_cfg_l = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params=ik_param_l,
    )
    diff_ik_l = DifferentialIKController(ik_cfg_l, scene.num_envs, device=sim.device)

    ik_param_r = {"lambda_val": 1e-1}
    ik_cfg_r = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params=ik_param_r,
    )
    diff_ik_r = DifferentialIKController(ik_cfg_r, scene.num_envs, device=sim.device)
    
    # Soft reset caches
    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = robot.data.default_joint_vel.clone() * 0.0
    root_pos0 = None
    root_rot0 = None

    # Right TIP target in base frame (stateful sample-and-hold)
    tip_tgt_pos_b = None
    tip_tgt_quat_b = None

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
    
    # Optional frame visuals
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

    # construct label image slicer
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r"))

    # construct US simulator
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
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
        height=ROBOT_HEIGHT,
        height_img=ROBOT_HEIGHT_IMG,
        visualize=sim_cfg["vis_seg_map"],
    )

    vertebra_viewer = VertebraViewer(
        scene.num_envs,
        len(patient_ids),
        target_stl_file_list,
        target_traj_file_list,
        if_vis=False,
        res=label_res,
        device=sim.device,
    )

    # fixed target on patient for LEFT arm (human frame), from YAML
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

    sim_dt = sim.get_physics_dt()
    step_i = 0
    sim_time_acc = 0.0
    reset_T = float(args_cli.reset_seconds)
    
    # ----------------------------
    # Full joint set for a single set_joint_position_target call
    # Note: left includes waist joints; right is arm-only (as per your current patterns)
    full_joint_ids_list = sorted(set(left_entity_cfg.joint_ids + right_entity_cfg.joint_ids))
    full_joint_ids = torch.tensor(full_joint_ids_list, device=sim.device, dtype=torch.long)

    # Column indices in the full vector corresponding to left/right joint ids (keeps original order)
    _jid_to_col = {jid: i for i, jid in enumerate(full_joint_ids_list)}
    left_full_cols = torch.tensor([_jid_to_col[j] for j in left_entity_cfg.joint_ids], device=sim.device, dtype=torch.long)
    right_full_cols = torch.tensor([_jid_to_col[j] for j in right_entity_cfg.joint_ids], device=sim.device, dtype=torch.long)

    print_wr = debug_print_waist_right_joints(robot, right_entity_cfg, every=60)  # ogni 60 step
    # ----------------------------
    # Debug buffers for EE tracking (env 0)
    # ----------------------------
    if PLOT_ENABLE:
        t_hist = []
        l_pos_err_hist = []
        l_ang_err_hist = []
        r_pos_err_hist = []
        r_ang_err_hist = []

        PLOT_EVERY = 20           # steps

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle("EE Tracking Errors (env 0)")
        plt.ion()
        plt.show()

    while simulation_app.is_running():
        if step_i % sim_cfg["episode_length"] == 0:
            # reset robot joint state
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()

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

        
            # reset IK and set neutral command (hold current pose)
            diff_ik_l.reset()
            ik_commands_pose_l = torch.zeros(
                scene.num_envs,
                diff_ik_l.action_dim,
                device=sim.device,
            )
            diff_ik_l.set_command(ik_commands_pose_l, left_ee_pos_b, left_ee_quat_b)
            
            diff_ik_r.reset()
            ik_commands_pose_r = torch.zeros(
                scene.num_envs,
                diff_ik_r.action_dim,
                device=sim.device,
            )
            diff_ik_r.set_command(ik_commands_pose_r, right_ee_pos_b, right_ee_quat_b)

        if IK_ENABLE:
            # Human pose
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            # WORLD poses (BASE + EE)
            base_w = robot.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            
            right_ee_w = robot.data.body_state_w[:, right_ee_id, 0:7]
            left_ee_w = robot.data.body_state_w[:, left_ee_id, 0:7]  
            left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                left_ee_w[:, 0:3], left_ee_w[:, 3:7],
            )

            R21 = np.array(
                [
                    [0.0, 0.0, 1.0],  # x' = z
                    [1.0, 0.0, 0.0],  # y' = x
                    [0.0, 1.0, 0.0],  # z' = y
                ],
                dtype=float,
            )

            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            left_ee_pos_w, left_ee_quat_w = left_ee_w[:, 0:3], left_ee_w[:, 3:7]

            RotMat_torch = torch.as_tensor(R21, dtype=torch.float32, device=sim.device).unsqueeze(0).expand(
                scene.num_envs, -1, -1
            )  # (N, 3, 3)

            # Current EE rotation in WORLD as matrix
            left_ee_rotmat_w = matrix_from_quat(left_ee_quat_w)  # (N, 3, 3)

            # Assuming RotMat maps EE frame -> US frame: R_W^US = R_W^EE * R_EE^US
            left_us_rotmat_w = torch.bmm(left_ee_rotmat_w, RotMat_torch)

            # Back to quaternion for the slicer
            left_us_quat_w = quat_from_matrix(left_us_rotmat_w)  # (N, 4)
            left_us_pos_w = left_ee_pos_w

            US_slicer.current_x_z_x_angle_cmd = US_cmd_hold
            US_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_rot)

            # Update US image given current EE pose (world) and human pose
            US_slicer.slice_US(world_to_human_pos, world_to_human_rot, left_us_pos_w, left_us_quat_w)
            if sim_cfg["vis_us"]:
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg["vis_seg_map"]:
                US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, left_us_pos_w, left_us_quat_w)

            # ---------------------------------------------------------
            # LEFT target EE pose: take it from USSlicer (human frame)
            # ---------------------------------------------------------

            # target EE pose in HUMAN frame (USSlicer output)
            human_ee_target_pos  = US_slicer.human_to_ee_target_pos.to(sim.device)   # (N,3) expected
            human_ee_target_quat = US_slicer.human_to_ee_target_quat.to(sim.device)  # (N,4) wxyz expected

            # HUMAN -> WORLD (still in US frame orientation)
            world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
                world_to_human_pos,
                world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # orientation: US frame -> robot frame (undo EE->US)
            world_ee_target_rot_us = matrix_from_quat(world_ee_target_quat_us)
            world_ee_target_rot_robot = torch.bmm(
                world_ee_target_rot_us,
                RotMat_torch.transpose(1, 2),
            )
            world_ee_target_quat = quat_from_matrix(world_ee_target_rot_robot)

            # -----------------------------------------
            # Convert WORLD target EE pose -> BASE frame
            # -----------------------------------------
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_ee_target_pos,
                world_ee_target_quat,
            )

            base_to_ee_target_pose = torch.cat(
                [base_to_ee_target_pos, base_to_ee_target_quat],
                dim=-1,
            )

            # ----------------------------
            # Command IK for LEFT arm
            # ----------------------------
            diff_ik_l.set_command(base_to_ee_target_pose)

            # Get Jacobian
            left_US_jacobian = robot.root_physx_view.get_jacobians()[
                :, left_ee_id - 1, :, left_entity_cfg.joint_ids
            ]

            # rotate jacobian to base frame
            base_RotMat = matrix_from_quat(quat_inv(base_quat_w))  # [N,3,3]
            left_US_jacobian[:, 0:3, :] = torch.bmm(base_RotMat, left_US_jacobian[:, 0:3, :])
            left_US_jacobian[:, 3:6, :] = torch.bmm(base_RotMat, left_US_jacobian[:, 3:6, :])

            # current joint positions
            left_US_joint_pos = robot.data.joint_pos[:, left_entity_cfg.joint_ids]

            # compute desired joint positions with IK
            left_joint_pos_des = diff_ik_l.compute(
                left_ee_pos_b,
                left_ee_quat_b,
                left_US_jacobian,
                left_US_joint_pos,
            )
            
            # Current right wrist pose in BASE
            right_ee_w = robot.data.body_state_w[:, right_ee_id, 0:7]
            right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                right_ee_w[:, 0:3],
                right_ee_w[:, 3:7],
            )

            # point on trajectory: human_to_traj_pos + tip_along * traj_drct
            tip_along = torch.full((scene.num_envs, 1), TIP_ALONG_TRAJ, device=sim.device, dtype=torch.float32)

            # optional safety clamp within segment (prevents silly requests)
            tip_along = torch.clamp(
                tip_along,
                -vertebra_viewer.traj_half_length.unsqueeze(-1) + 1e-4,
                +vertebra_viewer.traj_half_length.unsqueeze(-1) - 1e-4,
            )

            tip_tgt_pos_h = vertebra_viewer.human_to_traj_pos + tip_along * vertebra_viewer.traj_drct  # (N,3)

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

            # HUMAN -> WORLD
            tip_tgt_pos_w, tip_tgt_quat_w = combine_frame_transforms(
                world_to_human_pos, world_to_human_rot,
                tip_tgt_pos_h, tip_tgt_quat_h
            )

            # WORLD -> BASE  (this overwrites the old random tip_tgt_*_b)
            tip_tgt_pos_b, tip_tgt_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w,
                tip_tgt_pos_w, tip_tgt_quat_w
            )

            # Convert TIP target -> desired wrist pose in BASE using fixed offset
            wrist_tgt_pos_b, wrist_tgt_quat_b = combine_frame_transforms(
                tip_tgt_pos_b,
                tip_tgt_quat_b,
                tip_to_wrist_pos,
                tip_to_wrist_quat,
            )

            # Set IK command for the right wrist
            wrist_tgt_pos_b[:, 2] += 0.1
            wrist_tgt_pos_b[:, 1] += 0.0
            right_cmd_pose = torch.cat([wrist_tgt_pos_b, wrist_tgt_quat_b], dim=-1)
            diff_ik_r.set_command(right_cmd_pose)

            # Right Jacobian (rotate to base frame like left)
            right_jacobian = robot.root_physx_view.get_jacobians()[
                :, right_ee_id - 1, :, right_entity_cfg.joint_ids
            ]
            right_jacobian[:, 0:3, :] = torch.bmm(base_RotMat, right_jacobian[:, 0:3, :])
            right_jacobian[:, 3:6, :] = torch.bmm(base_RotMat, right_jacobian[:, 3:6, :])

            # Current right joint positions
            right_joint_pos = robot.data.joint_pos[:, right_entity_cfg.joint_ids]

            # Solve IK -> desired right joint positions
            right_joint_pos_des = diff_ik_r.compute(
                right_ee_pos_b,
                right_ee_quat_b,
                right_jacobian,
                right_joint_pos,
            )

            # Single PD target application for both arms
            # Start from current joint positions for the full set, then overwrite the controlled subsets
            full_joint_pos_des = robot.data.joint_pos[:, full_joint_ids]
            full_joint_pos_des[:, left_full_cols] = left_joint_pos_des
            full_joint_pos_des[:, right_full_cols] = right_joint_pos_des
            
            robot.set_joint_position_target(
                full_joint_pos_des,
                joint_ids=full_joint_ids,
            )

            if step_i % 10 == 0:
                # TIP target pose in WORLD (for visualization)
                tip_tgt_pos_w_vis, tip_tgt_quat_w_vis = combine_frame_transforms(
                    base_pos_w, base_quat_w, tip_tgt_pos_b, tip_tgt_quat_b
                )
                marker_indices = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
                frame_vis.visualize(left_ee_w[:, 0:3], left_ee_w[:, 3:7], marker_indices=marker_indices)
                frame_vis.visualize(right_ee_w[:, 0:3], right_ee_w[:, 3:7], marker_indices=marker_indices)
                frame_vis.visualize(tip_tgt_pos_w_vis, tip_tgt_quat_w_vis, marker_indices=marker_indices)
                right_tip_pos_w, right_tip_quat_w = combine_frame_transforms(
                    right_ee_w[:, 0:3], right_ee_w[:, 3:7],
                    wrist_to_tip_pos, wrist_to_tip_quat
                )
                frame_vis.visualize(right_tip_pos_w, right_tip_quat_w, marker_indices=marker_indices)

            # ----------------------------
            # Debug: tracking errors (env 0)
            # ----------------------------
            # Position errors (meters)
            if PLOT_ENABLE:
                # --- position error vectors in BASE (env 0) ---
                left_pos_err_vec  = (left_ee_pos_b - base_to_ee_target_pos)        # (N,3)
                right_pos_err_vec = (right_ee_pos_b - wrist_tgt_pos_b)             # (N,3)

                # --- orientation error: quaternion -> euler xyz in deg (env 0) ---
                # quat_angle_error_deg ti dà uno scalare: qui invece vogliamo le 3 componenti
                # Calcolo: q_err = q_target * inv(q_current), poi euler(q_err)
                ql_err = quat_mul(base_to_ee_target_quat, quat_inv(left_ee_quat_b))     # (N,4) wxyz
                qr_err = quat_mul(wrist_tgt_quat_b,       quat_inv(right_ee_quat_b))    # (N,4) wxyz

                l_roll, l_pitch, l_yaw = euler_xyz_from_quat(ql_err)
                r_roll, r_pitch, r_yaw = euler_xyz_from_quat(qr_err)

                left_ang_err_vec_deg  = torch.stack([torch.rad2deg(l_roll),  torch.rad2deg(l_pitch),  torch.rad2deg(l_yaw)],  dim=-1)  # (N,3)
                right_ang_err_vec_deg = torch.stack([torch.rad2deg(r_roll), torch.rad2deg(r_pitch), torch.rad2deg(r_yaw)], dim=-1)     # (N,3)

                # Log env 0
                t_hist.append(sim_time_acc)

                l_pos_err_hist.append(left_pos_err_vec[0].detach().cpu().numpy())       # (3,)
                r_pos_err_hist.append(right_pos_err_vec[0].detach().cpu().numpy())      # (3,)

                l_ang_err_hist.append(left_ang_err_vec_deg[0].detach().cpu().numpy())   # (3,)
                r_ang_err_hist.append(right_ang_err_vec_deg[0].detach().cpu().numpy())  # (3,)

                # Update live plot
                if (step_i % PLOT_EVERY) == 0 and len(t_hist) > 2:
                    ax00, ax01 = axes[0, 0], axes[0, 1]
                    ax10, ax11 = axes[1, 0], axes[1, 1]

                    ax00.cla(); ax01.cla(); ax10.cla(); ax11.cla()

                    # Left pos xyz
                    lpos = np.stack(l_pos_err_hist, axis=0)  # (T,3)
                    ax00.plot(t_hist, lpos[:, 0], label="ex")
                    ax00.plot(t_hist, lpos[:, 1], label="ey")
                    ax00.plot(t_hist, lpos[:, 2], label="ez")
                    ax00.set_title("Left EE position error components [m] (BASE)")
                    ax00.set_xlabel("time [s]")
                    ax00.set_ylabel("m")
                    ax00.legend()
                    ax00.grid(True, which="both")
                    ax00.minorticks_on()

                    # Left ang rpy
                    lang = np.stack(l_ang_err_hist, axis=0)  # (T,3)
                    ax01.plot(t_hist, lang[:, 0], label="roll")
                    ax01.plot(t_hist, lang[:, 1], label="pitch")
                    ax01.plot(t_hist, lang[:, 2], label="yaw")
                    ax01.set_title("Left EE orientation error components [deg]")
                    ax01.set_xlabel("time [s]")
                    ax01.set_ylabel("deg")
                    ax01.legend()
                    ax01.grid(True, which="both")
                    ax01.minorticks_on()

                    # Right pos xyz
                    rpos = np.stack(r_pos_err_hist, axis=0)  # (T,3)
                    ax10.plot(t_hist, rpos[:, 0], label="ex")
                    ax10.plot(t_hist, rpos[:, 1], label="ey")
                    ax10.plot(t_hist, rpos[:, 2], label="ez")
                    ax10.set_title("Right EE position error components [m] (BASE)")
                    ax10.set_xlabel("time [s]")
                    ax10.set_ylabel("m")
                    ax10.legend()
                    ax10.grid(True, which="both")
                    ax10.minorticks_on()

                    # Right ang rpy
                    rang = np.stack(r_ang_err_hist, axis=0)  # (T,3)
                    ax11.plot(t_hist, rang[:, 0], label="roll")
                    ax11.plot(t_hist, rang[:, 1], label="pitch")
                    ax11.plot(t_hist, rang[:, 2], label="yaw")
                    ax11.set_title("Right EE orientation error components [deg]")
                    ax11.set_xlabel("time [s]")
                    ax11.set_ylabel("deg")
                    ax11.legend()
                    ax11.grid(True, which="both")
                    ax11.minorticks_on()

                    fig.tight_layout()
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        print_wr(step_i)
        # ----------------------------
        # SCENE UPDATE (mandatory)
        # ----------------------------
        scene.write_data_to_sim()
        sim.step()
        step_i += 1
        scene.update(sim_dt)
        sim_time_acc += sim_dt

# Main
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