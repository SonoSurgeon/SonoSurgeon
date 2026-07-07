# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import sys
import copy
from collections.abc import Sequence

import gymnasium as gym
import nibabel as nib
import numpy as np
import torch
import wandb
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    subtract_frame_transforms,
)

from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *

# Pink / Pinocchio
import pinocchio as pin
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pink.solve_ik import solve_ik
import logging


class JointLimitFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "is out of limits" in msg and "Value" in msg:
            return False
        return True


logging.getLogger().addFilter(JointLimitFilter())
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTDiscreteMotionGenerator,
    GTMotionGenerator,
)
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer


# =============================================================================
# YAML configuration
# =============================================================================

raw_scene_cfg = YAML().load(
    open(f"{PACKAGE_DIR}/tasks/robot_US_guidance_Pink/cfgs/robotic_US_guidance_Pink.yaml", "r")
)

active_cfg = raw_scene_cfg["active_cfg"]
scene_cfg = raw_scene_cfg["configs"][active_cfg]

scene_cfg["if_record_traj"] = raw_scene_cfg.get("if_record_traj", False)
scene_cfg["record_path"] = raw_scene_cfg.get("record_path", None)

# TODO: fix observation scale
if scene_cfg["sim"]["us"] == "net":
    scene_cfg["observation"]["scale"] = scene_cfg["observation"]["scale_net"]

robot_cfg = scene_cfg["robot"]

# robot type/side from YAML (supports G1 / H1)
robot_type = str(robot_cfg.get("type", "g1")).lower()     # 'g1' or 'h1'
robot_side = str(robot_cfg.get("side", "left")).lower()   # 'left' or 'right'

if robot_type not in ["g1", "h1"]:
    raise ValueError(f"robot.type must be 'g1' or 'h1', got: {robot_type!r}")

# select base articulation config depending on robot type
if robot_type == "g1":
    if active_cfg == "surgery_target":
        base_robot_cfg: ArticulationCfg = G1_TOOLS_SURGERY_CFG
    else:
        base_robot_cfg: ArticulationCfg = G1_TOOLS_SURGERY_CFG2 # G1_CFG_TOOLS_BASEFIX  # G1 config without hands
else:
    base_robot_cfg: ArticulationCfg = H12_TOOLS_SURGERY_CFG3 #H12_CFG_TOOLS_BASEFIX  # H1-2 config without hands

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

# =============================================================================
# Patient / bed / datasets
# =============================================================================

# patient
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(
        float(patient_cfg["pos"][0]),
        float(patient_cfg["pos"][1]),
        float(patient_cfg["pos"][2]),
    ),
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

# dataset paths
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

CAMERA_EYE = [0.596, 0.543, 1.682]
CAMERA_TARGET = (0.088, -0.227, 1.296)

# =============================================================================
# Env config
# =============================================================================

@configclass
class roboticUSEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = scene_cfg["sim"]["episode_length"]
    action_scale = 1
    action_space = 3
    observation_space = [1, 150, 200]
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


# =============================================================================
# Env
# =============================================================================

class roboticUSEnv(DirectRLEnv):
    cfg: roboticUSEnvCfg

    def __init__(self, cfg: roboticUSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.sim.has_gui():
            self.sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

        self.robot_type = robot_type
        self.EE_LINK_SIDE = robot_side

        if "left" in self.EE_LINK_SIDE:
            self.EE_LINK_NAME = "left_wrist_yaw_link"
        elif "right" in self.EE_LINK_SIDE:
            self.EE_LINK_NAME = "right_wrist_yaw_link"
        else:
            raise ValueError("robot.side must contain 'left' or 'right'")

        self.joint_pattern = rf"(waist_(roll|pitch|yaw)_joint|{self.EE_LINK_SIDE}_(shoulder|elbow|wrist)_.*)"
        self.robot_entity_cfg = SceneEntityCfg(
            "robot_US",
            joint_names=[self.joint_pattern],
            body_names=[self.EE_LINK_NAME],
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_entity_cfg.body_ids[-1]

        # Pink / Pinocchio controller setup.
        # Keep the environment identical to the DIK guidance task; only the EE controller differs.
        if self.robot_type == "g1":
            self._pink_urdf_path = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
        else:
            self._pink_urdf_path = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_2_handless.urdf"

        if not os.path.exists(self._pink_urdf_path):
            raise FileNotFoundError(f"URDF not found for Pink: {self._pink_urdf_path}")

        self.pin_model = pin.buildModelFromUrdf(self._pink_urdf_path)
        self.pin_datas = [self.pin_model.createData() for _ in range(self.scene.num_envs)]
        self.name_to_qidx = self._build_name_to_qidx(self.pin_model)
        self.isaac_joint_names = list(self.robot.data.joint_names)

        self._pink_cfgs: list[Configuration] = []
        self._pink_tasks: list[FrameTask] = []
        for env_i in range(self.scene.num_envs):
            q0 = self._isaac_to_pin_q(self.robot.data.joint_pos[env_i])
            self._pink_cfgs.append(Configuration(self.pin_model, self.pin_datas[env_i], q0))
            self._pink_tasks.append(
                FrameTask(frame=self.EE_LINK_NAME, position_cost=1.0, orientation_cost=1.0)
            )

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

        # load CT maps
        ct_map_list = []
        for ct_map_file in ct_map_file_list:
            ct_map = nib.load(ct_map_file).get_fdata()
            ct_min_max = scene_cfg["sim"]["ct_range"]
            ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
            ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
            ct_map_list.append(ct_map)

        # label conversion map
        label_convert_map = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
        )

        # US cfg
        us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
        us_generative_cfg = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
        )
        self.sim_cfg = scene_cfg["sim"]

        self.init_cmd_pose_min = (
            torch.tensor(self.sim_cfg["patient_xz_init_range"][0], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.init_cmd_pose_max = (
            torch.tensor(self.sim_cfg["patient_xz_init_range"][1], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )

        if scene_cfg["observation"]["3D"]:
            img_thickness = us_cfg["image_3D_thickness"]
        else:
            img_thickness = 1

        roll_adj = scene_cfg["motion_planning"]["US_roll_adj"]

        # construct US simulator
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
            roll_adj=roll_adj,
            visualize=self.sim_cfg["vis_seg_map"],
            sim_mode=scene_cfg["sim"]["us"],
            us_generative_cfg=us_generative_cfg,
            height=ROBOT_HEIGHT,
            height_img=ROBOT_HEIGHT_IMG,
        )
        
        self.US_slicer.current_x_z_x_angle_cmd = (self.init_cmd_pose_min + self.init_cmd_pose_max) / 2

        self.human_world_poses = self.human.data.root_state_w
        self.human_init_root_state = self.human.data.root_state_w.clone()

        # construct ground truth motion generator
        motion_plan_cfg = scene_cfg["motion_planning"]
        self.max_action = torch.tensor(
            scene_cfg["action"]["max_action"], device=self.sim.device
        ).reshape((1, -1))

        self.goal_cmd_pose = (
            torch.tensor(motion_plan_cfg["patient_xz_goal"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.use_vertebra_goal = motion_plan_cfg["use_vertebra_goal"]
        self.gt_motion_generator = GTDiscreteMotionGenerator(
            goal_cmd_pose=self.goal_cmd_pose,
            scale=torch.tensor(motion_plan_cfg["scale"], device=self.sim.device),
            num_envs=self.scene.num_envs,
            surface_map_list=self.US_slicer.surface_map_list,
            surface_normal_list=self.US_slicer.surface_normal_list,
            label_res=label_res,
            US_height=self.US_slicer.height,
        )

        self.vertebra_viewer = VertebraViewer(
            self.scene.num_envs,
            len(human_usd_list),
            target_stl_file_list,
            target_traj_file_list,
            False,
            label_res,
            self.sim.device,
        )

        # observation space to image
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.cfg.observation_space[0], self.cfg.observation_space[1], self.cfg.observation_space[2]),
            dtype=np.uint8,
        )

        self.cfg.observation_space[0] = self.US_slicer.img_thickness

        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.cfg.observation_space[0], self.cfg.observation_space[1], self.cfg.observation_space[2]),
            dtype=np.float32,
        )

        self.termination_direct = True
        self.observation_mode = scene_cfg["observation"]["mode"]
        self.action_mode = scene_cfg["action"]["mode"]
        self.action_scale = (
            torch.tensor(scene_cfg["action"]["scale"], device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )

        self.w_pos = scene_cfg["reward"]["w_pos"]

        # success termination params (from YAML)
        self.success_circle = float(scene_cfg["reward"]["success_circle"])
        self.success_bonus = float(scene_cfg["reward"]["success_bonus"])
        self.success_hold_s = float(scene_cfg["reward"]["success_hold_s"])

        # env step time (action step)
        self._dt_env = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        self.success_hold_steps = max(1, int(round(self.success_hold_s / self._dt_env)))

        # per-env counters/flags
        self.success_count = torch.zeros(self.scene.num_envs, device=self.sim.device, dtype=torch.int32)
        self.success_reached = torch.zeros(self.scene.num_envs, device=self.sim.device, dtype=torch.bool)
        self.success_bonus_given = torch.zeros(self.scene.num_envs, device=self.sim.device, dtype=torch.bool)

        # total success counters (easy logging)
        self.success_total = 0

        self.single_action_space = gym.spaces.Box(
            low=-(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            high=(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            shape=(self.cfg.action_space,),
            dtype=np.float32,
        )

        wandb.init()
        self.num_step = 0

        # --- run mode ---
        entry = os.path.basename(sys.argv[0]).lower()
        if "play" in entry:
            self._run_mode = "play"
        elif "train" in entry:
            self._run_mode = "train"
        else:
            self._run_mode = "train"

        # --- single switch: record only if YAML says so AND we are in play ---
        self._record_traj = bool(scene_cfg.get("if_record_traj", False)) and (self._run_mode == "play")

        if self._record_traj:
            self._max_T = int(self.max_episode_length)
            self._N = int(self.scene.num_envs)

            # Per-step buffers (current episode rollout)
            self._buf_cmd_pose = torch.zeros((self._N, self._max_T, 3), device=self.sim.device)
            self._buf_dist_to_goal = torch.zeros((self._N, self._max_T), device=self.sim.device)
            self._buf_success_mask = torch.zeros((self._N, self._max_T), device=self.sim.device, dtype=torch.uint8)

            # Exported tensors = last completed episode per env (zero-padded)
            self.last_cmd_pose_traj = torch.zeros_like(self._buf_cmd_pose)
            self.last_dist_to_goal_traj = torch.zeros_like(self._buf_dist_to_goal)
            self.last_success_mask_traj = torch.zeros_like(self._buf_success_mask)

            # IK error trajs (only if recording)
            self.ik_err6d_trajs = []

            self.manipulability_trajs = []  # list of (N_envs,) tensors, appended every step

            """
            debug
            obs_shape = (
                self.US_slicer.img_thickness,
                int(us_cfg["image_size"][1]),
                int(us_cfg["image_size"][0]),
            )

            self._buf_policy_obs_env0 = torch.zeros(
                (self._max_T, *obs_shape), device=self.sim.device, dtype=torch.float32
            )
            self._buf_policy_raw_action_env0 = torch.zeros(
                (self._max_T, self.cfg.action_space), device=self.sim.device, dtype=torch.float32
            )
            self._buf_policy_final_action_env0 = torch.zeros(
                (self._max_T, self.cfg.action_space), device=self.sim.device, dtype=torch.float32
            )

            self.last_policy_obs_env0 = torch.zeros_like(self._buf_policy_obs_env0)
            self.last_policy_raw_action_env0 = torch.zeros_like(self._buf_policy_raw_action_env0)
            self.last_policy_final_action_env0 = torch.zeros_like(self._buf_policy_final_action_env0)
            """ 


    def get_US_target_pose(self):
        
        # Compute vertebra-based target exactly once from the current anatomy state
        vertebra_to_US_2d_pos = torch.tensor(scene_cfg["motion_planning"]["vertebra_to_US_2d_pos"]).to(self.sim.device)
        vertebra_2d_pos = self.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
        US_target_2d_pos = vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0)
        US_target_2d_angle = scene_cfg["motion_planning"]["US_target_2d_angle"] * torch.ones_like(vertebra_2d_pos[:, 0:1])
        US_target_2d = torch.cat([US_target_2d_pos, US_target_2d_angle], dim=-1)
        self.goal_cmd_pose = US_target_2d

    def _setup_scene(self):
        """Configuration for the robotic US guidance scene."""

        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # lights
        dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        dome_light_cfg.func("/World/Light", dome_light_cfg)

        # robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # bed
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

        # clone envs and register
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human

    def _get_observations(self) -> dict:
        # get human frame
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )

        # EE pose in WORLD, robot frame (raw from Isaac)
        ee_pose_w_robot = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[-1], 0:7]
        ee_pos_w = ee_pose_w_robot[:, 0:3]
        ee_quat_w = ee_pose_w_robot[:, 3:7]

        # rotate orientation robot -> US frame
        ee_rotmat_w = matrix_from_quat(ee_quat_w)
        us_rotmat_w = torch.bmm(ee_rotmat_w, self.RotMat)
        us_quat_w = quat_from_matrix(us_rotmat_w)

        # pose in US frame (for slicer)
        self.US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)

        self.num_step += 1

        if self._record_traj and hasattr(self, "base_to_ee_target_pose"):
            world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
            ee_pose_w_robot = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[-1], 0:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7],
                ee_pose_w_robot[:, 0:3], ee_pose_w_robot[:, 3:7],
            )

            pos_err = self.base_to_ee_target_pose[:, 0:3] - ee_pos_b
            R_t = matrix_from_quat(self.base_to_ee_target_pose[:, 3:7])
            R_c = matrix_from_quat(ee_quat_b)
            R_e = torch.bmm(R_t, R_c.transpose(1, 2))
            tr = R_e[:, 0, 0] + R_e[:, 1, 1] + R_e[:, 2, 2]
            ang = torch.acos(torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0))
            axis = torch.stack([R_e[:, 2, 1] - R_e[:, 1, 2], R_e[:, 0, 2] - R_e[:, 2, 0], R_e[:, 1, 0] - R_e[:, 0, 1]], dim=-1)
            axis = axis / (2.0 * torch.sin(ang).clamp_min(1e-6)).unsqueeze(-1)
            rot_err = axis * ang.unsqueeze(-1)

            self.ik_err6d_trajs.append(torch.cat([pos_err, rot_err], dim=-1).clone())

        if self.observation_mode == "US":
            self.US_slicer.slice_US(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            US_img = self.US_slicer.us_img_tensor.permute(0, 3, 1, 2) * self.cfg.observation_scale
            observations = {"policy": US_img}
        elif self.observation_mode == "CT":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            CT_img = self.US_slicer.ct_img_tensor.permute(0, 3, 1, 2) * self.cfg.observation_scale
            observations = {"policy": CT_img}
        elif self.observation_mode == "seg":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            label_img = self.US_slicer.label_img_tensor.permute(0, 3, 1, 2) * self.cfg.observation_scale
            observations = {"policy": label_img}
            
        else:
            raise ValueError("Invalid observation mode")

        if self.sim_cfg["vis_us"] and self.num_step % self.sim_cfg["vis_int"] == 0:
            self.US_slicer.visualize(self.observation_mode)

        # if self._record_traj:
        #    t0 = int(torch.clamp(self.episode_length_buf[0], 0, self._max_T - 1).item())
        #    self._buf_policy_obs_env0[t0] = observations["policy"][0].detach()

        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # if action is 6 dim, convert to 3 dim (xz pos + y rot)
        if actions.shape[-1] == 6:
            actions = actions[:, [0, 2, 5]]

        # raw_actions_for_log = actions.clone()

        if self.action_mode == "continuous":
            actions = torch.clamp(actions * self.action_scale, -self.max_action, self.max_action)
        elif self.action_mode == "discrete":
            actions = torch.sign(actions) * self.action_scale
        else:
            raise ValueError("Invalid action mode")
        self.actions = actions

        actions *= scene_cfg["action"]["w_action"]

        # if self._record_traj:
        #     t0 = int(torch.clamp(self.episode_length_buf[0], 0, self._max_T - 1).item())
        #     self._buf_policy_raw_action_env0[t0] = raw_actions_for_log[0].detach()
        #     self._buf_policy_final_action_env0[t0] = actions[0].detach()
            
        # action: dx, dz in image frame -> human frame
        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        human_to_ee_rot_mat = matrix_from_quat(human_to_ee_quat)

        dx_dz_human = (
            actions[:, 0].unsqueeze(1) * human_to_ee_rot_mat[:, :, 0]
            + actions[:, 1].unsqueeze(1) * human_to_ee_rot_mat[:, :, 1]
        )
        cmd = torch.cat([dx_dz_human[:, [0, 2]], actions[:, 2:3]], dim=-1)
        self.US_slicer.update_cmd(cmd)

        # compute desired WORLD EE pose from cmd (US frame)
        world_to_ee_target_pos, world_to_ee_target_rot_us = self.US_slicer.compute_world_ee_pose_from_cmd(
            self.world_to_human_pos, self.world_to_human_rot
        )

        # rotate target orientation from US frame back to robot frame
        world_to_ee_target_rotmat_us = matrix_from_quat(world_to_ee_target_rot_us)
        world_to_ee_target_rotmat_robot = torch.bmm(
            world_to_ee_target_rotmat_us,
            self.RotMat.transpose(1, 2),
        )
        world_to_ee_target_rot = quat_from_matrix(world_to_ee_target_rotmat_robot)

        # compute BASE->EE target for differential IK
        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            world_to_ee_target_rot,
        )
        base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

        self.base_to_ee_target_pose = base_to_ee_target_pose

        # record extras
        self.extras["human_to_ee_pos"] = human_to_ee_pos
        self.extras["human_to_ee_quat"] = human_to_ee_quat


    # -------------------------------------------------------------------------
    # Pink / Pinocchio helper methods
    # -------------------------------------------------------------------------
    def _build_name_to_qidx(self, model: pin.Model) -> dict[str, int]:
        """Map Pinocchio 1-DoF joint names to their q index."""
        name_to_qidx: dict[str, int] = {}
        for j_id, joint in enumerate(model.joints):
            if joint.nq == 1:
                name_to_qidx[model.names[j_id]] = joint.idx_q
        return name_to_qidx

    def _isaac_to_pin_q(self, q_isaac: torch.Tensor) -> np.ndarray:
        """Convert one Isaac joint-position vector to a Pinocchio q vector."""
        q_pin = np.zeros(self.pin_model.nq, dtype=np.float64)
        q_np = q_isaac.detach().cpu().numpy()

        for i, jname in enumerate(self.isaac_joint_names):
            qidx = self.name_to_qidx.get(jname, None)
            if qidx is not None:
                q_pin[qidx] = float(q_np[i])

        return q_pin

    def _pin_to_isaac_q(self, q_pin: np.ndarray) -> np.ndarray:
        """Convert one Pinocchio q vector to Isaac joint-name order."""
        q_isaac = np.zeros(len(self.isaac_joint_names), dtype=np.float64)

        for i, jname in enumerate(self.isaac_joint_names):
            qidx = self.name_to_qidx.get(jname, None)
            if qidx is not None:
                q_isaac[i] = float(q_pin[qidx])

        return q_isaac

    def _reset_pink_configs(self) -> None:
        """Synchronize Pink configurations with the current Isaac joint state."""
        q_all = self.robot.data.joint_pos
        for env_i in range(self.scene.num_envs):
            q_pin = self._isaac_to_pin_q(q_all[env_i])
            self._pink_cfgs[env_i].update(q_pin)

    def _compute_pink_target_from_world_us(
        self,
        world_to_ee_target_pos: torch.Tensor,
        world_to_ee_target_quat_us: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a WORLD target in US-frame convention to BASE target in robot EE convention."""
        world_to_ee_target_rotmat_us = matrix_from_quat(world_to_ee_target_quat_us)
        world_to_ee_target_rotmat_robot = torch.bmm(
            world_to_ee_target_rotmat_us,
            self.RotMat.transpose(1, 2),
        )
        world_to_ee_target_quat_robot = quat_from_matrix(world_to_ee_target_rotmat_robot)

        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            world_to_ee_target_quat_robot,
        )

        return torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

    def _apply_pink_target_base(self, base_to_ee_target_pose: torch.Tensor) -> None:
        """Solve Pink IK for the US chain and send Isaac joint-position targets."""
        target_pos_np = (
            base_to_ee_target_pose[:, 0:3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        target_rot_np = (
            matrix_from_quat(base_to_ee_target_pose[:, 3:7])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

        q_all = self.robot.data.joint_pos
        q_next_ctrl_np = np.zeros(
            (self.scene.num_envs, len(self.robot_entity_cfg.joint_ids)),
            dtype=np.float32,
        )

        dt = float(self.physics_dt)

        for env_i in range(self.scene.num_envs):
            q_pin = self._isaac_to_pin_q(q_all[env_i])
            self._pink_cfgs[env_i].update(q_pin)

            self._pink_tasks[env_i].set_target(
                pin.SE3(target_rot_np[env_i], target_pos_np[env_i])
            )

            vel = solve_ik(
                self._pink_cfgs[env_i],
                tasks=[self._pink_tasks[env_i]],
                dt=dt,
                solver="quadprog",
                damping=scene_cfg["action"]["damping"],
                safety_break=False,
            )
            self._pink_cfgs[env_i].integrate_inplace(vel, dt)

            q_next_isaac = self._pin_to_isaac_q(self._pink_cfgs[env_i].q)
            q_next_ctrl_np[env_i, :] = q_next_isaac[self.robot_entity_cfg.joint_ids].astype(
                np.float32,
                copy=False,
            )

        q_next_ctrl_t = torch.from_numpy(q_next_ctrl_np).to(
            device=self.sim.device,
            dtype=self.robot.data.joint_pos.dtype,
        )

        # Keep targets inside Isaac joint limits.
        joint_ids_t = torch.as_tensor(
            self.robot_entity_cfg.joint_ids,
            device=self.sim.device,
            dtype=torch.long,
        )
        joint_limits = self.robot.data.joint_pos_limits
        jmin = joint_limits[0, joint_ids_t, 0]
        jmax = joint_limits[0, joint_ids_t, 1]
        margin = 1e-3
        q_next_ctrl_t = torch.clamp(
            q_next_ctrl_t,
            jmin.unsqueeze(0) + margin,
            jmax.unsqueeze(0) - margin,
        )

        self.robot.set_joint_position_target(q_next_ctrl_t, joint_ids=joint_ids_t)

    def _apply_action(self):
        if not hasattr(self, "base_to_ee_target_pose"):
            return

        self._apply_pink_target_base(self.base_to_ee_target_pose)

        if self._record_traj:
            world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
            US_jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
            ]
            base_rotmat = matrix_from_quat(quat_inv(world_to_base_pose[:, 3:7]))
            US_jacobian[:, 0:3, :] = torch.bmm(base_rotmat, US_jacobian[:, 0:3, :])
            US_jacobian[:, 3:6, :] = torch.bmm(base_rotmat, US_jacobian[:, 3:6, :])

            JJT = torch.bmm(US_jacobian, US_jacobian.transpose(1, 2))
            eps = 1e-12
            JJT = JJT + eps * torch.eye(6, device=JJT.device, dtype=JJT.dtype).unsqueeze(0)
            det_JJT = torch.linalg.det(JJT)
            manipulability = torch.sqrt(torch.clamp(det_JJT, min=0.0))
            self.manipulability_trajs.append(manipulability.detach().cpu())


    def _get_rewards(self) -> torch.Tensor:
        # current cmd pose in human frame
        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        self.cur_cmd_pose = self.gt_motion_generator.human_cmd_state_from_ee_pose(
            cur_human_ee_pos, cur_human_ee_quat
        )

        # distance to goal (shaping)
        cur_distance_to_goal = (
            torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1) * self.w_pos
        )
        cur_distance_to_goal += torch.norm(
            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1
        )

        # potential-based shaping (delta distance)
        reward = self.distance_to_goal - cur_distance_to_goal
        self.distance_to_goal = cur_distance_to_goal

        # --- asymmetric penalty near target (smooth sigmoid gate) ---
        alpha_max = float(scene_cfg["reward"]["alpha_max"])
        dead = 0.01

        # sigmoid gate parameters
        # eps: "success-like" radius where you want the extra discouragement to kick in
        # tau: softness of the transition (larger = smoother / wider transition)
        eps = 3.5
        tau = 0.2 * eps

        # gate ~1 near target, ~0 far away
        g = torch.sigmoid((eps - cur_distance_to_goal) / tau)

        # scale in [1, alpha_max]
        scale = 1.0 + (alpha_max - 1.0) * g

        # only amplify meaningful negative steps (avoid punishing tiny jitter)
        reward = torch.where(reward < -dead, reward * scale, reward)

        # success condition
        inside = cur_distance_to_goal <= self.success_circle
        self.success_count = torch.where(
            inside,
            self.success_count + 1,
            torch.zeros_like(self.success_count),
        )

        # latch success once hold is satisfied
        new_success = (self.success_count >= self.success_hold_steps) & (~self.success_reached)
        self.success_reached |= (self.success_count >= self.success_hold_steps)

        # one-time terminal success bonus
        give_bonus = self.success_reached & (~self.success_bonus_given)
        reward = reward + give_bonus.to(reward.dtype) * self.success_bonus
        self.success_bonus_given |= give_bonus

        # count successes (global + per-episode mean)
        if new_success.any():
            n = int(new_success.sum().item())
            self.success_total += n

        self.total_reward += reward

        # extras
        self.extras["cur_cmd_pose"] = self.cur_cmd_pose
        self.extras["goal_cmd_pose"] = self.goal_cmd_pose


        # per-step logging into fixed buffers (commit only on episode end)
        if self._record_traj:
            t = torch.clamp(self.episode_length_buf, 0, self._max_T - 1)
            idx = torch.arange(self._N, device=self.sim.device)

            self._buf_success_mask[idx, t] = self.success_reached.to(torch.uint8)  # latched => 1 until end ep
            self._buf_cmd_pose[idx, t, :] = self.cur_cmd_pose
            self._buf_dist_to_goal[idx, t] = cur_distance_to_goal

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.termination_direct:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros_like(self.episode_length_buf)

        out_of_bounds = torch.zeros_like(self.US_slicer.no_collide)

        # --- SUCCESS TERMINATION POLICY ---
        # 1) In play-record: mai terminare per success (vuoi registrare episodio completo)
        # 2) In train: NON terminare per success (evita episodi corti e instability)
        # 3) In other play/eval (se mai): puoi decidere di terminare
        if self._record_traj:
            success_done = torch.zeros_like(self.success_reached, dtype=torch.bool)
        elif self._run_mode == "train":
            success_done = torch.zeros_like(self.success_reached, dtype=torch.bool)
        else:
            success_done = self.success_reached.clone()

        terminated = out_of_bounds | success_done
        return terminated, time_out

    def _move_towards_target(
        self,
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 300,
    ):
        """Move the EE towards a target pose expressed in the human frame using Pink."""
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(num_steps):
            self._sim_step_counter += 1

            self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
            self.world_to_human_pos, self.world_to_human_rot = (
                self.human_world_poses[:, 0:3],
                self.human_world_poses[:, 3:7],
            )

            world_to_ee_target_pos, world_to_ee_target_quat_us = combine_frame_transforms(
                self.world_to_human_pos,
                self.world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            base_to_ee_target_pose = self._compute_pink_target_from_world_us(
                world_to_ee_target_pos,
                world_to_ee_target_quat_us,
            )
            self.base_to_ee_target_pose = base_to_ee_target_pose
            self._apply_pink_target_base(base_to_ee_target_pose)

            self.scene.write_data_to_sim()
            self.sim.step(render=False)

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

                    
    def _reset_idx(self, env_ids):

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if self._record_traj:
            _env_ids_t = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

            # export full padded episode buffers
            self.last_cmd_pose_traj[_env_ids_t] = self._buf_cmd_pose[_env_ids_t]
            self.last_dist_to_goal_traj[_env_ids_t] = self._buf_dist_to_goal[_env_ids_t]
            self.last_success_mask_traj[_env_ids_t] = self._buf_success_mask[_env_ids_t]

            # export debug policy traces for env 0
            #if 0 in set(torch.as_tensor(env_ids).cpu().tolist()):
            #    self.last_policy_obs_env0.copy_(self._buf_policy_obs_env0)
            #    self.last_policy_raw_action_env0.copy_(self._buf_policy_raw_action_env0)
            #    self.last_policy_final_action_env0.copy_(self._buf_policy_final_action_env0)

            # save to disk at every reset (play mode, homogeneous episode length)
            self._save_traj_logs()

            # reset buffers for next episode
            self._buf_cmd_pose[_env_ids_t].zero_()
            self._buf_dist_to_goal[_env_ids_t].zero_()
            self._buf_success_mask[_env_ids_t].zero_()
            
            #if 0 in set(torch.as_tensor(env_ids).cpu().tolist()):
            #    self._buf_policy_obs_env0.zero_()
            #    self._buf_policy_raw_action_env0.zero_()
            #    self._buf_policy_final_action_env0.zero_()



        super()._reset_idx(env_ids)

        # reset robot
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()

        right_elbow_joint_id = self.robot.find_joints("right_elbow_joint")[0][0]
        self.robot.set_joint_position_target(
            torch.full((self.scene.num_envs, 1), 1.0, device=self.sim.device),
            joint_ids=[right_elbow_joint_id],
        )


        # reset human
        self.human.write_root_state_to_sim(self.human_init_root_state)
        self.human.reset()

        self._reset_pink_configs()

        # initial EE pose
        self.US_root_pose_w = self.robot.data.root_state_w[:, 0:7]
        self.US_ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[-1], 0:7]

        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            self.US_root_pose_w[:, 0:3],
            self.US_root_pose_w[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )

        # update human frame
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )

        # random initial cmd target
        cmd_target_poses = torch.rand((self.scene.num_envs, 3), device=self.sim.device)
        min_init = self.init_cmd_pose_min
        max_init = self.init_cmd_pose_max
        cmd_target_poses = cmd_target_poses * (max_init - min_init) + min_init

        self.US_slicer.update_cmd(cmd_target_poses - self.US_slicer.current_x_z_x_angle_cmd)
        _world_to_ee_init_pos, _world_to_ee_init_rot = self.US_slicer.compute_world_ee_pose_from_cmd(
            self.world_to_human_pos, self.world_to_human_rot
        )

        # move arm towards initial target
        self._move_towards_target(
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        # update EE pose in US frame after motion
        self.US_ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[-1], 0:7]
        ee_pos_w = self.US_ee_pose_w[:, 0:3]
        ee_quat_w = self.US_ee_pose_w[:, 3:7]
        ee_rotmat_w = matrix_from_quat(ee_quat_w)
        us_rotmat_w = torch.bmm(ee_rotmat_w, self.RotMat)
        us_quat_w = quat_from_matrix(us_rotmat_w)
        self.US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)

        # update human pose
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )

        # log previous episode stats (if available)
        if hasattr(self, "total_reward"):
            wandb.log({"D0": self.D0.mean().item()})
            wandb.log({"DT": self.distance_to_goal.mean().item()})
            wandb.log({"D0_minus_DT": (self.D0 - self.distance_to_goal).mean().item()})
            wandb.log({"total_reward_check": self.total_reward.mean().item()})

            wandb.log(
                {
                    "pos err": (
                        torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1)
                        * self.US_slicer.label_res
                    ).mean().item()
                }
            )
            wandb.log(
                {
                    "pos err std": (
                        torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1)
                        * self.US_slicer.label_res
                    ).std().item()
                }
            )
            wandb.log(
                {
                    "pos err max": (
                        torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1)
                        * self.US_slicer.label_res
                    ).max().item()
                }
            )
            wandb.log(
                {
                    "pos err min": (
                        torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1)
                        * self.US_slicer.label_res
                    ).min().item()
                }
            )
            wandb.log(
                {
                    "rot err": (
                        torch.norm(self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1)
                        * 180 / torch.pi
                    ).mean().item()
                }
            )
            wandb.log(
                {
                    "rot err std": (
                        torch.norm(self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1)
                        * 180 / torch.pi
                    ).std().item()
                }
            )
            wandb.log(
                {
                    "rot err max": (
                        torch.norm(self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1)
                        * 180 / torch.pi
                    ).max().item()
                }
            )
            wandb.log(
                {
                    "rot err min": (
                        torch.norm(self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1)
                        * 180 / torch.pi
                    ).min().item()
                }
            )
            # log total successes (simple)
            wandb.log({"success_total": float(self.success_total)})

        # init distance to goal
        if self.use_vertebra_goal:
            self.get_US_target_pose()

        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        self.cur_cmd_pose = self.gt_motion_generator.human_cmd_state_from_ee_pose(
            cur_human_ee_pos, cur_human_ee_quat
        )
        self.distance_to_goal = (
            torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1) * self.w_pos
        )
        self.distance_to_goal += torch.norm(
            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1
        )

        self.D0 = self.distance_to_goal.clone()
        self.total_reward = torch.zeros(self.scene.num_envs, device=self.sim.device)
        self.success_count[env_ids] = 0
        self.success_reached[env_ids] = False
        self.success_bonus_given[env_ids] = False

        # extras
        self.extras["human_to_ee_pos"] = cur_human_ee_pos
        self.extras["human_to_ee_quat"] = cur_human_ee_quat
        self.extras["cur_cmd_pose"] = self.cur_cmd_pose
        self.extras["goal_cmd_pose"] = self.goal_cmd_pose

    def _save_traj_logs(self):
        if not getattr(self, "_record_traj", False):
            return

        rp_cfg = str(scene_cfg["record_path"]).strip()

        # se nel YAML è "/recordings" o comunque assoluto -> lo rendo relativo (e quindi scrivibile)
        if os.path.isabs(rp_cfg):
            rp_cfg = rp_cfg.lstrip("/")  # "/recordings" -> "recordings"

        record_path = os.path.join(PACKAGE_DIR, rp_cfg)
        os.makedirs(record_path, exist_ok=True)

        try:
            torch.save(self.last_cmd_pose_traj.detach().cpu(), os.path.join(record_path, "cmd_pose_trajs.pt"))
            torch.save(self.goal_cmd_pose.detach().cpu(), os.path.join(record_path, "goal_cmd_pose.pt"))
            torch.save(self.last_dist_to_goal_traj.detach().cpu(), os.path.join(record_path, "dist_to_goal.pt"))
            torch.save(self.last_success_mask_traj.detach().cpu(), os.path.join(record_path, "success_mask.pt"))
            if len(self.ik_err6d_trajs) > 0:
                torch.save(
                    torch.stack(self.ik_err6d_trajs, dim=1).detach().cpu(),
                    os.path.join(record_path, f"ik_err6d_trajs_{self._run_mode}.pt"),
                )
            if hasattr(self, "manipulability_trajs") and len(self.manipulability_trajs) > 0:
                manipulability_tensor = torch.stack(self.manipulability_trajs, dim=1)  # (N, T)
                torch.save(
                    manipulability_tensor,
                    os.path.join(record_path, "yoshikawa_manipulability.pt"),
                )
            self.manipulability_trajs = []

            """
            torch.save(
                self.last_policy_obs_env0.detach().cpu(),
                os.path.join(record_path, "policy_obs_env0.pt"),
            )
            torch.save(
                self.last_policy_raw_action_env0.detach().cpu(),
                os.path.join(record_path, "policy_raw_action_env0.pt"),
            )
            torch.save(
                self.last_policy_final_action_env0.detach().cpu(),
                os.path.join(record_path, "policy_final_action_env0.pt"),
            )
            """

        except Exception as e:
            print(f"[LOG][ERROR] saving failed: {e!r}", flush=True)
            raise