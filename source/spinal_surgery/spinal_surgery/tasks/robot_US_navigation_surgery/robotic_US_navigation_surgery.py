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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
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

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from spinal_surgery.assets.fr3_US import *
from spinal_surgery.assets.kuka_drill import *
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

scene_cfg = YAML().load(
    open(
        f"{PACKAGE_DIR}/tasks/robot_US_navigation_surgery/cfgs/robotic_US_navigation_surgery.yaml",
        "r",
    )
)

if scene_cfg["sim"]["us"] == "net":
    scene_cfg["observation"]["scale"] = scene_cfg["observation"]["scale_net"]
us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
us_generative_cfg = YAML().load(
    open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
)
robot_cfg = scene_cfg["robot"]

# robot
if scene_cfg["robot"]["type"] == "kuka":
    robot_articulation_cfg = KUKA_HIGH_PD_CFG
    INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "lbr_joint_0": robot_cfg["joint_pos"][0],
            "lbr_joint_1": robot_cfg["joint_pos"][1],
            "lbr_joint_2": robot_cfg["joint_pos"][2],
            "lbr_joint_3": robot_cfg["joint_pos"][3],  # -1.2,
            "lbr_joint_4": robot_cfg["joint_pos"][4],
            "lbr_joint_5": robot_cfg["joint_pos"][5],  # 1.5,
            "lbr_joint_6": robot_cfg["joint_pos"][6],
        },
        pos=(
            float(robot_cfg["pos"][0]),
            float(robot_cfg["pos"][1]),
            float(robot_cfg["pos"][2]),
        ),  # ((0.0, -0.75, 0.4))
    )

elif scene_cfg["robot"]["type"] == "fr3":
    robot_articulation_cfg = FR3_HIGH_PD_US_CFG
    INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": robot_cfg["joint_pos"][0],
            "fr3_joint2": robot_cfg["joint_pos"][1],
            "fr3_joint3": robot_cfg["joint_pos"][2],
            "fr3_joint4": robot_cfg["joint_pos"][3],  # -1.2,
            "fr3_joint5": robot_cfg["joint_pos"][4],
            "fr3_joint6": robot_cfg["joint_pos"][5],  # 1.5,
            "fr3_joint7": robot_cfg["joint_pos"][6],
        },
        pos=(
            float(robot_cfg["pos"][0]),
            float(robot_cfg["pos"][1]),
            float(robot_cfg["pos"][2]),
        ),  # ((0.0, -0.75, 0.4))
    )

robot_drill_cfg = scene_cfg["robot_drill"]
INIT_STATE_ROBOT_DRILL = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_drill_cfg["joint_pos"][0],
        "lbr_joint_1": robot_drill_cfg["joint_pos"][1],
        "lbr_joint_2": robot_drill_cfg["joint_pos"][2],
        "lbr_joint_3": robot_drill_cfg["joint_pos"][3],  # -1.2,
        "lbr_joint_4": robot_drill_cfg["joint_pos"][4],
        "lbr_joint_5": robot_drill_cfg["joint_pos"][5],  # 1.5,
        "lbr_joint_6": robot_drill_cfg["joint_pos"][6],
    },
    pos=(
        float(robot_drill_cfg["pos"][0]),
        float(robot_drill_cfg["pos"][1]),
        float(robot_drill_cfg["pos"][2]),
    ),  # ((0.0, -0.75, 0.4))
)
DRILL_TO_TIP_POS = np.array([0.0, 0.0, -0.135]).astype(np.float32)  # -0.135
DRILL_TO_TIP_QUAT = (
    R.from_euler("YXZ", [180, 0, 0], degrees=True).as_quat().astype(np.float32)
)
# patient
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(
        float(patient_cfg["pos"][0]),
        float(patient_cfg["pos"][1]),
        float(patient_cfg["pos"][2]),
    ),  # 0.7
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
    ),  # 0.7
    rot=(0.5, 0.5, 0.5, 0.5),
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

# Viewer camera
CAMERA_EYE = (-1.160, 0.814, 1.1185)
CAMERA_TARGET = (-0.364, 0.336, 0.747)

US_obs_style = scene_cfg["observation"]["style"]

@configclass
class roboticUSNavigationSurgeryCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = scene_cfg["sim"]["episode_length"]  # 5 # 300
    action_scale = 1
    action_space = 5
    observation_space = [1, 1, 1] # placeholder
    state_space = 0
    observation_scale = scene_cfg["observation"]["scale"]

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120, render_interval=decimation
    )

    robot_cfg: ArticulationCfg = robot_articulation_cfg.replace(
        prim_path="/World/envs/env_.*/Robot_US", init_state=INIT_STATE_ROBOT_US
    )

    robot_drill_cfg: ArticulationCfg = KUKA_HIGH_PD_DRILL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_drill", init_state=INIT_STATE_ROBOT_DRILL
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=2.0, replicate_physics=False
    )


class roboticUSNavigationSurgeryEnv(DirectRLEnv):
    cfg: roboticUSNavigationSurgeryCfg

    def __init__(
        self, cfg: roboticUSNavigationSurgeryCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        if self.sim.has_gui():
            self.sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

        if scene_cfg["robot"]["type"] == "kuka":
            self.robot_entity_cfg = SceneEntityCfg(
                "robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"]
            )
        else:
            self.robot_entity_cfg = SceneEntityCfg(
                "robot_US", joint_names=["fr3_joint.*"], body_names=["fr3_link8"]
            )
        self.robot_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_entity_cfg.body_ids[-1]

        self.robot_drill_entity_cfg = SceneEntityCfg(
            "robot_drill", joint_names=["lbr_joint_.*"], body_names=["screw_6_65"]
        )  # "screw_6_65"
        self.robot_drill_entity_cfg.resolve(self.scene)
        self.drill_ee_jacobi_idx = self.robot_drill_entity_cfg.body_ids[-1]

        # define ik controllers
        ik_params_drill = {"lambda_val": 0.01}
        ik_params = {"lambda_val": 0.1}
        pose_diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params,
        )
        self.pose_diff_ik_controller = DifferentialIKController(
            pose_diff_ik_cfg, self.scene.num_envs, device=self.sim.device
        )
        diff_ik_cfg_drill = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params=ik_params_drill,
        )
        self.diff_ik_controller_drill = DifferentialIKController(
            diff_ik_cfg_drill, self.scene.num_envs, device=self.sim.device
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

        self.motion_plan_cfg = scene_cfg["motion_planning"]
        
        if US_obs_style == "small":

            thickness_offsets = None

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

            self.us_img_thickness = img_thickness
            self.us_img_height = 200 // res
            self.us_img_width = 150 // res

            self.cfg.observation_space[0] = self.us_img_thickness
            self.cfg.observation_space[1] = self.us_img_height
            self.cfg.observation_space[2] = self.us_img_width

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
                roll_adj=self.motion_plan_cfg["US_roll_adj"],
                visualize=self.sim_cfg["vis_seg_map"],
                sim_mode=scene_cfg["sim"]["us"],
                us_generative_cfg=us_generative_cfg,
            )

        elif US_obs_style == "large":
            
            if scene_cfg["observation"]["3D"]:
                img_thickness = scene_cfg["observation"]["image_3D_thickness"]
            else:
                img_thickness = 1
                
            thickness_offsets = scene_cfg["observation"].get("image_3D_offsets", None)
            if thickness_offsets is not None and len(thickness_offsets) != img_thickness:
                raise ValueError(
                    f"image_3D_offsets length ({len(thickness_offsets)}) must match image_3D_thickness ({img_thickness})"
                )
                
            # Single full-resolution US stream shared by probe policy and surgery policy.
            self.us_img_height = int(us_cfg["image_size"][0])
            self.us_img_width = int(us_cfg["image_size"][1])
            self.us_img_thickness = int(img_thickness)

            self.cfg.observation_space[0] = self.us_img_thickness
            self.cfg.observation_space[1] = self.us_img_height
            self.cfg.observation_space[2] = self.us_img_width

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
                [self.us_img_height, self.us_img_width],
                us_cfg["resolution"],
                img_thickness=self.us_img_thickness,
                thickness_offsets=thickness_offsets,
                roll_adj=self.motion_plan_cfg["US_roll_adj"],
                visualize=self.sim_cfg["vis_seg_map"],
                sim_mode=scene_cfg["sim"]["us"],
                us_generative_cfg=us_generative_cfg,
            )

        else:
            raise ValueError(f"Unsupported observation style: {US_obs_style}, either 'small' or 'large' expected.")

        self.US_slicer.current_x_z_x_angle_cmd = (
            self.init_cmd_pose_min + self.init_cmd_pose_max
        ) / 2

        self.human_world_poses = (
            self.human.data.root_state_w
        )  # these are already the initial poses

        # construct ground truth motion generator
        self.vertebra_viewer = VertebraViewer(
            self.scene.num_envs,
            len(human_usd_list),
            target_stl_file_list,
            target_traj_file_list,
            self.sim_cfg["vis_us"],
            label_res,
            self.sim.device,
        )

        self.goal_pose_ph = self.motion_plan_cfg["patient_xz_goal"]

        self.gt_motion_generator = GTDiscreteMotionGenerator(
            goal_cmd_pose=self.goal_pose_ph,
            scale=torch.tensor(self.motion_plan_cfg["scale"], device=self.sim.device),
            num_envs=self.scene.num_envs,
            surface_map_list=self.US_slicer.surface_map_list,
            surface_normal_list=self.US_slicer.surface_normal_list,
            label_res=label_res,
            US_height=self.US_slicer.height,
        )

        # Robot US spawn position randomization from YAML
        robot_rand_cfg = robot_cfg.get("randomization", {})
        robot_rand_pos_cfg = robot_rand_cfg["position"]

        robot_rand_pos_min = np.array(
            [
                float(robot_rand_pos_cfg["x"][0]),
                float(robot_rand_pos_cfg["y"][0]),
                float(robot_rand_pos_cfg["z"][0]),
            ],
            dtype=np.float32,
        )
        robot_rand_pos_max = np.array(
            [
                float(robot_rand_pos_cfg["x"][1]),
                float(robot_rand_pos_cfg["y"][1]),
                float(robot_rand_pos_cfg["z"][1]),
            ],
            dtype=np.float32,
        )

        self.robot_base_pos_nominal = torch.tensor(
            np.array(robot_cfg["pos"], dtype=np.float32),
            device=self.sim.device,
            dtype=torch.float32,
        )

        self.robot_spawn_pos_rand_min = torch.tensor(
            robot_rand_pos_min, device=self.sim.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.scene.num_envs, 1)

        self.robot_spawn_pos_rand_max = torch.tensor(
            robot_rand_pos_max, device=self.sim.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.scene.num_envs, 1)

        self.if_random_spawn_robot = bool(robot_cfg.get("pose_randomization", False))

        # Drill robot spawn position randomization from YAML
        robot_drill_rand_cfg = robot_drill_cfg.get("randomization", {})
        robot_drill_rand_pos_cfg = robot_drill_rand_cfg["position"]

        robot_drill_rand_pos_min = np.array(
            [
                float(robot_drill_rand_pos_cfg["x"][0]),
                float(robot_drill_rand_pos_cfg["y"][0]),
                float(robot_drill_rand_pos_cfg["z"][0]),
            ],
            dtype=np.float32,
        )
        robot_drill_rand_pos_max = np.array(
            [
                float(robot_drill_rand_pos_cfg["x"][1]),
                float(robot_drill_rand_pos_cfg["y"][1]),
                float(robot_drill_rand_pos_cfg["z"][1]),
            ],
            dtype=np.float32,
        )

        self.robot_drill_base_pos_nominal = torch.tensor(
            np.array(robot_drill_cfg["pos"], dtype=np.float32),
            device=self.sim.device,
            dtype=torch.float32,
        )

        self.robot_drill_spawn_pos_rand_min = torch.tensor(
            robot_drill_rand_pos_min, device=self.sim.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.scene.num_envs, 1)

        self.robot_drill_spawn_pos_rand_max = torch.tensor(
            robot_drill_rand_pos_max, device=self.sim.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.scene.num_envs, 1)

        self.if_random_spawn_robot_drill = bool(robot_drill_cfg.get("pose_randomization", False))

        # drill rand
        self.rand_joint_pos_max = (
            torch.tensor(self.motion_plan_cfg["joint_pos_rand_max"])
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
            .to(self.sim.device)
        )

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
        
        self.obs_img = torch.zeros(
            (
                self.scene.num_envs,
                self.us_img_thickness,
                self.us_img_height,
                self.us_img_width,
            ),
            device=self.sim.device,
            dtype=torch.float32,
        )
        
        self.probe_obs_img = torch.zeros(
            (
                self.scene.num_envs,
                1,
                self.us_img_height,
                self.us_img_width,
            ),
            device=self.sim.device,
            dtype=torch.float32,
        )

        self.left_policy = torch.jit.load(
            robot_drill_cfg["probe_policy"],
            map_location=self.sim.device,
        )
        self.left_policy.eval()

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

        self.guidance_action_scale = torch.tensor(
            [0.5, 0.5, 0.05],
            device=self.sim.device,
        ).reshape((1, -1)).repeat(self.scene.num_envs, 1)

        self.guidance_max_action = torch.tensor(
            [1.0, 1.0, 0.2],
            device=self.sim.device,
        ).reshape((1, -1)).repeat(self.scene.num_envs, 1)

        if not hasattr(self, "prev_probe_cmd"):
            self.prev_probe_cmd = self.US_slicer.current_x_z_x_angle_cmd.clone()

        self.last_probe_policy_update_step = -1

        # wandb.init()
        self.num_step = 0
            

    def _sample_random_probe_init_cmd(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Sample a random probe command exactly like in the guidance training env."""
        if env_ids is None:
            env_ids_t = torch.arange(
                self.scene.num_envs,
                device=self.sim.device,
                dtype=torch.long,
            )
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

        num = int(env_ids_t.numel())

        cmd_min = self.init_cmd_pose_min[env_ids_t]
        cmd_max = self.init_cmd_pose_max[env_ids_t]

        rand_u = torch.rand((num, 3), device=self.sim.device)
        rand_cmd = cmd_min + rand_u * (cmd_max - cmd_min)

        delta_cmd = torch.zeros_like(self.US_slicer.current_x_z_x_angle_cmd)
        delta_cmd[env_ids_t] = (
            rand_cmd - self.US_slicer.current_x_z_x_angle_cmd[env_ids_t]
        )

        self.US_slicer.update_cmd(delta_cmd)

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

        self.robot_drill = Articulation(self.cfg.robot_drill_cfg)

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

        # human:
        human_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Human",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_file_list,
                random_choice=False,
                scale=(label_res, label_res, label_res),
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
        self.scene.articulations["robot_drill"] = self.robot_drill
        self.scene.rigid_objects["human"] = self.human

        self.drill_to_tip_pos = (
            torch.tensor(DRILL_TO_TIP_POS, device=self.sim.device)
            .reshape((1, -1))
            .repeat(self.scene.num_envs, 1)
        )
        self.drill_to_tip_quat = (
            torch.tensor(DRILL_TO_TIP_QUAT, device=self.sim.device)
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
        
        self.world_to_human_pos = self.human_world_poses[:, 0:3]
        self.world_to_human_rot = self.human_world_poses[:, 3:7]
        
        ee_pose_w_robot = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]

        # define world to human poses
        ee_pos_w = ee_pose_w_robot[:, 0:3]
        ee_quat_w = ee_pose_w_robot[:, 3:7]

        # pose in US frame (for slicer)
        self.US_ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
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
        
        self.obs_img = obs_img.detach()

        if self.obs_img.ndim != 4:
            raise ValueError(f"Expected obs_img with shape (N, C, H, W), got {tuple(self.obs_img.shape)}")
        center_idx = self.obs_img.shape[1] // 2
        self.probe_obs_img = self.obs_img[:, center_idx:center_idx + 1, :, :].detach()


        if self.sim_cfg["vis_us"] and self.num_step % self.sim_cfg["vis_int"] == 0:
            self.US_slicer.visualize(self.observation_mode)
            
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

        if hasattr(self, "total_rewards"):
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

        safety_critical = self.tip_pos_along_traj > -self.safe_height
        safe_close = torch.logical_and(
            safety_critical,
            self.tip_to_traj_dist < self.vertebra_viewer.traj_radius + 0.002,
        )
        safe_close = torch.logical_and(
            safe_close, self.tip_pos_along_traj < self.vertebra_viewer.traj_half_length
        )

        actions5 = actions

        actions6 = torch.zeros((actions5.shape[0], 6), device=actions5.device, dtype=actions5.dtype)
        actions6[:, 0:5] = actions5
        actions6[:, 5] = 0.0  # yaw non attuato

        actions6[safe_close, 0:2] *= 0.2
        actions6[safe_close, 3:] *= 0.2

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

        self.diff_ik_controller_drill.set_command(
            torch.cat([drill_next_ee_pos_b, drill_next_ee_quat_b], dim=-1)
        )


    def _apply_action(self):
        self.get_drill_ee_pose_b()
        self.get_US_ee_pose_b()

        # # get joint position targets
        drill_jacobian = self.robot_drill.root_physx_view.get_jacobians()[
            :, self.drill_ee_jacobi_idx - 1, :, self.robot_drill_entity_cfg.joint_ids
        ]
        drill_joint_pos = self.robot_drill.data.joint_pos[
            :, self.robot_drill_entity_cfg.joint_ids
        ]
        # compute the joint commands
        joint_pos_des = self.diff_ik_controller_drill.compute(
            self.drill_ee_pos_b, self.drill_ee_quat_b, drill_jacobian, drill_joint_pos
        )
        # apply joint oosition target
        self.robot_drill.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_drill_entity_cfg.joint_ids
        )

        self._apply_us_command()  # fills _us_target_*_b
        # drill target already cached in _pre_physics_step: _drill_target_*_b
                # set new command
        self.pose_diff_ik_controller.set_command(self.base_to_ee_target_pose)

        # get joint position targets
        US_jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
        ]
        US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = self.pose_diff_ik_controller.compute(
            self.US_ee_pos_b, self.US_ee_quat_b, US_jacobian, US_joint_pos
        )
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
        )



    def _update_us_command_from_left_policy(self, probe_obs_img: torch.Tensor) -> None:
        """Update the probe surface command online from the frozen guidance policy."""

        with torch.no_grad():
            actions = self.left_policy(probe_obs_img.to(self.sim.device))

        # stessa logica della guidance task singola
        if actions.shape[-1] == 6:
            actions = actions[:, [0, 2, 5]]

        if self.action_mode == "continuous":
            actions = torch.clamp(
                actions * self.guidance_action_scale,
                -self.guidance_max_action,
                self.guidance_max_action,
            )
        elif self.action_mode == "discrete":
            actions = torch.sign(actions) * self.guidance_action_scale
        else:
            raise ValueError("Invalid action mode")

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

        cmd_before = self.US_slicer.current_x_z_x_angle_cmd.clone()

        self.US_slicer.update_cmd(cmd)

        cmd_after = self.US_slicer.current_x_z_x_angle_cmd.clone()
        cmd_step = cmd_after - cmd_before
        """
        if self.num_step % 25 == 0:
            print(
                "[PROBE CMD]",
                "before", cmd_before[0].detach().cpu().numpy(),
                "step", cmd_step[0].detach().cpu().numpy(),
                "after", cmd_after[0].detach().cpu().numpy(),
            )

        if self.num_step % 25 == 0:
            angle = self.US_slicer.current_x_z_x_angle_cmd[:, 2]
            print(
                "[PROBE ANGLE]",
                "min", angle.min().item(),
                "max", angle.max().item(),
                "mean", angle.mean().item(),
            )"""

        world_ee_target_pos, world_ee_target_quat = self.US_slicer.compute_world_ee_pose_from_cmd(
            self.world_to_human_pos,
            self.world_to_human_rot,
        )

        self.US_slicer.human_to_ee_target_pos, self.US_slicer.human_to_ee_target_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            world_ee_target_pos,
            world_ee_target_quat,
        )

    def _apply_us_command(self):
        self.get_US_ee_pose_b()

        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]

        # Aggiorna la probe policy una sola volta per observation step.
        # Evita doppi update durante i substep fisici dovuti a decimation.
        if (
            hasattr(self, "left_policy")
            and self.left_policy is not None
            and self.num_step > 0
            and self.last_probe_policy_update_step != self.num_step
        ):
            self._update_us_command_from_left_policy(self.probe_obs_img)
            self.last_probe_policy_update_step = self.num_step

        world_ee_target_pos, world_ee_target_quat = combine_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            self.world_to_base_pose[:, 0:3],
            self.world_to_base_pose[:, 3:7],
            world_ee_target_pos,
            world_ee_target_quat,
        )

        self.base_to_ee_target_pose = torch.cat(
            [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
        )

    def _get_rewards(self) -> torch.Tensor:

        reward = torch.zeros(self.scene.num_envs, device=self.sim.device)
        penalty = torch.zeros(self.scene.num_envs, device=self.sim.device)
        cost = torch.zeros(self.scene.num_envs, device=self.sim.device)

        if self.sim_cfg["vis_us"]:
            self.vertebra_viewer.update_tip_vis(
                self.human_to_tip_pos, self.human_to_tip_quat
            )

        # free space
        free_region = self.tip_pos_along_traj <= -self.safe_height
        reward[free_region] += self.w_pos * (
            torch.abs(self.last_tip_pos_along_traj[free_region] + self.safe_height)
            - torch.abs(self.tip_pos_along_traj[free_region] + self.safe_height)
        )

        # safety critical space
        safety_critical = self.tip_pos_along_traj > -self.safe_height
        safe_close = torch.logical_and(
            safety_critical,
            self.tip_to_traj_dist < self.vertebra_viewer.traj_radius + 0.002,
        )
        close = self.tip_to_traj_dist < self.vertebra_viewer.traj_radius + 0.002
        safe_close = torch.logical_and(
            safe_close, self.tip_pos_along_traj < self.vertebra_viewer.traj_half_length
        )
        unsafe = torch.logical_and(torch.logical_not(safe_close), safety_critical)
        self.ever_unsafe[unsafe] = 1
        always_safe = torch.logical_not(self.ever_unsafe)

        # reward insertion
        always_safe_and_close = torch.logical_and(always_safe, safe_close)
        self.max_tip_pos_along_traj = torch.maximum(
            self.tip_pos_along_traj, self.max_tip_pos_along_traj
        )

        w_i = 0.75 # 1.5

        reward[safe_close] += w_i * self.w_insertion * (
            torch.abs(
                self.last_max_tip_pos_along_traj[safe_close]
                - self.vertebra_viewer.traj_half_length[safe_close]
            )
            - torch.abs(
                self.max_tip_pos_along_traj[safe_close]
                - self.vertebra_viewer.traj_half_length[safe_close]
            )
        )
        # to avoid loop
        self.total_insertion[safe_close] += torch.abs(
            self.last_max_tip_pos_along_traj[safe_close]
            - self.vertebra_viewer.traj_half_length[safe_close]
        ) - torch.abs(
            self.max_tip_pos_along_traj[safe_close]
            - self.vertebra_viewer.traj_half_length[safe_close]
        )

        last_safe_close_now_unsafe = torch.logical_and(self.last_safe_close, unsafe)
        last_unsafe_now_safe_close = torch.logical_and(self.last_unsafe, safe_close)
        reward[last_unsafe_now_safe_close] += (
            self.w_cost * self.total_insertion[last_unsafe_now_safe_close]
        )

        penalty[last_safe_close_now_unsafe] = self.total_insertion[
            last_safe_close_now_unsafe
        ]

        reward += self.w_pos * (self.last_tip_to_traj_dist - self.tip_to_traj_dist)
        reward += self.w_angle * (
            torch.abs(self.last_traj_to_tip_sin) - torch.abs(self.traj_to_tip_sin)
        )
        
        any_close = torch.logical_or(
            self.last_close,
            close,
        )
        reward[any_close] += self.w_insertion * (
            self.last_tip_to_traj_dist[any_close] - self.tip_to_traj_dist[any_close]
        )

        w_ca = 0.05
        angle_small_threshold = 0.08

        angle_small = torch.abs(self.traj_to_tip_sin) < angle_small_threshold

        any_close_angle = torch.logical_or(
            any_close,
            angle_small,
        )

        angle_progress = (
            torch.abs(self.last_traj_to_tip_sin)
            - torch.abs(self.traj_to_tip_sin)
        )

        reward[any_close_angle] += w_ca * self.w_insertion * angle_progress[any_close_angle]

        reward -= penalty * self.w_cost

        self.total_dist = torch.sqrt(
            torch.abs(self.tip_pos_along_traj - self.vertebra_viewer.traj_half_length)
            ** 2
            + self.tip_to_traj_dist
        )
        self.angle = torch.asin(self.traj_to_tip_sin) * 180 / torch.pi
        self.insert_err = torch.abs(
            self.tip_pos_along_traj - self.vertebra_viewer.traj_half_length
        )
        self.inserted = (
            self.tip_pos_along_traj > -self.vertebra_viewer.traj_half_length
        ).reshape((-1,))

        self.last_tip_pos_along_traj = copy.deepcopy(self.tip_pos_along_traj)
        self.last_tip_to_traj_dist = copy.deepcopy(self.tip_to_traj_dist)
        self.last_traj_to_tip_sin = copy.deepcopy(self.traj_to_tip_sin)
        self.last_unsafe = copy.deepcopy(unsafe)
        self.last_safe_close = copy.deepcopy(safe_close)
        self.last_close = copy.deepcopy(close)
        # self.last_traj_pos_along_traj_safe_close[safe_close] = copy.deepcopy(self.tip_pos_along_traj[safe_close])
        self.last_max_tip_pos_along_traj = copy.deepcopy(self.max_tip_pos_along_traj)


        self.extras["cost"] = unsafe.to(torch.float32)
        self.total_costs += self.extras["cost"]

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

        if scene_cfg["if_record_traj"]:
            if not hasattr(self, "tip_pos_along_traj_trajs"):
                self.tip_pos_along_traj_trajs = []
            if not hasattr(self, "tip_to_traj_dist_trajs"):
                self.tip_to_traj_dist_trajs = []
            if not hasattr(self, "tip_roll_err_deg_trajs"):
                self.tip_roll_err_deg_trajs = []
            if not hasattr(self, "tip_pitch_err_deg_trajs"):
                self.tip_pitch_err_deg_trajs = []
            self.tip_pos_along_traj_trajs.append(self.tip_pos_along_traj)
            self.tip_to_traj_dist_trajs.append(self.tip_to_traj_dist)
            self.tip_roll_err_deg_trajs.append(self.tip_roll_err_deg)
            self.tip_pitch_err_deg_trajs.append(self.tip_pitch_err_deg)

        if hasattr(self, "total_rewards"):
            self.total_rewards += reward.detach()

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
        num_steps: int = 100,
    ):
        """Move ONLY the US arm EE towards a pose expressed in the HUMAN frame (US convention).

        This is used at reset time to move the probe from its spawn pose to a random
        initial pose sampled inside patient_xz_init_range.
        """
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(num_steps):
            self._sim_step_counter += 1
            # set actions into buffers

            # 1) HUMAN pose in WORLD
            self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
            self.world_to_human_pos = self.human_world_poses[:, 0:3]
            self.world_to_human_rot = self.human_world_poses[:, 3:7]

            # 2) Target EE in WORLD (orientation in US frame)
            world_ee_target_pos, world_ee_target_quat = combine_frame_transforms(
                self.world_to_human_pos,
                self.world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # get current joint positions
            self.US_ee_pose_w = self.robot.data.body_state_w[
                :, self.robot_entity_cfg.body_ids[-1], 0:7
            ]

            # get current ee
            US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
                self.world_to_base_pose[:, 0:3],
                self.world_to_base_pose[:, 3:7],
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                self.world_to_base_pose[:, 0:3],
                self.world_to_base_pose[:, 3:7],
                world_ee_target_pos,
                world_ee_target_quat,
            )
            base_to_ee_target_pose = torch.cat(
                [base_to_ee_target_pos, base_to_ee_target_quat], dim=-1
            )

            # set new command
            self.pose_diff_ik_controller.set_command(base_to_ee_target_pose)

            # get joint position targets
            US_jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
            ]
            US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            # compute the joint commands
            joint_pos_des = self.pose_diff_ik_controller.compute(
                US_ee_pos_b, US_ee_quat_b, US_jacobian, US_joint_pos
            )
            self.robot.set_joint_position_target(
                joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids
            )

            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

    def position_probe(self, num_steps: int = 240):
        """
        Run the frozen probe policy to position the US probe.

        num_steps is now the number of policy steps, not raw physics steps.
        This matches DirectRLEnv decimation.
        """
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        dt = float(self.physics_dt)

        for _ in range(num_steps):

            # One observation and one policy update per RL/policy step.
            self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
            self.world_to_human_pos = self.human_world_poses[:, 0:3]
            self.world_to_human_rot = self.human_world_poses[:, 3:7]

            self.get_drill_ee_pose_b()
            self._get_observations()

            self._apply_us_command()

            # Hold this target for decimation physics substeps.
            for _sub in range(self.cfg.decimation):
                self._sim_step_counter += 1

                self.get_US_ee_pose_b()
                self.pose_diff_ik_controller.set_command(self.base_to_ee_target_pose)

                US_jacobian = self.robot.root_physx_view.get_jacobians()[
                    :, self.US_ee_jacobi_idx - 1, :, self.robot_entity_cfg.joint_ids
                ]
                US_joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

                joint_pos_des = self.pose_diff_ik_controller.compute(
                    self.US_ee_pos_b,
                    self.US_ee_quat_b,
                    US_jacobian,
                    US_joint_pos,
                )

                self.robot.set_joint_position_target(
                    joint_pos_des,
                    joint_ids=self.robot_entity_cfg.joint_ids,
                )

                self.scene.write_data_to_sim()
                self.sim.step(render=False)

                if (
                    self._sim_step_counter % self.cfg.sim.render_interval == 0
                    and is_rendering
                ):
                    self.sim.render()

                self.scene.update(dt=dt)

    def get_US_ee_pose_b(self):
        self.US_root_pose_w = self.robot.data.root_state_w[:, 0:7]

        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]

        self.US_ee_pos_b, self.US_ee_quat_b = subtract_frame_transforms(
            self.US_root_pose_w[:, 0:3],
            self.US_root_pose_w[:, 3:7],
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )

    def get_drill_ee_pose_b(self):
        self.drill_root_pose_w = self.robot_drill.data.root_state_w[:, 0:7]

        self.drill_ee_pose_w = self.robot_drill.data.body_state_w[
            :, self.robot_drill_entity_cfg.body_ids[-1], 0:7
        ]

        self.drill_ee_pos_b, self.drill_ee_quat_b = subtract_frame_transforms(
            self.drill_root_pose_w[:, 0:3],
            self.drill_root_pose_w[:, 3:7],
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )
        
    def get_traj_to_tip_state(self):
        self.human_to_ee_pos, self.human_to_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.drill_ee_pose_w[:, 0:3],
            self.drill_ee_pose_w[:, 3:7],
        )
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

    def reset_controllers(self):
        self.pose_diff_ik_controller.reset()
        self.diff_ik_controller_drill.reset()

        self.get_US_ee_pose_b()

        ik_commands_pose = torch.zeros(
            self.scene.num_envs,
            self.pose_diff_ik_controller.action_dim,
            device=self.sim.device,
        )
        self.pose_diff_ik_controller.set_command(
            ik_commands_pose, self.US_ee_pos_b, self.US_ee_quat_b
        )

        self.get_drill_ee_pose_b()

        ik_commands_pose = torch.zeros(
            self.scene.num_envs,
            self.diff_ik_controller_drill.action_dim,
            device=self.sim.device,
        )
        self.diff_ik_controller_drill.set_command(
            ik_commands_pose, self.drill_ee_pos_b, self.drill_ee_quat_b
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids_tensor = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

        # Log metrics from the episode that has just ended,
        # before resetting the environment state.
        if bool(scene_cfg.get("use_wandb", False)) and hasattr(self, "total_rewards"):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

            final_angular_error_deg = torch.asin(
                torch.clamp(torch.abs(self.traj_to_tip_sin[env_ids_tensor]), 0.0, 1.0)
            ) * 180.0 / torch.pi

            wandb.log({
                "Total reward": self.total_rewards[env_ids_tensor].mean().item(),
                "Final radial error mm": (self.tip_to_traj_dist[env_ids_tensor].mean() * 1000.0).item(),
                "Final angular error deg": final_angular_error_deg.mean().item(),
                "Total insertion": self.total_insertion.mean().item()
            })

        super()._reset_idx(env_ids)

        # reconstruct random maps
        self.US_slicer.construct_T_maps()
        self.US_slicer.construct_Vl_maps()

        #  reset single articulation joint state
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        rand_joint_pos = (
            torch.rand(
                (self.scene.num_envs, self.rand_joint_pos_max.shape[1]),
                device=self.sim.device,
            )
            * 2
            - 1
        )
        rand_joint_pos = rand_joint_pos * self.rand_joint_pos_max

        if self.if_random_spawn_robot:
            # Randomize robot_US spawn position around nominal base position for all envs
            rand_u = torch.rand((self.scene.num_envs, 3), device=self.sim.device)
            rand_offset = self.robot_spawn_pos_rand_min + rand_u * (
                self.robot_spawn_pos_rand_max - self.robot_spawn_pos_rand_min
            )

            # Root pose must be written in WORLD frame
            rand_base_pos_world = (
                self.scene.env_origins
                + self.robot_base_pos_nominal.unsqueeze(0)
                + rand_offset
            )

            root_state = self.robot.data.default_root_state.clone()
            root_state[:, 0:3] = rand_base_pos_world

            # Keep default orientation and velocities unchanged
            self.robot.write_root_state_to_sim(root_state)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        self.robot.set_joint_position_target(
            joint_pos, joint_ids=self.robot_entity_cfg.joint_ids
        )

        joint_pos = self.robot_drill.data.default_joint_pos.clone()
        joint_vel = self.robot_drill.data.default_joint_vel.clone()

        if self.if_random_spawn_robot_drill:
            # Randomize robot_drill spawn position around nominal base position for all envs
            rand_u = torch.rand((self.scene.num_envs, 3), device=self.sim.device)
            rand_offset = self.robot_drill_spawn_pos_rand_min + rand_u * (
                self.robot_drill_spawn_pos_rand_max - self.robot_drill_spawn_pos_rand_min
            )

            # Root pose must be written in WORLD frame
            rand_base_pos_world = (
                self.scene.env_origins
                + self.robot_drill_base_pos_nominal.unsqueeze(0)
                + rand_offset
            )

            root_state = self.robot_drill.data.default_root_state.clone()
            root_state[:, 0:3] = rand_base_pos_world

            # Keep default orientation and velocities unchanged
            self.robot_drill.write_root_state_to_sim(root_state)

        self.robot_drill.write_joint_state_to_sim(joint_pos + rand_joint_pos, joint_vel)
        self.robot_drill.reset()
        self.robot_drill.set_joint_position_target(
            joint_pos + rand_joint_pos, joint_ids=self.robot_drill_entity_cfg.joint_ids
        )

        self.reset_controllers()
        self.last_probe_policy_update_step = -1

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

        self._sample_random_probe_init_cmd(env_ids)
        init_world_ee_pos, init_world_ee_quat = self.US_slicer.compute_world_ee_pose_from_cmd(
            self.world_to_human_pos,
            self.world_to_human_rot,
        )
        init_human_ee_pos, init_human_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            init_world_ee_pos,
            init_world_ee_quat,
        )

        # Phase 1: move the probe from spawn to a random initial pose on the back.
        self._move_towards_target(
            init_human_ee_pos,
            init_human_ee_quat,
            num_steps=100,
        )

        self.get_drill_ee_pose_b()
        self._get_observations()

        # Phase 2: run only the frozen probe policy to position the probe online.
        self.position_probe(num_steps=250)

        # Refresh final state after online probe positioning.
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos = self.human_world_poses[:, 0:3]
        self.world_to_human_rot = self.human_world_poses[:, 3:7]
        self.world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        self.get_US_ee_pose_b()

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

        self.ever_unsafe = torch.zeros(
            self.scene.num_envs,
            dtype=torch.bool,
            device=self.sim.device,
        )
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

        # persist frequently: reset
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
