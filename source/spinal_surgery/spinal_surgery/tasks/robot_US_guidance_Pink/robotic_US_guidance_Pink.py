from __future__ import annotations

import os
import time
from collections.abc import Sequence

import gymnasium as gym
import nibabel as nib
import numpy as np
import torch
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
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    subtract_frame_transforms,
)

from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import G1_TOOLS_BASE_FIX_CFG
from spinal_surgery.assets.unitreeH1 import H12_CFG_TOOLS_BASEFIX
from spinal_surgery.lab.kinematics.gt_motion_generator import (
    GTDiscreteMotionGenerator,
)
from spinal_surgery.lab.kinematics.vertebra_viewer import VertebraViewer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer

# --- Pink / Pinocchio ---
import pinocchio as pin
import pink
from pink.configuration import Configuration
from pink.tasks import FrameTask, PostureTask
from pink.solve_ik import solve_ik
from pink.exceptions import NoSolutionFound
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

### YAML configuration ###
scene_cfg = YAML().load(
    open(
        f"{PACKAGE_DIR}/tasks/robot_US_guidance_Pink/cfgs/robotic_US_guidance_Pink.yaml",
        "r",
    )
)

# If using US network, change observation scaling
if scene_cfg["sim"]["us"] == "net":
    scene_cfg["observation"]["scale"] = scene_cfg["observation"]["scale_net"]

robot_cfg = scene_cfg["robot"]
robot_type = str(robot_cfg.get("type", "g1")).lower()   # 'g1' or 'h1'
robot_side = str(robot_cfg.get("side", "left")).lower() # 'left' or 'right'

if robot_type not in ["g1", "h1"]:
    raise ValueError(f"robot.type must be 'g1' or 'h1', got: {robot_type!r}")

### Robot initial state (Unitree G1 / H1) ###

# Select base articulation config depending on robot type
if robot_type == "g1":
    base_robot_cfg: ArticulationCfg = G1_TOOLS_BASE_FIX_CFG
elif robot_type == "h1":
    # H1-2 config without hands (as in the standalone script)
    base_robot_cfg: ArticulationCfg = H12_CFG_TOOLS_BASEFIX
else:
    raise ValueError(f"Unknown robot type: {robot_type!r}")

# Initialize robot pose from YAML 
robot_pose_cfg = scene_cfg[robot_type]

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
)  # xyzw → wxyz

# Heights for US slicer and surface motion planning
ROBOT_HEIGHT = float(robot_pose_cfg["height"])
ROBOT_HEIGHT_IMG = float(robot_pose_cfg["height_img"])

# Build a clean initial state for the selected robot (do NOT mutate base config)
robot_init_state = ArticulationCfg.InitialStateCfg(
    pos=base_pos,
    rot=q_wxyz,
    joint_pos=base_robot_cfg.init_state.joint_pos,
    joint_vel=base_robot_cfg.init_state.joint_vel,
)

robot_articulation_cfg = base_robot_cfg.replace(init_state=robot_init_state)

### Patient / bed / datasets ###

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

# Dataset paths
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
scale = 1.0 / label_res

### Pink / Pinocchio helpers ###

# NOTE: adjust the H1 URDF path to match your local asset layout.
if robot_type == "g1":
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif robot_type == "h1":
    # TODO: replace this with the correct H1 URDF path if different
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/h1/h1_body29_hand14.urdf"       # STILL NEED THIS ONE
else:
    raise ValueError(f"Unsupported robot_type for URDF selection: {robot_type!r}")


def build_pinocchio_model(urdf_path: str):
    """Build Pinocchio model and data from URDF path."""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def build_name_to_qidx(model: pin.Model):
    """Mapping joint name -> q index (ignoring floating base)."""
    name_to_qidx = {}
    for j_id, joint in enumerate(model.joints):
        name = model.names[j_id]
        if joint.nq == 1:  # 1-DoF joints only
            name_to_qidx[name] = joint.idx_q
    return name_to_qidx


def isaac_to_pin_q(
    q_isaac: torch.Tensor,
    isaac_joint_names: list[str],
    model: pin.Model,
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    """Isaac joint vector -> Pinocchio q (fixed base)."""
    q_pin = np.zeros(model.nq, dtype=float)
    q_isaac_np = q_isaac.detach().cpu().numpy()
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            q_pin[name_to_qidx[jname]] = q_isaac_np[i]
    return q_pin


def pin_to_isaac_q(
    q_pin: np.ndarray,
    isaac_joint_names: list[str],
    name_to_qidx: dict[str, int],
) -> np.ndarray:
    """Pinocchio q -> Isaac joint vector (same order as Isaac)."""
    q_isaac = np.zeros(len(isaac_joint_names), dtype=float)
    for i, jname in enumerate(isaac_joint_names):
        if jname in name_to_qidx:
            q_isaac[i] = q_pin[name_to_qidx[jname]]
    return q_isaac


def safe_solve_ik(configuration, tasks, dt, **kwargs):
    """Wrapper around Pink's solve_ik that never throws in the RL loop."""
    try:
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            # pass through optional args, e.g. solver, safety_break, limits, ...
            **kwargs,
        )
        return velocity
    except NoSolutionFound:
        # Minimal, non-crashing fallback: zero velocity in tangent space
        return np.zeros(configuration.model.nv)
    except Exception:
        # Catch-all to avoid killing a long training run
        return np.zeros(configuration.model.nv)

### Env config ###

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

### Env ###

class roboticUSEnv(DirectRLEnv):
    cfg: roboticUSEnvCfg

    def __init__(self, cfg: roboticUSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache robot type and side from YAML
        self.robot_type = robot_type
        self.EE_LINK_SIDE = robot_side

        # EE link & kinematic chain
        if "left" in self.EE_LINK_SIDE:
            self.EE_LINK_NAME = "left_wrist_yaw_link"
        elif "right" in self.EE_LINK_SIDE:
            self.EE_LINK_NAME = "right_wrist_yaw_link"
        else:
            raise ValueError("robot.side must contain 'left' o 'right'")

        self.joint_pattern = rf"(waist_(roll|pitch|yaw)_joint|{self.EE_LINK_SIDE}_(shoulder|elbow|wrist)_.*)"
        self.robot_entity_cfg = SceneEntityCfg(
            "robot_US",
            joint_names=[self.joint_pattern],
            body_names=[self.EE_LINK_NAME],
        )
        self.robot_entity_cfg.resolve(self.scene)
        self.US_ee_jacobi_idx = self.robot_entity_cfg.body_ids[-1]

        # Rotation G1 frame -> US frame (SonoGym convention)
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
        )  # (N,3,3)

        # Label / CT load
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

        # Label conversion map
        label_convert_map = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", "r")
        )

        # US slicer config
        us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", "r"))
        us_generative_cfg = YAML().load(
            open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_generative_cfg.yaml", "r")
        )
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
            visualize=self.sim_cfg["vis_seg_map"],
            sim_mode=scene_cfg["sim"]["us"],
            us_generative_cfg=us_generative_cfg,
            height=ROBOT_HEIGHT,
            height_img=ROBOT_HEIGHT_IMG,
        )
        self.US_slicer.current_x_z_x_angle_cmd = (
            self.init_cmd_pose_min + self.init_cmd_pose_max
        ) / 2

        # Human root pose (initial)
        self.human_world_poses = self.human.data.root_state_w
        self.human_init_root_state = self.human.data.root_state_w.clone()

        # GT motion generator config
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

        # Observation & action spaces initialization
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.cfg.observation_space[0],
                self.cfg.observation_space[1],
                self.cfg.observation_space[2],
            ),
            dtype=np.uint8,
        )

        self.cfg.observation_space[0] = self.US_slicer.img_thickness

        self.single_observation_space["policy"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.cfg.observation_space[0],
                self.cfg.observation_space[1],
                self.cfg.observation_space[2],
            ),
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

        # Reward configuration 
        reward_cfg = scene_cfg["reward"]
        self.w_pos = reward_cfg.get("w_pos", 0.03)

        # Success parameters 
        self.success_radius = reward_cfg.get("success_radius", 0.01)  # [m] default
        self.success_radius_human = self.success_radius / label_res
        self.success_time = reward_cfg.get("success_time", 1.0)       # [s] default
        self.success_bonus = reward_cfg.get("success_bonus", 1.0)
        self.action_penalty = reward_cfg.get("action_penalty", 0.0)
        self.penalty_radius_human = self.success_radius_human

        self.success_steps_required = max(
            1, int(np.ceil(self.success_time / self.physics_dt))
        )

        # counter of consecutive "inside radius" steps
        self.success_counter = torch.zeros(
            self.scene.num_envs,
            device=self.sim.device,
            dtype=torch.long,
        )

        self.single_action_space = gym.spaces.Box(
            low=-(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            high=(self.max_action[0, :] / self.action_scale[0, :]).cpu().numpy(),
            shape=(self.cfg.action_space,),
            dtype=np.float32,
        )

        self.num_step = 0

        # Pink IK setup
        self._pin_model, self._pin_data = build_pinocchio_model(URDF_PATH)
        self._name_to_qidx = build_name_to_qidx(self._pin_model)

        # Joint names in Isaac (order)
        self._joint_names = list(self.robot.data.joint_names)

        # q0: default joint pos env 0
        q0_isaac = self.robot.data.default_joint_pos[0]
        q0_pin = isaac_to_pin_q(
            q0_isaac, self._joint_names, self._pin_model, self._name_to_qidx
        )
        self._pin_configuration = Configuration(self._pin_model, self._pin_data, q0_pin)

        # EE task in Pink 
        self._ee_task = FrameTask(
            frame=self.EE_LINK_NAME,
            position_cost=1.0,
            orientation_cost=1.0,
        )
        self._ik_tasks = [self._ee_task]

        # Indices of controlled joints (waist + selected arm)
        self._ctrl_joint_ids = self.robot_entity_cfg.joint_ids

        # Global counters for successes / timeouts
        self.total_successes = 0
        self.total_timeouts = 0

        # Local episode logging counter
        self.log_counter = 0

        # Local log buffer to be saved to .pt
        self.stats_log = {
            "total_reward_mean": [],
            "pos_err_mean": [],
            "pos_err_max": [],
            "total_successes": [],
            "total_timeouts": [],
        }

    ### Utility: target US pose from vertebrae ###

    def get_US_target_pose(self):
        vertebra_to_US_2d_pos = torch.tensor(
            scene_cfg["motion_planning"]["vertebra_to_US_2d_pos"],
            device=self.sim.device,
        )

        vertebra_2d_pos = self.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
        US_target_2d_pos = vertebra_2d_pos + vertebra_to_US_2d_pos.unsqueeze(0)

        US_target_2d_angle = self.goal_cmd_pose[:, 2:3] * torch.ones_like(
            vertebra_2d_pos[:, 0:1]
        )

        US_target_2d = torch.cat([US_target_2d_pos, US_target_2d_angle], dim=-1)
        self.goal_cmd_pose = US_target_2d

    ### Scene setup ###

    def _setup_scene(self):
        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # lights
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        )
        dome_light_cfg.func("/World/Light", dome_light_cfg)

        # robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # medical bed
        if scene_cfg["sim"]["vis_us"]:
            usd_folder = "usd_colored"
        else:
            usd_folder = "usd_no_contact"

        medical_bed_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bed",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/{usd_folder}/hospital_bed.usd",
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
        medical_bed = RigidObject(medical_bed_cfg)

        # human
        human_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Human",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_file_list,
                random_choice=False,
                scale=(label_res, label_res, label_res),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # Keep the human rigidly fixed in space (patient as a static/kinematic body)
                    disable_gravity=True,          # no gravity applied to the human
                    kinematic_enabled=True,        # not affected by physics impulses/forces
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

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human

    ### Observations ###

    def _get_observations(self) -> dict:
        # human frame
        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )

        # EE pose in WORLD in G1 frame 
        ee_pose_w_robot = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]
        ee_pos_w = ee_pose_w_robot[:, 0:3]
        ee_quat_w = ee_pose_w_robot[:, 3:7]

        # Rotating orientation G1 -> US frame
        ee_rotmat_w = matrix_from_quat(ee_quat_w)
        us_rotmat_w = torch.bmm(ee_rotmat_w, self.RotMat)
        us_quat_w = quat_from_matrix(us_rotmat_w)

        # Pose in US frame 
        self.US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)

        self.num_step += 1

        if self.observation_mode == "US":
            self.US_slicer.slice_US(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            US_img = (
                self.US_slicer.us_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
            observations = {"policy": US_img}
        elif self.observation_mode == "CT":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            CT_img = (
                self.US_slicer.ct_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
            observations = {"policy": CT_img}
        elif self.observation_mode == "seg":
            self.US_slicer.slice_label_img(
                self.world_to_human_pos,
                self.world_to_human_rot,
                self.US_ee_pose_w[:, 0:3],
                self.US_ee_pose_w[:, 3:7],
            )
            label_img = (
                self.US_slicer.label_img_tensor.permute(0, 3, 1, 2)
                * self.cfg.observation_scale
            )
            observations = {"policy": label_img}
        else:
            raise ValueError("Invalid observation mode")

        if self.sim_cfg["vis_us"] and self.num_step % self.sim_cfg["vis_int"] == 0:
            self.US_slicer.visualize(self.observation_mode)

        return observations

    ### Pre-physics step: target in US frame → WORLD frame → BASE frame ###

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Optional 6D → 3D conversion
        if actions.shape[-1] == 6:
            actions = actions[:, [0, 2, 5]]

        if self.action_mode == "continuous":
            actions = torch.clamp(
                actions * self.action_scale, -self.max_action, self.max_action
            )
        elif self.action_mode == "discrete":
            actions = torch.sign(actions) * self.action_scale
        else:
            raise ValueError("Invalid action mode")

        self.actions = actions

        # Transform action from US frame to human frame
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
        cmd = torch.cat(
            [dx_dz_human[:, [0, 2]], actions[:, 2:3]],
            dim=-1,
        )
        self.US_slicer.update_cmd(cmd)

        # Target WORLD pose 
        world_to_ee_target_pos, world_to_ee_target_rot_us = (
            self.US_slicer.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos,
                self.world_to_human_rot,
            )
        )

        # Rotate orientation from US frame to G1 robot frame
        world_to_ee_target_rotmat_us = matrix_from_quat(world_to_ee_target_rot_us)
        world_to_ee_target_rotmat_robot = torch.bmm(
            world_to_ee_target_rotmat_us,
            self.RotMat.transpose(1, 2),
        )
        world_to_ee_target_rot = quat_from_matrix(world_to_ee_target_rotmat_robot)

        # Target BASE→EE pose (G1 frame) for Pink
        world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3],
            world_to_base_pose[:, 3:7],
            world_to_ee_target_pos,
            world_to_ee_target_rot,
        )

        # Store target for Pink 
        self._base_to_ee_target_pos = base_to_ee_target_pos
        self._base_to_ee_target_quat = base_to_ee_target_quat

        # extras
        self.extras["human_to_ee_pos"] = human_to_ee_pos
        self.extras["human_to_ee_quat"] = human_to_ee_quat

    ### Apply action with Pink IK ###

    def _apply_action(self):
        # If no valid target yet (first steps), do nothing
        if not hasattr(self, "_base_to_ee_target_pos"):
            return

        num_envs = self.scene.num_envs
        ctrl_ids = self._ctrl_joint_ids

        arm_q_list = []

        # Solve IK for each env separately
        for env_id in range(num_envs):

            # select target pose for this env
            target_pos_np = (
                self._base_to_ee_target_pos[env_id].detach().cpu().numpy().astype(float)
            )
            R_target = (
                matrix_from_quat(
                    self._base_to_ee_target_quat[env_id].unsqueeze(0)
                )[0]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )

            #set target in solver
            T_target = pin.SE3(R_target, target_pos_np)
            self._ee_task.set_target(T_target)

            # get joint pos + pinocchio configuration
            q_curr_isaac = self.robot.data.joint_pos[env_id]
            q_curr_pin = isaac_to_pin_q(
                q_curr_isaac,
                self._joint_names,
                self._pin_model,
                self._name_to_qidx,
            )

            self._pin_configuration.update(q_curr_pin)

            # solve IK
            vel = safe_solve_ik(
                self._pin_configuration,
                tasks=self._ik_tasks,
                dt=self.cfg.sim.dt,
                solver="quadprog",
                damping=1e-2,
                safety_break=False,
            )
            # update pinocchio configuration
            self._pin_configuration.integrate_inplace(vel, self.cfg.sim.dt)
            q_next_pin = self._pin_configuration.q.copy()
            q_next_isaac = pin_to_isaac_q(
                q_next_pin,
                self._joint_names,
                self._name_to_qidx,
            )

            arm_q_np = q_next_isaac[ctrl_ids]
            arm_q_list.append(arm_q_np)

        # Stack and convert to torch
        arm_q_np_stacked = np.stack(arm_q_list, axis=0)
        arm_q_t = torch.from_numpy(arm_q_np_stacked).to(
            device=self.sim.device,
            dtype=self.robot.data.joint_pos.dtype,
        )

        # Clamp on joint limits
        joint_limits = self.robot.data.joint_pos_limits  # (N, num_joints, 2)
        joint_ids_t = torch.as_tensor(
            ctrl_ids, device=self.sim.device, dtype=torch.long
        )
        joint_min = joint_limits[0, joint_ids_t, 0]
        joint_max = joint_limits[0, joint_ids_t, 1]
        joint_min = joint_min.unsqueeze(0).expand_as(arm_q_t)
        joint_max = joint_max.unsqueeze(0).expand_as(arm_q_t)
        safety_margin = 1e-3
        arm_q_t = torch.clamp(
            arm_q_t,
            joint_min + safety_margin,
            joint_max - safety_margin,
        )

        # Apply PD position targets
        self.robot.set_joint_position_target(
            arm_q_t,
            joint_ids=joint_ids_t,
        )

        # Logging delta-q norm
        dq = arm_q_t - self.robot.data.joint_pos[:, ctrl_ids]
        if scene_cfg["if_record_traj"]:
            if not hasattr(self, "dq_norm_trajs"):
                self.dq_norm_trajs = []
            dq_norm = torch.linalg.norm(dq, dim=-1)
            self.dq_norm_trajs.append(dq_norm.clone())

    ### Reward ###

    def _get_rewards(self) -> torch.Tensor:
        # Current cmd pose in human frame
        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            self.world_to_human_pos,
            self.world_to_human_rot,
            self.US_ee_pose_w[:, 0:3],
            self.US_ee_pose_w[:, 3:7],
        )
        self.cur_cmd_pose = self.gt_motion_generator.human_cmd_state_from_ee_pose(
            cur_human_ee_pos,
            cur_human_ee_quat,
        )

        # Distance to goal = w_pos * pos_err + ang_err
        pos_err_xy = torch.norm(
            self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2],
            dim=-1,
        )
        ang_err = torch.norm(
            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3],
            dim=-1,
        )

        # Position error in human frame 
        self.pos_err_human = pos_err_xy

        cur_distance_to_goal = pos_err_xy * self.w_pos + ang_err

        # compute reward
        reward = self.distance_to_goal - cur_distance_to_goal
        self.distance_to_goal = cur_distance_to_goal

        # Success region and action penalty near the target
        within_radius = self.pos_err_human <= self.success_radius_human

        # Optional action penalty in a small radius to encourage staying still
        if self.action_penalty > 0.0 and hasattr(self, "actions"):
            in_penalty_region = self.pos_err_human <= self.penalty_radius_human

            action_norm = torch.norm(self.actions, dim=-1)
            penalty = self.action_penalty * (action_norm**2)

            reward = reward - in_penalty_region.float() * penalty

        # Success counter and terminal bonus
        self.success_counter = torch.where(
            within_radius,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter),
        )

        just_succeeded = self.success_counter == self.success_steps_required

        if self.success_bonus > 0.0:
            reward = reward + just_succeeded.float() * self.success_bonus

        # Accumulate reward for logging
        self.total_reward += reward

        # extras
        self.extras["cur_cmd_pose"] = self.cur_cmd_pose
        self.extras["goal_cmd_pose"] = self.goal_cmd_pose
        self.extras["within_radius"] = within_radius
        self.extras["just_succeeded"] = just_succeeded

        if scene_cfg["if_record_traj"]:
            self.cmd_pose_trajs.append(self.cur_cmd_pose)

        return reward
    
    ### Termination check ###

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Timeout condition
        if self.termination_direct:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros_like(
                self.episode_length_buf,
                dtype=torch.bool,
            )

        # Success = enough consecutive steps inside radius
        success = self.success_counter >= self.success_steps_required

        terminated = success.clone()

        # Store last success/timeout for logging in _reset_idx
        self._last_success = success
        self._last_timeout = time_out

        self.extras["success"] = success

        return terminated, time_out

    def _move_towards_target(
        self,
        human_ee_target_pos: torch.Tensor,
        human_ee_target_quat: torch.Tensor,
        num_steps: int = 200,
    ):
        """Move the EE from spawn pose towards a random target pose expressed in the human frame (US)."""
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(num_steps):
            self._sim_step_counter += 1

            # human frame in WORLD
            self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
            self.world_to_human_pos, self.world_to_human_rot = (
                self.human_world_poses[:, 0:3],
                self.human_world_poses[:, 3:7],
            )

            # target EE in WORLD (US frame)
            world_ee_target_pos, world_ee_target_quat_us = combine_frame_transforms(
                self.world_to_human_pos,
                self.world_to_human_rot,
                human_ee_target_pos,
                human_ee_target_quat,
            )

            # from US frame to G1 frame
            world_ee_target_rot_us = matrix_from_quat(world_ee_target_quat_us)
            world_ee_target_rot_robot = torch.bmm(
                world_ee_target_rot_us,
                self.RotMat.transpose(1, 2),
            )
            world_ee_target_quat = quat_from_matrix(world_ee_target_rot_robot)

            # base in WORLD
            world_to_base_pose = self.robot.data.root_link_state_w[:, 0:7]
            base_pos_w = world_to_base_pose[:, 0:3]
            base_quat_w = world_to_base_pose[:, 3:7]

            # target in BASE frame
            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                world_ee_target_pos,
                world_ee_target_quat,
            )

            # Pink IK on all envs
            num_envs = self.scene.num_envs
            ctrl_ids = self._ctrl_joint_ids
            arm_q_list = []

            for env_id in range(num_envs):
                target_pos_np = (
                    base_to_ee_target_pos[env_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                R_target = (
                    matrix_from_quat(
                        base_to_ee_target_quat[env_id].unsqueeze(0)
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )

                T_target = pin.SE3(R_target, target_pos_np)
                self._ee_task.set_target(T_target)

                q_curr_isaac = self.robot.data.joint_pos[env_id]
                q_curr_pin = isaac_to_pin_q(
                    q_curr_isaac,
                    self._joint_names,
                    self._pin_model,
                    self._name_to_qidx,
                )

                self._pin_configuration.update(q_curr_pin)

                vel = safe_solve_ik(
                    self._pin_configuration,
                    tasks=self._ik_tasks,
                    dt=self.cfg.sim.dt,
                    solver="quadprog",
                    damping=1e-2,
                    safety_break=False,
                )

                self._pin_configuration.integrate_inplace(vel, self.cfg.sim.dt)
                q_next_pin = self._pin_configuration.q.copy()
                q_next_isaac = pin_to_isaac_q(
                    q_next_pin,
                    self._joint_names,
                    self._name_to_qidx,
                )

                arm_q_np = q_next_isaac[ctrl_ids]
                arm_q_list.append(arm_q_np)

            arm_q_np_stacked = np.stack(arm_q_list, axis=0)
            arm_q_t = torch.from_numpy(arm_q_np_stacked).to(
                device=self.sim.device,
                dtype=self.robot.data.joint_pos.dtype,
            )

            # Clamp to joint limits
            joint_limits = self.robot.data.joint_pos_limits
            joint_ids_t = torch.as_tensor(
                ctrl_ids, device=self.sim.device, dtype=torch.long
            )
            joint_min = joint_limits[0, joint_ids_t, 0]
            joint_max = joint_limits[0, joint_ids_t, 1]
            joint_min = joint_min.unsqueeze(0).expand_as(arm_q_t)
            joint_max = joint_max.unsqueeze(0).expand_as(arm_q_t)
            safety_margin = 1e-3
            arm_q_t = torch.clamp(
                arm_q_t,
                joint_min + safety_margin,
                joint_max - safety_margin,
            )

            self.robot.set_joint_position_target(
                arm_q_t,
                joint_ids=joint_ids_t,
            )

            self.scene.write_data_to_sim()
            self.sim.step(render=False)

            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

    ### Reset ###

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset success counter for the environments being reset
        env_ids_t = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)
        self.success_counter[env_ids_t] = 0

        # robot reset
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()

        # human reset
        self.human.write_root_state_to_sim(self.human_init_root_state)
        self.human.reset()

        # initial robot/human poses
        self.US_root_pose_w = self.robot.data.root_state_w[:, 0:7]
        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]

        # change orientation from G1 frame to US frame
        ee_pos_w = self.US_ee_pose_w[:, 0:3]
        ee_quat_w = self.US_ee_pose_w[:, 3:7]
        ee_rotmat_w = matrix_from_quat(ee_quat_w)
        us_rotmat_w = torch.bmm(ee_rotmat_w, self.RotMat)
        us_quat_w = quat_from_matrix(us_rotmat_w)
        self.US_ee_pose_w = torch.cat([ee_pos_w, us_quat_w], dim=-1)

        self.human_world_poses = self.human.data.body_link_state_w[:, 0, 0:7]
        self.world_to_human_pos, self.world_to_human_rot = (
            self.human_world_poses[:, 0:3],
            self.human_world_poses[:, 3:7],
        )

        # random targets on surface
        cmd_target_poses = torch.rand(
            (self.scene.num_envs, 3), device=self.sim.device
        )
        min_init = self.init_cmd_pose_min
        max_init = self.init_cmd_pose_max
        cmd_target_poses = cmd_target_poses * (max_init - min_init) + min_init

        self.US_slicer.update_cmd(
            cmd_target_poses - self.US_slicer.current_x_z_x_angle_cmd
        )

        # compute initial WORLD EE pose from initial command
        world_to_ee_init_pos, world_to_ee_init_rot = (
            self.US_slicer.compute_world_ee_pose_from_cmd(
                self.world_to_human_pos, self.world_to_human_rot
            )
        )
        if scene_cfg["if_record_traj"]:
            if not hasattr(self, "mtt_target"):
                self.mtt_target = []
            self.mtt_target = self.US_slicer.current_x_z_x_angle_cmd.clone()

        # move towards target (Pink IK)
        self._move_towards_target(
            self.US_slicer.human_to_ee_target_pos,
            self.US_slicer.human_to_ee_target_quat,
        )

        # update EE pose in US frame after motion
        self.US_ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[-1], 0:7
        ]

        # rotate orientation from G1 frame to US frame
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

        # Local logging if reward already defined
        if hasattr(self, "total_reward"):
            env_ids_t = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

            # update global success / timeout counters
            if hasattr(self, "_last_success") and hasattr(self, "_last_timeout"):
                succ_here = self._last_success[env_ids_t]
                to_here = self._last_timeout[env_ids_t]
                self.total_successes += int(succ_here.sum().item())
                self.total_timeouts += int(to_here.sum().item())

            # position error in mm in patient frame (for logging)
            pos_err = (
                torch.norm(
                    self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2],
                    dim=-1,
                )
                * self.US_slicer.label_res
            )

            pos_err_mean = pos_err.mean().item()
            pos_err_max = pos_err.max().item()
            total_reward_mean = self.total_reward.mean().item()

            # update log counter
            self.log_counter += 1

            # update in-memory log buffer
            self.stats_log["total_reward_mean"].append(total_reward_mean)
            self.stats_log["pos_err_mean"].append(pos_err_mean)
            self.stats_log["pos_err_max"].append(pos_err_max)
            self.stats_log["total_successes"].append(self.total_successes)
            self.stats_log["total_timeouts"].append(self.total_timeouts)

            # path where to save .pt file
            record_path = PACKAGE_DIR + scene_cfg["record_path"]
            os.makedirs(record_path, exist_ok=True)
            stats_file = os.path.join(record_path, "training_stats.pt")

            torch.save(self.stats_log, stats_file)

        # Init distance to goal and reward
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
            torch.norm(self.cur_cmd_pose[:, 0:2] - self.goal_cmd_pose[:, 0:2], dim=-1)
            * self.w_pos
        )
        self.distance_to_goal += torch.norm(
            self.cur_cmd_pose[:, 2:3] - self.goal_cmd_pose[:, 2:3], dim=-1
        )

        self.D0 = self.distance_to_goal.clone()
        self.total_reward = torch.zeros(self.scene.num_envs, device=self.sim.device)

        self.extras["human_to_ee_pos"] = cur_human_ee_pos
        self.extras["human_to_ee_quat"] = cur_human_ee_quat
        self.extras["cur_cmd_pose"] = self.cur_cmd_pose
        self.extras["goal_cmd_pose"] = self.goal_cmd_pose

        if scene_cfg["if_record_traj"]:
            record_path = PACKAGE_DIR + scene_cfg["record_path"]
            if hasattr(self, "cmd_pose_trajs"):
                if not os.path.exists(record_path):
                    os.makedirs(record_path)
                self.cmd_pose_trajs = torch.stack(self.cmd_pose_trajs, dim=1)
                torch.save(self.cmd_pose_trajs, record_path + "cmd_pose_trajs.pt")
                torch.save(self.goal_cmd_pose, record_path + "goal_cmd_pose.pt")
            self.cmd_pose_trajs = [self.cur_cmd_pose]

            if hasattr(self, "dq_norm_trajs"):
                self.dq_norm_trajs = torch.stack(self.dq_norm_trajs, dim=1)
                torch.save(self.dq_norm_trajs, record_path + "dq_norm_trajs.pt")
                self.dq_norm_trajs = []

            if hasattr(self, "mtt_target"):
                torch.save(self.mtt_target, record_path + "mtt_target.pt")
                self.mtt_target = []