# guided_surgery_mock_bimanual_pink_qp_g1h1.py  (FIXED: frame visualization variables)
# Copyright...
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Spawn robot + visualize frames (no IK).")
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
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    subtract_frame_transforms,
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *

from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import cProfile


scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", "r"))
sim_cfg_yaml = scene_cfg["sim"]

# robot selector
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")  # default: g1

LEFT_EE_NAME = "left_wrist_yaw_link"
RIGHT_EE_NAME = "right_wrist_yaw_link"

if robot_type == "g1":
    ROBOT_CFG: ArticulationCfg = G1_PROVA.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/G1"
    URDF_PATH = f"{ASSETS_DATA_DIR}/unitree/robots/urdf/g1/g1_body29_hand14.urdf"
elif robot_type == "h1":
    ROBOT_CFG: ArticulationCfg = H12_CFG_TOOLS_BASEFIX.copy()
    ROBOT_CFG.prim_path = "/World/envs/env_.*/H1"
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

# Right tool: wrist -> tip
if robot_type == "g1":
    DRILL_TO_TIP_POS = np.array([0.305, 0.0, 0.0], dtype=np.float32)
else:
    DRILL_TO_TIP_POS = np.array([0.328, 0.0, 0.0], dtype=np.float32)

q_xyzw = R.from_euler("YXZ", [0, 0, 0], degrees=True).as_quat().astype(np.float32)
DRILL_TO_TIP_QUAT_WXYZ = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

WRIST_TO_TIP_POS = torch.tensor(DRILL_TO_TIP_POS, dtype=torch.float32)            # (3,)
WRIST_TO_TIP_QUAT_WXYZ = torch.tensor(DRILL_TO_TIP_QUAT_WXYZ, dtype=torch.float32)  # (4,) wxyz


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
    if robot_type == "g1":
        g1 = ROBOT_CFG
    else:
        h1 = ROBOT_CFG


def run(sim: SimulationContext, scene: InteractiveScene):
    robot_name = "g1" if robot_type == "g1" else "h1"
    robot: Articulation = scene[robot_name]

    # Resolve wrist body ids (needed to read body_state_w)
    left_cfg = SceneEntityCfg(robot_name, body_names=[LEFT_EE_NAME])
    left_cfg.resolve(scene)
    left_ee_id = int(left_cfg.body_ids[-1])

    right_cfg = SceneEntityCfg(robot_name, body_names=[RIGHT_EE_NAME])
    right_cfg.resolve(scene)
    right_ee_id = int(right_cfg.body_ids[-1])

    # Precompute fixed transforms for wrist -> tip (batched)
    # Precompute fixed transforms for wrist -> tip (batched)
    wrist_to_tip_pos = WRIST_TO_TIP_POS.to(sim.device).unsqueeze(0).repeat(scene.num_envs, 1)
    wrist_to_tip_quat = WRIST_TO_TIP_QUAT_WXYZ.to(sim.device).unsqueeze(0).repeat(scene.num_envs, 1)

    # Left probe tip offset: along +Z of LEFT EE by ROBOT_HEIGHT (meters)
    ROBOT_HEIGHT = float(scene_cfg[robot_type]["height"])
    left_tip_offset_pos = torch.tensor([ROBOT_HEIGHT, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    left_tip_offset_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)


    # Inverse transform tip -> wrist (in same convention used by IsaacLab math utils)
    tip_to_wrist_pos, tip_to_wrist_quat = subtract_frame_transforms(
        wrist_to_tip_pos,
        wrist_to_tip_quat,
        torch.zeros_like(wrist_to_tip_pos),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device).repeat(scene.num_envs, 1),
    )

    left_joint_pattern = rf"(torso_joint|waist_(pitch|roll|yaw)_joint|left_(shoulder|elbow|wrist)_.*)"
    right_joint_pattern = rf"(torso_joint|waist_(pitch|roll|yaw)_joint|right_(shoulder|elbow|wrist)_.*)"

    # Resolve EE and joint ids (for PD application)
    left_entity_cfg = SceneEntityCfg(robot_name, joint_names=[left_joint_pattern], body_names=[LEFT_EE_NAME])
    left_entity_cfg.resolve(scene)
    left_ee_id = left_entity_cfg.body_ids[-1]

    right_entity_cfg = SceneEntityCfg(robot_name, joint_names=[right_joint_pattern], body_names=[RIGHT_EE_NAME])
    right_entity_cfg.resolve(scene)
    right_ee_id = right_entity_cfg.body_ids[-1]
    
    # Full joint set for a single set_joint_position_target call
    # Note: left includes waist joints; right is arm-only (as per your current patterns)
    full_joint_ids_list = sorted(set(left_entity_cfg.joint_ids + right_entity_cfg.joint_ids))
    full_joint_ids = torch.tensor(full_joint_ids_list, device=sim.device, dtype=torch.long)

    def _make_frame_vis(path: str, scale: float) -> VisualizationMarkers:
        return VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=path,
                markers={
                    "frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(scale*0.3, scale*0.3, scale*0.3),
                    ),
                },
            )
        )

    vis_right_tip = _make_frame_vis("/Visuals/frames/right_tip", 0.08)
    vis_left_tip = _make_frame_vis("/Visuals/frames/right_tip", 0.08)

    # Reset caches
    default_pos = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(robot.data.joint_vel)

    sim_dt = sim.get_physics_dt()
    step_i = 0
    sim_time_acc = 0.0

    # A simple “dummy” tip target in BASE frame:
    # put it 0.6m forward and 0.2m above the base, pointing forward (identity quat)
    tip_tgt_pos_b = torch.tensor([0.60, 0.00, 0.20], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    tip_tgt_quat_b = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)

    while simulation_app.is_running():
        # periodic reset
        if step_i % int(sim_cfg_yaml["episode_length"]) == 0:
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()

            robot.set_joint_position_target(default_pos[:, full_joint_ids], joint_ids=full_joint_ids)

        # ---------------------------------------------------------
        # Compute quantities needed for visualization (WORLD)
        # ---------------------------------------------------------
        # Base pose in WORLD
        base_w = robot.data.root_link_state_w[:, 0:7]
        base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]

        # Wrist poses in WORLD
        left_ee_w = robot.data.body_state_w[:, left_ee_id, 0:7]
        right_ee_w = robot.data.body_state_w[:, right_ee_id, 0:7]

        # Left probe tip in WORLD = left wrist o (offset along +Z by ROBOT_HEIGHT)
        left_tip_pos_w, left_tip_quat_w = combine_frame_transforms(
            left_ee_w[:, 0:3], left_ee_w[:, 3:7],
            left_tip_offset_pos, left_tip_offset_quat
        )

        # Current RIGHT tip pose in WORLD = (right wrist in WORLD) o (wrist->tip)
        right_tip_pos_w, right_tip_quat_w = combine_frame_transforms(
            right_ee_w[:, 0:3], right_ee_w[:, 3:7],
            wrist_to_tip_pos, wrist_to_tip_quat
        )

        # Tip target pose in WORLD = (base in WORLD) o (tip target in BASE)
        tip_tgt_pos_w_vis, tip_tgt_quat_w_vis = combine_frame_transforms(
            base_pos_w, base_quat_w,
            tip_tgt_pos_b, tip_tgt_quat_b
        )

        # ---------------------------------------------------------
        # Visualize frames (WORLD)
        # ---------------------------------------------------------
        if step_i % 10 == 0:
            idx = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
            vis_right_tip.visualize(right_tip_pos_w, right_tip_quat_w, marker_indices=idx)

        # Step sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_i += 1
        sim_time_acc += sim_dt


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(
        RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    )

    sim.reset()
    print("[INFO] Setup complete. Running…")
    run(sim, scene)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")
    simulation_app.close()