# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Spawn SonoGym scene with Unitree G1 or H1 (no actions).
Launch Isaac Sim first.
"""

import argparse
from isaaclab.app import AppLauncher

# --- CLI
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--reset_seconds",
    type=float,
    default=5.0,
    help="Reset every X seconds of *sim time* (<=0 disables).",
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
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_inv,
    matrix_from_quat,
    quat_from_matrix,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# --- SonoGym assets & cfg
from spinal_surgery import ASSETS_DATA_DIR, PACKAGE_DIR
from spinal_surgery.assets.unitreeG1 import *
from spinal_surgery.assets.unitreeH1 import *
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer

# --- Other libs
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R
import cProfile
import nibabel as nib
import numpy as np
import os
import torch
import time as pytime
import matplotlib.pyplot as plt


# -----------------------------
# TRANSPARENCY (LAST ATTEMPT)
# -----------------------------
def apply_human_transparency_usd(
    human_prim_path: str,
    opacity: float = 0.15,
    also_try_shader_inputs: bool = True,
    verbose: bool = True,
) -> None:
    """
    Make the human transparent by targeting *prototype geometry* when the human is instanceable.

    Strategy:
      1) If Human is an instance -> operate on prim.GetPrototype() (where the meshes live).
      2) Set displayOpacity on ALL UsdGeom.Gprim found (most robust).
      3) If Looks/DefaultMaterial exists, also set enable_opacity / opacity_constant on its shader.
      4) Bind DefaultMaterial on the prototype root (optional but helps inheritance).
    """
    from pxr import Usd, UsdGeom, UsdShade, Vt
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage not available")

    opacity = float(max(0.0, min(1.0, opacity)))

    human_prim = stage.GetPrimAtPath(human_prim_path)
    if not human_prim or not human_prim.IsValid():
        raise ValueError(f"Human prim not found: {human_prim_path}")

    target_root = human_prim
    proto = None
    if human_prim.IsInstance():
        proto = human_prim.GetPrototype()
        if proto and proto.IsValid():
            target_root = proto
            if verbose:
                print(f"[INFO] Human is instanceable. Using prototype: {proto.GetPath()}")
        else:
            if verbose:
                print("[WARN] Human is instanceable but prototype invalid. Falling back to instance root.")

    # 1) displayOpacity on prototype geometry
    op_val = Vt.FloatArray([opacity])
    n_gprim = 0
    for p in Usd.PrimRange(target_root):
        gp = UsdGeom.Gprim(p)
        if gp and gp.GetPrim().IsValid():
            attr = gp.GetDisplayOpacityAttr()
            if not attr or not attr.IsValid():
                attr = gp.CreateDisplayOpacityAttr()
            attr.Set(op_val)
            n_gprim += 1

    if verbose:
        print(f"[INFO] Set displayOpacity={opacity:.2f} on {n_gprim} Gprim(s) under {target_root.GetPath()}")

    # 2) Optionally: tweak shader inputs under Looks (on the INSTANCE path, because Looks lives there)
    if also_try_shader_inputs:
        looks = stage.GetPrimAtPath(human_prim_path + "/Looks")
        if looks and looks.IsValid():
            # try DefaultMaterial first (from your dump)
            mat_path = human_prim_path + "/Looks/DefaultMaterial"
            shader_path = mat_path + "/DefaultMaterial"
            mat_prim = stage.GetPrimAtPath(mat_path)
            sh_prim = stage.GetPrimAtPath(shader_path)

            if mat_prim and mat_prim.IsValid() and sh_prim and sh_prim.IsValid():
                shader = UsdShade.Shader(sh_prim)

                def _set(name: str, value) -> bool:
                    inp = shader.GetInput(name)
                    if inp:
                        inp.Set(value)
                        return True
                    return False

                changed = False
                changed |= _set("enable_opacity", 1.0)
                changed |= _set("enable_opacity_texture", 0.0)
                changed |= _set("opacity_constant", opacity)
                _set("opacity_threshold", 0.0)
                _set("opacity_mode", 0)  # best-effort

                if verbose:
                    if changed:
                        print(f"[INFO] Tweaked shader opacity inputs on {shader_path}")
                    else:
                        print(f"[WARN] Could not tweak shader opacity inputs on {shader_path}")

                # 3) bind DefaultMaterial to prototype root (helps inheritance if bindings are missing)
                try:
                    material = UsdShade.Material(mat_prim)
                    UsdShade.MaterialBindingAPI(target_root).Bind(material)
                    if verbose:
                        print(f"[INFO] Bound {mat_path} to {target_root.GetPath()}")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Material bind failed: {e}")
            else:
                if verbose:
                    print("[WARN] DefaultMaterial prim/shader not found; skipping shader tweak.")
        else:
            if verbose:
                print("[WARN] Looks not found; skipping shader tweak.")


def wait_and_apply_transparency(
    sim: SimulationContext,
    scene: InteractiveScene,
    num_envs: int,
    opacity: float = 0.15,
    warmup_steps: int = 3,
):
    """Wait a few sim frames so referenced/instanced prims are fully realized, then apply transparency."""
    sim_dt = sim.get_physics_dt()
    # warmup frames
    for _ in range(warmup_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    for i in range(num_envs):
        apply_human_transparency_usd(f"/World/envs/env_{i}/Human", opacity=opacity, verbose=True)


# -----------------------------
# YAML + scene cfg
# -----------------------------
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

label_res = patient_cfg["label_res"]

# robot selector
robot_cfg = scene_cfg["robot"]
robot_type = robot_cfg.get("type", "g1")  # default: g1

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

# init pose
ROBOT_CFG.init_state.pos = scene_cfg[robot_type]["pos"]
q_xyzw = R.from_euler("z", scene_cfg[robot_type]["yaw"], degrees=True).as_quat()
q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
ROBOT_CFG.init_state.rot = q_wxyz

ROBOT_HEIGHT = scene_cfg[robot_type]["height"] + 0.1
ROBOT_HEIGHT_IMG = scene_cfg[robot_type]["height_img"] + 0.1 

IK_ENABLE = True


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Minimal SonoGym scene + Unitree robot (G1 or H1)."""

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


def isaac_to_scipy_quat(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def run(sim: SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    robot = scene[ROBOT_KEY]
    human = scene["human"]



    SIDE = "left" if "left" in EE_LINK_NAME else ("right" if "right" in EE_LINK_NAME else None)
    joint_pattern = rf"(torso_joint|waist_(pitch|roll|yaw)_joint|{SIDE}_(shoulder|elbow|wrist)_.*)"

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
        height=ROBOT_HEIGHT,
        height_img=ROBOT_HEIGHT_IMG,
        visualize=sim_cfg["vis_seg_map"],
    )

    # -----------------------------
    # FIXED PATIENT XZ TARGET (from YAML)
    # -----------------------------
    if "patient_xz_target" not in sim_cfg:
        raise KeyError("Missing 'patient_xz_target' in scene_cfg['sim'] (YAML).")

    _xz_tgt = sim_cfg["patient_xz_target"]
    if len(_xz_tgt) == 2:
        # [x, z] -> append angle=0
        _xz_tgt = [_xz_tgt[0], _xz_tgt[1], 0.0]
    elif len(_xz_tgt) != 3:
        raise ValueError(f"'patient_xz_target' must have length 2 or 3, got {len(_xz_tgt)}")

    fixed_xz_target = torch.tensor(_xz_tgt, device=sim.device, dtype=torch.float32).view(1, 3)
    fixed_xz_target = fixed_xz_target.expand(scene.num_envs, -1)

    ik_params = {"lambda_val": 1e-1}
    pose_diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params=ik_params,
    )
    pose_diff_ik_controller = DifferentialIKController(pose_diff_ik_cfg, scene.num_envs, device=sim.device)

    joint_names = list(robot.data.joint_names)
    default_pos = robot.data.default_joint_pos.clone()
    for name, val in ROBOT_CFG.init_state.joint_pos.items():
        idx = joint_names.index(name)
        default_pos[:, idx] = val
    zero_vel = robot.data.default_joint_vel.clone() * 0.0

    sim_dt = sim.get_physics_dt()
    step_i = 0
    clock = pytime.time()

    while simulation_app.is_running():
        if step_i % sim_cfg["episode_length"] == 0:
            robot.write_joint_state_to_sim(default_pos, zero_vel)
            robot.reset()
            scene.reset()

            # IMPORTANT: apply AFTER reset + few warmup frames (so prototypes/instances are valid)
            wait_and_apply_transparency(sim, scene, args_cli.num_envs, opacity=0.15, warmup_steps=3)

            # Hold current pose for IK
            base_w = robot.data.root_link_state_w[:, 0:7]
            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            ee_parent_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w, ee_parent_w[:, 0:3], ee_parent_w[:, 3:7]
            )

            pose_diff_ik_controller.reset()
            ik_commands_pose = torch.zeros(scene.num_envs, pose_diff_ik_controller.action_dim, device=sim.device)
            pose_diff_ik_controller.set_command(ik_commands_pose, ee_pos_b, ee_quat_b)

            print(f"[INFO] clock dt: {pytime.time() - clock:.2f}s")

        if IK_ENABLE:
            human_world_poses = human.data.root_state_w
            world_to_human_pos = human_world_poses[:, 0:3]
            world_to_human_rot = human_world_poses[:, 3:7]

            base_w = robot.data.root_link_state_w[:, 0:7]
            ee_parent_w = robot.data.body_state_w[:, US_ee_jacobi_idx, 0:7]

            R21 = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=float)

            base_pos_w, base_quat_w = base_w[:, 0:3], base_w[:, 3:7]
            wrist_pos_w, wrist_quat_w = ee_parent_w[:, 0:3], ee_parent_w[:, 3:7]

            ee_pos_w = wrist_pos_w
            ee_quat_w = wrist_quat_w

            RotMat_torch = torch.as_tensor(R21, dtype=torch.float32, device=sim.device).unsqueeze(0).expand(
                scene.num_envs, -1, -1
            )

            ee_rotmat_w = matrix_from_quat(ee_quat_w)
            us_rotmat_w = torch.bmm(ee_rotmat_w, RotMat_torch)
            ee_quat_w = quat_from_matrix(us_rotmat_w)


            # Force the surface planner command to a fixed target (x, z, angle) every episode reset
            US_slicer.current_x_z_x_angle_cmd[:] = fixed_xz_target
            US_slicer.current_x_z_x_angle_cmd = torch.clamp(
                US_slicer.current_x_z_x_angle_cmd,
                torch.tensor(US_slicer.x_z_range[0], device=sim.device, dtype=torch.float32),
                torch.tensor(US_slicer.x_z_range[1], device=sim.device, dtype=torch.float32),
            )

            # Keep a fixed target during the episode (no incremental updates)
            US_slicer.current_x_z_x_angle_cmd[:] = fixed_xz_target

            US_slicer.slice_US(world_to_human_pos, world_to_human_rot, ee_pos_w, ee_quat_w)
            if sim_cfg["vis_us"]:
                US_slicer.visualize(key="US", first_n=1)
            if sim_cfg["vis_seg_map"]:
                US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, ee_pos_w, ee_quat_w)

            world_to_ee_target_pos, world_to_ee_target_rot = US_slicer.compute_world_ee_pose_from_cmd(
                world_to_human_pos, world_to_human_rot
            )

            world_to_ee_target_xyz = torch.bmm(
                matrix_from_quat(world_to_ee_target_rot[:, 0:4]),
                RotMat_torch.transpose(1, 2),
            )
            world_to_ee_target_rot = quat_from_matrix(world_to_ee_target_xyz)

            base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
                base_pos_w, base_quat_w, world_to_ee_target_pos, world_to_ee_target_rot
            )

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pos_w, base_quat_w, ee_parent_w[:, 0:3], ee_parent_w[:, 3:7]
            )

            base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)
            pose_diff_ik_controller.set_command(base_to_ee_target_pose)

            US_jacobian = robot.root_physx_view.get_jacobians()[:, US_ee_jacobi_idx - 1, :, robot_entity_cfg.joint_ids]
            base_RotMat = matrix_from_quat(quat_inv(base_quat_w))
            US_jacobian[:, 0:3, :] = torch.bmm(base_RotMat, US_jacobian[:, 0:3, :])
            US_jacobian[:, 3:6, :] = torch.bmm(base_RotMat, US_jacobian[:, 3:6, :])

            US_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            joint_pos_des = pose_diff_ik_controller.compute(ee_pos_b, ee_quat_b, US_jacobian, US_joint_pos)

            robot.set_joint_position_target(
                joint_pos_des,
                joint_ids=torch.tensor(robot_entity_cfg.joint_ids, device=sim.device, dtype=torch.long),
            )

        if step_i % 100 == 0:
            jids = robot_entity_cfg.joint_ids
            q = robot.data.joint_pos[0, jids].detach().cpu().numpy()
            names = [robot.data.joint_names[i] for i in jids]

            # Format as "name: +0.12" (%.2f)
            msg = " | ".join([f"{n}: {v:+.2f}" for n, v in zip(names, q)])
            print(f"[JOINTS step={step_i}] {msg}")

        scene.write_data_to_sim()
        sim.step()
        step_i += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False))

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
    simulation_app.close()