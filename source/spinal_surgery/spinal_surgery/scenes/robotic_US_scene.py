# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
import nibabel as nib
import cProfile
import time
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from spinal_surgery.assets.kuka_drill import *
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR

def isaac_to_scipy_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from IsaacLab convention (w, x, y, z) to SciPy (x, y, z, w)."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/robotic_US_scene.yaml", 'r'))

# robot
robot_cfg = scene_cfg['robot']
INIT_STATE_ROBOT_US = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_cfg['joint_pos'][0],
        "lbr_joint_1": robot_cfg['joint_pos'][1],
        "lbr_joint_2": robot_cfg['joint_pos'][2],
        "lbr_joint_3": robot_cfg['joint_pos'][3], # -1.2,
        "lbr_joint_4": robot_cfg['joint_pos'][4],
        "lbr_joint_5": robot_cfg['joint_pos'][5], # 1.5,
        "lbr_joint_6": robot_cfg['joint_pos'][6],
    },
    pos = robot_cfg['pos'] # ((0.0, -0.75, 0.4))
)
# drill robot (visual only)
INIT_STATE_ROBOT_DRILL = ArticulationCfg.InitialStateCfg(
    joint_pos={
        "lbr_joint_0": robot_cfg["joint_pos"][0],
        "lbr_joint_1": robot_cfg["joint_pos"][1],
        "lbr_joint_2": robot_cfg["joint_pos"][2],
        "lbr_joint_3": robot_cfg["joint_pos"][3],
        "lbr_joint_4": robot_cfg["joint_pos"][4],
        "lbr_joint_5": robot_cfg["joint_pos"][5],
        "lbr_joint_6": robot_cfg["joint_pos"][6],
    },
    pos=(2.0, 2.0, 0.0),
)
# patient
patient_cfg = scene_cfg['patient']
quat = R.from_euler("yxz", patient_cfg['euler_yxz'], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=patient_cfg['pos'], # 0.7
    rot=((quat[3], quat[0], quat[1], quat[2]))
)

# bed
bed_cfg = scene_cfg['bed']
quat = R.from_euler("xyz", bed_cfg['euler_xyz'], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg['pos'], 
    rot=((quat[3], quat[0], quat[1], quat[2]))
)
scale_bed = bed_cfg['scale']
# use stl: Totalsegmentator_dataset_v2_subset_body_contact
human_usd_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_body_from_urdf/" + p_id for p_id in patient_cfg['id_list']
]
human_stl_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset_stl/" + p_id for p_id in patient_cfg['id_list']
]
human_raw_list = [
            f"{ASSETS_DATA_DIR}/HumanModels/selected_dataset/" + p_id for p_id in patient_cfg['id_list']
]

usd_file_list = [human_file + "/combined_wrapwrap/combined_wrapwrap.usd" for human_file in human_usd_list]
label_map_file_list = [human_file + "/combined_label_map.nii.gz" for human_file in human_stl_list]
ct_map_file_list = [human_file + "/ct.nii.gz" for human_file in human_raw_list]

label_res = patient_cfg['label_res']
scale = 1/label_res

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # kuka US
    robot_US = KUKA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_US",
        init_state=INIT_STATE_ROBOT_US
    )

    # kuka drill (visual only)
    robot_drill = KUKA_HIGH_PD_DRILL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_Drill",
        init_state=INIT_STATE_ROBOT_DRILL,
    )

    # medical bad
    medical_bad = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bed", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_no_contact/hospital_bed.usd",
            scale = (scale_bed, scale_bed, scale_bed),
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
            ), # Improves a lot of time count=8 0.014-0.013
        ),
        init_state = INIT_STATE_BED
    )


    # human: 
    human = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Human", 
        spawn=sim_utils.MultiUsdFileCfg(
        usd_path=usd_file_list,
        random_choice=False,
        scale = (label_res, label_res, label_res),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=0.001,
            max_angular_velocity=0.001,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           articulation_enabled=False,
           solver_position_iteration_count=8,
           solver_velocity_iteration_count=0,
        ),
        ),
        init_state = INIT_STATE_HUMAN,
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot_US"]
    human = scene['human']
    robot_drill = scene["robot_drill"]
    robot_entity_cfg = SceneEntityCfg("robot_US", joint_names=["lbr_joint_.*"], body_names=["lbr_link_ee"])
    robot_entity_cfg.resolve(scene)
    US_ee_jacobi_idx = robot_entity_cfg.body_ids[-1]

    # define ik controllers
    ik_params = {"lambda_val": 0.00001}
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params=ik_params)
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, scene.num_envs, device=sim.device)
    pose_diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls", ik_params=ik_params)
    pose_diff_ik_controller = DifferentialIKController(pose_diff_ik_cfg, scene.num_envs, device=sim.device)

    # construct label image slicer
    label_convert_map = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/label_conversion.yaml", 'r'))

    # construct US simulator
    us_cfg = YAML().load(open(f"{PACKAGE_DIR}/lab/sensors/cfgs/us_cfg.yaml", 'r'))
    sim_cfg = scene_cfg['sim']
    US_slicer = USSlicer(
        us_cfg,
        label_map_list, 
        ct_map_list,
        sim_cfg['if_use_ct'],
        human_stl_list,
        scene.num_envs, 
        sim_cfg['patient_xz_range'], 
        sim_cfg['patient_xz_init'], 
        sim.device, 
        label_convert_map,
        us_cfg['image_size'], 
        us_cfg['resolution'],
        visualize=sim_cfg['vis_seg_map'],
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # --- Error logging (for plotting) ---
    pos_err_hist = []   # position error norm [m]
    ang_err_hist = []   # orientation error [deg]
    time_hist    = []   # simulation time [s]

    # --- Live plot setup ---
    plt.ion()  # turn on interactive mode

    fig, (ax_pos, ax_ang) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("EE tracking errors")

    # frame visualization
    frame_vis = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/frames",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),  # rimpicciolisci se troppo grande
                ),
            },
        )
    )

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % sim_cfg['episode_length'] == 0:
            # reset counter
            # reset the scene entities
            # root state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            # keep drill robot at default pose (visual only)
            drill_joint_pos = robot_drill.data.default_joint_pos.clone()
            drill_joint_vel = robot_drill.data.default_joint_vel.clone()
            robot_drill.write_joint_state_to_sim(drill_joint_pos, drill_joint_vel)
            robot_drill.reset()

            diff_ik_controller.reset()
            pose_diff_ik_controller.reset()

            # get ee pose in base frame
            US_root_pose_w = robot.data.root_state_w[:, 0:7]

            US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
            # compute frame in root frame
            US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
                US_root_pose_w[:, 0:3], US_root_pose_w[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
            )

            ik_commands = torch.rand(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
            diff_ik_controller.set_command(ik_commands, US_ee_pos_b, US_ee_quat_b)
            ik_commands_pose = torch.zeros(scene.num_envs, pose_diff_ik_controller.action_dim, device=sim.device)
            pose_diff_ik_controller.set_command(ik_commands_pose, US_ee_pos_b, US_ee_quat_b)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        start = time.time()
        # Apply random action
        rand_x_z_angle = torch.rand((scene.num_envs, 3), device=sim.device) * 2.0 - 1.0
        rand_x_z_angle[:, 2] = (rand_x_z_angle[:, 2] / 10)
        US_slicer.update_cmd(rand_x_z_angle)

        # get human frame
        human_world_poses = human.data.root_state_w # these are already the initial poses
        # define world to human poses
        world_to_human_pos, world_to_human_rot = human_world_poses[:, 0:3], human_world_poses[:, 3:7]

        # update the view
        # get ee pose in wolrd frame
        US_ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]
        world_to_base_pose = robot.data.root_link_state_w[:, 0:7]
        US_ee_pos_b, US_ee_quat_b = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7]
        )

        marker_indices = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

        # disegna il frame sull'EE (WORLD frame)
        # frame_vis.visualize(US_ee_pose_w[:, :3], US_ee_pose_w[:, 3:], marker_indices=marker_indices)

        # update image simulation
        US_slicer.slice_US(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        if sim_cfg['vis_us']:
            US_slicer.visualize(key="US", first_n=1)
        
        # compute frame in root frame
        if sim_cfg['vis_seg_map']:
            US_slicer.update_plotter(world_to_human_pos, world_to_human_rot, US_ee_pose_w[:, 0:3], US_ee_pose_w[:, 3:7])
        world_to_ee_target_pos, world_to_ee_target_rot = US_slicer.compute_world_ee_pose_from_cmd(
            world_to_human_pos, world_to_human_rot)
        world_to_ee_target_pose = torch.cat([world_to_ee_target_pos, world_to_ee_target_rot], dim=-1)
        
        base_to_ee_target_pos, base_to_ee_target_quat = subtract_frame_transforms(
            world_to_base_pose[:, 0:3], world_to_base_pose[:, 3:7], world_to_ee_target_pos, world_to_ee_target_rot
        )
        base_to_ee_target_pose = torch.cat([base_to_ee_target_pos, base_to_ee_target_quat], dim=-1)

        # set new command
        pose_diff_ik_controller.set_command(base_to_ee_target_pose)

        # torch.cuda.synchronize()
        # # get joint position targets
        US_jacobian = robot.root_physx_view.get_jacobians()[:, US_ee_jacobi_idx-1, :, robot_entity_cfg.joint_ids]
        US_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # compute the joint commands
        joint_pos_des = pose_diff_ik_controller.compute(
            US_ee_pos_b, 
            US_ee_quat_b,
            US_jacobian, 
            US_joint_pos
        )
        
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        end = time.time()
        #print(f"Time taken for step: {end - start}")
        """
        ############# DEBUG #############
        # Commanded target pose in WORLD (from planner / US_slicer)
        target_pos_w_dbg = world_to_ee_target_pos[0].detach().cpu().numpy()
        target_quat_wxyz = world_to_ee_target_rot[0].detach().cpu().numpy()
        target_quat_xyzw = isaac_to_scipy_quat(target_quat_wxyz)
        target_quat_w_dbg = R.from_quat(target_quat_xyzw).as_euler("xyz", degrees=True)

        # Actual EE pose in WORLD (from simulation)
        ee_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[-1], 0:7]  # (N, 7)
        ee_pos_w_dbg  = ee_state_w[0, 0:3].detach().cpu().numpy()
        ee_quat_wxyz  = ee_state_w[0, 3:7].detach().cpu().numpy()
        ee_quat_xyzw  = isaac_to_scipy_quat(ee_quat_wxyz)
        ee_quat_w_dbg = R.from_quat(ee_quat_xyzw).as_euler("xyz", degrees=True)

        # --- Error in wrist/EE frame ---
        ee_pos_w_t  = ee_state_w[:, 0:3]   # (N, 3), torch
        ee_quat_w_t = ee_state_w[:, 3:7]   # (N, 4), torch

        err_pos_ee_t, err_quat_ee_t = subtract_frame_transforms(
            ee_pos_w_t, ee_quat_w_t,            # parent: current EE pose
            world_to_ee_target_pos,             # child: target pose
            world_to_ee_target_rot,
        )

        # Convert error quaternion Isaac(wxyz) -> SciPy(xyzw)
        err_quat_wxyz = err_quat_ee_t[0].detach().cpu().numpy()
        err_quat_xyzw = isaac_to_scipy_quat(err_quat_wxyz)

        pos_err_vec      = err_pos_ee_t[0].detach().cpu().numpy()           # shape (3,)
        ang_err_vec_deg  = R.from_quat(err_quat_xyzw).as_euler("xyz", degrees=True)

        pos_err_norm     = np.linalg.norm(pos_err_vec)
        ang_err_norm_deg = np.linalg.norm(ang_err_vec_deg)

        # --- Error logging (for plotting) ---
        if len(time_hist) == 0:
            t_now = 0.0
        else:
            t_now = time_hist[-1] + sim_dt

        time_hist.append(t_now)
        pos_err_hist.append(pos_err_vec)
        ang_err_hist.append(ang_err_vec_deg)

        # --- Live plot update (continuous, per-axis in EE frame) ---
        if count % 10 == 0 and len(time_hist) > 0:
            pos_arr = np.stack(pos_err_hist, axis=0)  # (T, 3)
            ang_arr = np.stack(ang_err_hist, axis=0)  # (T, 3)

            # Position error components
            ax_pos.clear()
            ax_pos.plot(time_hist, pos_arr[:, 0], label="ex (EE)")
            ax_pos.plot(time_hist, pos_arr[:, 1], label="ey (EE)")
            ax_pos.plot(time_hist, pos_arr[:, 2], label="ez (EE)")
            ax_pos.set_ylabel("Pos err [m] (EE frame)")
            ax_pos.grid(True)
            ax_pos.legend()

            # Orientation error components
            ax_ang.clear()
            ax_ang.plot(time_hist, ang_arr[:, 0], label="eroll (EE)")
            ax_ang.plot(time_hist, ang_arr[:, 1], label="epitch (EE)")
            ax_ang.plot(time_hist, ang_arr[:, 2], label="eyaw (EE)")
            ax_ang.set_xlabel("Time [s]")
            ax_ang.set_ylabel("Ang err [deg] (EE frame)")
            ax_ang.grid(True)
            ax_ang.legend()

            plt.pause(0.001)  # allow matplotlib to update the window

        if count % 60 == 0:
            print("\n[DBG] ---- EE pose (WORLD) ----")
            print(f"[CMD] target_pos_w    = {target_pos_w_dbg}")
            print(f"[CMD] target_quat_w   = {target_quat_w_dbg}")
            print(f"[SIM] ee_pos_w        = {ee_pos_w_dbg}")
            print(f"[SIM] ee_quat_w       = {ee_quat_w_dbg}")
            print(f"[ERR] pos_err_vec (EE)= {pos_err_vec} m")
            print(f"[ERR] ang_err_vec (EE)= {ang_err_vec_deg} deg")
            print(f"[ERR] ||pos_err||     = {pos_err_norm:.4f} m")
            print(f"[ERR] ||ang_err||     = {ang_err_norm_deg:.2f} deg")

        ep_len = sim_cfg["episode_length"]

        # Alla fine di ogni episodio (ultimo step)
        if count % ep_len == ep_len - 1 and len(pos_err_hist) >= ep_len:
            # Prendi solo gli ultimi ep_len passi (episodio corrente)
            pos_arr = np.stack(pos_err_hist[-ep_len:], axis=0)  # (ep_len, 3)
            ang_arr = np.stack(ang_err_hist[-ep_len:], axis=0)  # (ep_len, 3)

            # Norme per step
            pos_norms = np.linalg.norm(pos_arr, axis=1)        # (ep_len,)
            ang_norms = np.linalg.norm(ang_arr, axis=1)        # (ep_len,)

            # Medie sulle norme
            mean_pos_norm = pos_norms.mean()
            mean_ang_norm = ang_norms.mean()

            # (Opzionale) medie componente per componente
            mean_pos_vec = pos_arr.mean(axis=0)   # ex, ey, ez medi
            mean_ang_vec = ang_arr.mean(axis=0)   # eroll, epitch, eyaw medi

            print("\n[EP] ===== Mean errors over last episode =====")
            print(f"[EP] steps considered      = {ep_len}")
            print(f"[EP] mean ||pos_err||      = {mean_pos_norm:.4f} m")
            print(f"[EP] mean ||ang_err||      = {mean_ang_norm:.2f} deg")
            print(f"[EP] mean pos_err_vec (EE) = {mean_pos_vec} m")
            print(f"[EP] mean ang_err_vec (EE) = {mean_ang_vec} deg")
        """

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device) # , gravity=[0.0, 0.0, 0.0]
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    robot_scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
    scene = InteractiveScene(robot_scene_cfg)
    # load label maps
    label_map_list = []
    for label_map_file in label_map_file_list:
        label_map = nib.load(label_map_file).get_fdata()
        label_map_list.append(label_map)
    # load ct maps
    ct_map_list = []
    for ct_map_file in ct_map_file_list:
        ct_map = nib.load(ct_map_file).get_fdata()
        ct_min_max = scene_cfg['sim']['ct_range']
        ct_map = np.clip(ct_map, ct_min_max[0], ct_min_max[1])
        ct_map = (ct_map - ct_min_max[0]) / (ct_min_max[1] - ct_min_max[0]) * 255
        ct_map_list.append(ct_map)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, label_map_list, ct_map_list)


if __name__ == "__main__":
    # run the main function
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")

    # close sim app
    simulation_app.close()