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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import omni.usd
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

##
# Pre-defined configs
##
from spinal_surgery.assets.kuka_US import *
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation as R
from spinal_surgery.lab.kinematics.human_frame_viewer import HumanFrameViewer
from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.US_slicer import USSlicer
from ruamel.yaml import YAML
from spinal_surgery import PACKAGE_DIR

def _ensure_corner_markers(stage, num_envs: int, radius: float = 0.002):
    """Create 4 colored corner spheres per env (if not already present)."""
    # NOTE: Comments in English as requested.
    corner_colors = [
        (1.0, 0.2, 0.2),  # red
        (1.0, 0.2, 0.2),  # red
        (1.0, 0.2, 0.2),  # red
        (1.0, 0.2, 0.2),  # red
    ]

    spheres = [[None] * 4 for _ in range(num_envs)]
    for env_id in range(num_envs):
        env_path = f"/World/envs/env_{env_id}"
        for k in range(4):
            prim_path = f"{env_path}/debug/corner_{k}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                sphere = UsdGeom.Sphere.Define(stage, prim_path)
                sphere.GetRadiusAttr().Set(radius)

                # set display color
                color_attr = sphere.GetDisplayColorAttr()
                color_attr.Set([Gf.Vec3f(*corner_colors[k])])
            else:
                sphere = UsdGeom.Sphere(stage.GetPrimAtPath(prim_path))
            spheres[env_id][k] = sphere
    return spheres


def _set_sphere_world_pos(sphere: UsdGeom.Sphere, p_w):
    """Set sphere translation (world) using xformOps."""
    # NOTE: Comments in English as requested.
    xformable = UsdGeom.Xformable(sphere.GetPrim())
    ops = xformable.GetOrderedXformOps()
    if len(ops) == 0:
        op = xformable.AddTranslateOp()
    else:
        op = ops[0]
    op.Set(Gf.Vec3d(float(p_w[0]), float(p_w[1]), float(p_w[2])))


scene_cfg = YAML().load(open(f"{PACKAGE_DIR}/scenes/cfgs/unitree_scene.yaml", 'r'))

# patient
patient_cfg = scene_cfg["patient"]
quat = R.from_euler("yxz", patient_cfg["euler_yxz"], degrees=True).as_quat()
INIT_STATE_HUMAN = RigidObjectCfg.InitialStateCfg(
    pos=(
        float(patient_cfg["pos"][0]),
        float(patient_cfg["pos"][1]),
        float(patient_cfg["pos"][2])+0.1,
    ),  # 0.7
    rot=(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])),
)

# bed
bed_cfg = scene_cfg['bed']
quat = R.from_euler("xyz", bed_cfg['euler_xyz'], degrees=True).as_quat()
INIT_STATE_BED = AssetBaseCfg.InitialStateCfg(
    pos=bed_cfg['pos'], 
    rot=(0.5, 0.5, 0.5, 0.5),
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


    # medical bad
    medical_bed = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bed", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/usd_colored/hospital_bed.usd",
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



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, label_map_list: list, ct_map_list: list = None):
    """Runs the simulation loop."""

    human = scene['human']

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

    # -------------------------------------------------------------------------
    # Corner visualization of the operational (x,z) range on the patient's back
    # -------------------------------------------------------------------------
    stage = omni.usd.get_context().get_stage()
    corner_spheres = _ensure_corner_markers(stage, scene.num_envs, radius=0.005)
    # target marker (one sphere per env)
    target_spheres = []
    for env_id in range(scene.num_envs):
        prim_path = f"/World/envs/env_{env_id}/debug/target"
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.GetRadiusAttr().Set(0.007)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 1.0)])  # magenta
        target_spheres.append(sphere)

    target_cmd = torch.tensor([128.0, 140.0, 3.14], device=sim.device, dtype=torch.float32)

    # Precompute the 4 corner cmd-poses (x, z, angle). Angle chosen as mid-range.
    x_min = US_slicer.x_z_range[0, 0].item()
    z_min = US_slicer.x_z_range[0, 1].item()
    a_min = US_slicer.x_z_range[0, 2].item()

    x_max = US_slicer.x_z_range[1, 0].item()
    z_max = US_slicer.x_z_range[1, 1].item()
    a_max = US_slicer.x_z_range[1, 2].item()

    angle_mid = 0.5 * (a_min + a_max)

    # corners in (x,z): (min,min), (min,max), (max,min), (max,max)
    corners_xz = torch.tensor(
        [[x_min, z_min],
         [x_min, z_max],
         [x_max, z_min],
         [x_max, z_max]],
        device=sim.device,
        dtype=torch.float32,
    )  # (4,2)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % sim_cfg['episode_length'] == 0:      # initialize
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

            # --- update corner markers at each reset (patient is static, but safe and explicit) ---
            # Read human root pose (world->human) from the simulated rigid object
            # NOTE: Exact field names can vary across IsaacLab versions; these two are the most common.
            if hasattr(human.data, "root_pos_w"):
                world_to_human_pos = human.data.root_pos_w.clone()
                world_to_human_quat = human.data.root_quat_w.clone()
            else:
                # fallback: root_state_w = (pos(3), quat(4), linvel(3), angvel(3))
                root_state = human.data.root_state_w
                world_to_human_pos = root_state[:, 0:3].clone()
                world_to_human_quat = root_state[:, 3:7].clone()

            # Build per-env corner cmd poses: (num_envs,4,3)
            angles = torch.full((scene.num_envs, 4, 1), angle_mid, device=sim.device, dtype=torch.float32)
            corners = corners_xz.view(1, 4, 2).repeat(scene.num_envs, 1, 1)
            corner_cmd = torch.cat([corners, angles], dim=-1)  # (N,4,3)

            # Compute world EE pose for each corner without permanently changing slicer state
            saved_cmd = US_slicer.current_x_z_x_angle_cmd.clone()
            with torch.no_grad():
                for k in range(4):
                    US_slicer.current_x_z_x_angle_cmd[:] = corner_cmd[:, k, :]
                    p_w, q_w = US_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_quat)

                    # Place one sphere per env at this corner
                    for env_id in range(scene.num_envs):
                        p_w[env_id, 2] -= 0.11  # shift down by 13 cm along world z
                        _set_sphere_world_pos(corner_spheres[env_id][k], p_w[env_id])
                
                # --- place target sphere ---
                US_slicer.current_x_z_x_angle_cmd[:] = target_cmd.view(1, 3).repeat(scene.num_envs, 1)
                p_t, _ = US_slicer.compute_world_ee_pose_from_cmd(world_to_human_pos, world_to_human_quat)
                for env_id in range(scene.num_envs):
                    p_t[env_id, 2] -= 0.11  # same vertical shift as corners
                    _set_sphere_world_pos(target_spheres[env_id], p_t[env_id])

            US_slicer.current_x_z_x_angle_cmd[:] = saved_cmd

        start = time.time()

        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)

        end = time.time()
        print(f"Time taken for step: {end - start}")


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