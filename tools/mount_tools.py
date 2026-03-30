# mount_tools.py

import argparse
from isaaclab.app import AppLauncher

# --- CLI
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

import os
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

from spinal_surgery import ASSETS_DATA_DIR
from spinal_surgery.assets.unitreeH1 import H12_CFG_TOOLS_BASEFIX
from spinal_surgery.assets.unitreeG1 import G1_TOOLS_BM_CFG


def bake_tools_into_robot_usd_as_links(
    robot_usd_path: str,                      # main robot usd path
    left_tool_usd: str,                       # left tool usd path
    right_tool_usd: str,                      # right tool usd path
    out_usd_path: str,                        # output robot usd path with tools as links
    parent_left_link: str = "left_wrist_yaw_link",
    parent_right_link: str = "right_wrist_yaw_link",
    left_offset_xyz=(0.0, 0.0, 0.0),
    left_rpy_deg=(0.0, 0.0, 0.0),
    right_offset_xyz=(0.0, 0.0, 0.0),
    right_rpy_deg=(0.0, 0.0, 0.0),
    tool_visual_scale=(1.0, 1.0),             # (sx, dx)
    child_left_name="tool_left_link",
    child_right_name="tool_right_link",
    joint_group="Joints",
    mass_left=0.1,
    mass_right=0.1,
    collisions_enabled=True,
    geom_local_offset=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
):
    """Mount tools as fixed links attached to wrists in a new USD."""

    # open robot USD
    stage = Usd.Stage.Open(robot_usd_path)
    root = stage.GetDefaultPrim()
    root_path = root.GetPath().pathString

    # find wrist links sx/dx
    wl = wr = None
    for p in stage.Traverse():
        n = p.GetName()
        if n == parent_left_link:
            wl = p
        if n == parent_right_link:
            wr = p

    if wl is None or wr is None:
        raise RuntimeError(
            f"Cannot find wrist links '{parent_left_link}' / '{parent_right_link}' in {robot_usd_path}"
        )

    # clean old links
    for old in (
        f"{root_path}/{child_left_name}",
        f"{root_path}/{child_right_name}",
        f"{root_path}/{joint_group}/LeftToolFixed",
        f"{root_path}/{joint_group}/RightToolFixed",
    ):
        prim = stage.GetPrimAtPath(old)
        if prim.IsValid():
            stage.RemovePrim(prim.GetPath())

    # joints root
    jroot = f"{root_path}/{joint_group}"

    # helper for child link
    def _make_child_link(child_name, tool_usd, scale, geom_off):
        link_path = f"{root_path}/{child_name}"
        geom_path = f"{link_path}/geometry"

        # link transform
        link_xf = UsdGeom.Xform.Define(stage, Sdf.Path(link_path))
        link_prim = link_xf.GetPrim()
        link_prim.SetInstanceable(False)

        # rigid + mass
        UsdPhysics.RigidBodyAPI.Apply(link_prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(link_prim).CreateMassAttr(1e-3)

        # geometry xform
        geom_xf = UsdGeom.Xform.Define(stage, Sdf.Path(geom_path))
        geom_prim = geom_xf.GetPrim()
        geom_prim.SetInstanceable(False)
        UsdGeom.Imageable(geom_xf).CreateVisibilityAttr().Set("inherited")

        # reference to tool usd
        ref_stage = Usd.Stage.Open(tool_usd)
        ref_root = ref_stage.GetDefaultPrim()
        ref_path = ref_root.GetPath() if ref_root else Sdf.Path("/")
        geom_prim.GetReferences().ClearReferences()
        geom_prim.GetReferences().AddReference(tool_usd, ref_path)

        # offset/scale
        api = UsdGeom.XformCommonAPI(geom_xf)
        api.SetTranslate(tuple(float(v) for v in geom_off))
        api.SetRotate((0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        api.SetScale((scale, scale, scale))

        # isable collisions (optional)
        if not collisions_enabled:
            root_s = str(link_path)
            for p in stage.Traverse():
                if str(p.GetPath()).startswith(root_s):
                    UsdPhysics.CollisionAPI.Apply(p).GetCollisionEnabledAttr().Set(False)

        return link_path

    sL, sR = tool_visual_scale
    left_link_path = _make_child_link(child_left_name, left_tool_usd, sL, geom_local_offset[0])
    right_link_path = _make_child_link(child_right_name, right_tool_usd, sR, geom_local_offset[1])

    # fixed joint sx/dx
    jL = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{jroot}/LeftToolFixed"))
    jR = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{jroot}/RightToolFixed"))
    jL.CreateBody0Rel().SetTargets([wl.GetPath()])
    jL.CreateBody1Rel().SetTargets([Sdf.Path(left_link_path)])
    jR.CreateBody0Rel().SetTargets([wr.GetPath()])
    jR.CreateBody1Rel().SetTargets([Sdf.Path(right_link_path)])

    # initialize pos/rot of local joints (frame wrist → frame tool)
    for joint, off, rpy in (
        (jL, left_offset_xyz, left_rpy_deg),
        (jR, right_offset_xyz, right_rpy_deg),
    ):
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))

        q_xyzw = R.from_euler("xyz", rpy, degrees=True).as_quat()  # (x,y,z,w)
        qw, qx, qy, qz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])

        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, off)))
        joint.CreateLocalRot1Attr(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))

    # masses
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(left_link_path)).CreateMassAttr(float(mass_left))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(right_link_path)).CreateMassAttr(float(mass_right))

    # saver new usd
    os.makedirs(os.path.dirname(out_usd_path), exist_ok=True)
    stage.GetRootLayer().Export(out_usd_path)
    print(f"[bake_tools_into_robot_usd_as_links] wrote: {out_usd_path}")


def main():
    # usd tools
    left_usd = f"{ASSETS_DATA_DIR}/SurgicalTools/US_probes/usd/linear_moved.usd"
    right_usd = f"{ASSETS_DATA_DIR}/SurgicalTools/drills/usd/drill_screw_moved2.usd"

    # usd robot "clean" (no tools)
    SRC_G1_CFG = H12_CFG_TOOLS_BASEFIX
    #src_robot_usd = SRC_G1_CFG.spawn.usd_path

    SRC_G1_CFG = G1_TOOLS_BM_CFG
    src_robot_usd = SRC_G1_CFG.spawn.usd_path

    # usd di output con tools montati
    out_robot_usd = f"{ASSETS_DATA_DIR}/unitree/robots/g1-toolmount/g1_29dof_tools_bm.usd"

    print(f"[INFO] Source robot USD: {src_robot_usd}")
    print(f"[INFO] Left tool USD:    {left_usd}")
    print(f"[INFO] Right tool USD:   {right_usd}")
    print(f"[INFO] Output USD:       {out_robot_usd}")

    bake_tools_into_robot_usd_as_links(
        robot_usd_path=src_robot_usd,
        left_tool_usd=left_usd,
        right_tool_usd=right_usd,
        out_usd_path=out_robot_usd,
        parent_left_link="left_wrist_yaw_link",
        parent_right_link="right_wrist_yaw_link",
        left_offset_xyz=(-0.0, 0.0, 0.0),
        left_rpy_deg=(0.0, 0.0, 0.0),
        right_offset_xyz=(-0.04, 0.0, 0.0),
        right_rpy_deg=(0.0, 0.0, 0.0),
        tool_visual_scale=(1.0, 1.0),
        geom_local_offset=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        mass_left=0.1,
        mass_right=0.1,
        collisions_enabled=False,
        child_left_name="tool_left_link",
        child_right_name="tool_right_link",
        joint_group="Joints",
    )



if __name__ == "__main__":
    main()
    simulation_app.close()