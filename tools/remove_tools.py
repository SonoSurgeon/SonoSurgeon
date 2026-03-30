#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove both hands (left/right) from a Unitree G1 USD.
- Copies src -> dst
- Deletes any prim subtree whose path contains '/left_hand' or '/right_hand'
- No new links/joints are added.
"""

# --- Boot Isaac Lab / Omniverse so pxr is available even when run with python ---
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True)         # no UI
_sim = _app.app                           # keep this alive until we finish

# --- Imports that depend on Kit being up ---
from pxr import Usd, Sdf

import argparse
import os

# Hard-coded paths (adjust if needed)
SRC = "/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/unitree/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"
DST = "/home/idsia/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/unitree/robots/g1-toolmount/g1_29dof_nohands_bm.usd"

def export_copy(src: str, dst: str):
    """Copy root layer to a new USD file (non-flattened)."""
    stage = Usd.Stage.Open(src)
    if stage is None:
        raise RuntimeError(f"Cannot open src USD: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    stage.GetRootLayer().Export(dst)

def print_all_joints_and_links(stage: Usd.Stage):
    """Print all joints and all link prims inside the stage."""
    print("\n========================")
    print(" LIST OF ALL LINKS")
    print("========================")
    for prim in stage.Traverse():
        if prim.GetTypeName() in ["Xform", "RigidBody", "RigidPrim"]:
            print("  LINK:", prim.GetPath())

    print("\n========================")
    print(" LIST OF ALL JOINTS")
    print("========================")
    for prim in stage.Traverse():
        if "Joint" in prim.GetTypeName():   # catches RevoluteJoint, PrismaticJoint, FixedJoint, etc.
            print("  JOINT:", prim.GetPath())


def remove_hands(stage: Usd.Stage, dry: bool = False) -> int:
    targets = []
    for prim in stage.Traverse():
        p = str(prim.GetPath()).lower()
        if "/left_hand" in p or "/right_hand" in p:
            targets.append(prim.GetPath())
    # remove deepest first
    targets.sort(key=lambda x: len(str(x)), reverse=True)

    if dry:
        print(f"[DRY] Would remove {len(targets)} prim(s):")
        for t in targets[:20]:
            print("   -", t)
        if len(targets) > 20:
            print(f"   ... (+{len(targets)-20} more)")
        return 0

    for path in targets:
        stage.RemovePrim(path)
    return len(targets)

"""
def remove_hands(stage: Usd.Stage, dry: bool = False) -> int:

    # Hand link roots we want to delete entirely (their whole subtree goes away).
    HAND_LINK_ROOTS = [
        # Left hand
        "/h1_2/L_hand_base_link",
        "/h1_2/L_index_proximal",
        "/h1_2/L_index_intermediate",
        "/h1_2/L_middle_proximal",
        "/h1_2/L_middle_intermediate",
        "/h1_2/L_pinky_proximal",
        "/h1_2/L_pinky_intermediate",
        "/h1_2/L_ring_proximal",
        "/h1_2/L_ring_intermediate",
        "/h1_2/L_thumb_proximal_base",
        "/h1_2/L_thumb_proximal",
        "/h1_2/L_thumb_intermediate",
        "/h1_2/L_thumb_distal",
        "/h1_2/left_hand_camera_base_link",

        # Right hand
        "/h1_2/R_hand_base_link",
        "/h1_2/R_index_proximal",
        "/h1_2/R_index_intermediate",
        "/h1_2/R_middle_proximal",
        "/h1_2/R_middle_intermediate",
        "/h1_2/R_pinky_proximal",
        "/h1_2/R_pinky_intermediate",
        "/h1_2/R_ring_proximal",
        "/h1_2/R_ring_intermediate",
        "/h1_2/R_thumb_proximal_base",
        "/h1_2/R_thumb_proximal",
        "/h1_2/R_thumb_intermediate",
        "/h1_2/R_thumb_distal",
        "/h1_2/right_hand_camera_base_link",
    ]

    roots_set = set(HAND_LINK_ROOTS)
    targets = []

    for prim in stage.Traverse():
        p_str = str(prim.GetPath())
        # If the prim is exactly one of the hand link roots, we remove that prim (and its subtree)
        if p_str in roots_set:
            targets.append(prim.GetPath())

    # Remove deepest first (not strictly necessary here since we are using roots, but safe)
    targets.sort(key=lambda x: len(str(x)), reverse=True)

    if dry:
        print(f"[DRY] Would remove {len(targets)} hand root prim(s):")
        for t in targets:
            print("   -", t)
        return 0

    for path in targets:
        stage.RemovePrim(path)

    return len(targets)
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="List what would be removed, don't write")
    args = ap.parse_args()

    print("[INFO] Copying USD…")
    export_copy(SRC, DST)

    print("[INFO] Opening destination:", DST)
    stage = Usd.Stage.Open(DST)
    if stage is None:
        raise RuntimeError(f"Cannot open dst USD: {DST}")

    print("\n[INFO] Printing all joints and links in model:")
    print_all_joints_and_links(stage)

    print("[INFO] Opening destination:", DST)
    stage = Usd.Stage.Open(DST)
    if stage is None:
        raise RuntimeError(f"Cannot open dst USD: {DST}")

    removed = remove_hands(stage, dry=args.dry_run)

    print("\n[INFO] Printing all joints and links in model:")
    print_all_joints_and_links(stage)

    if args.dry_run:
        print("[DRY] No changes written.")
    else:
        stage.GetRootLayer().Save()
        print(f"[OK] Removed {removed} prim(s). Saved: {DST}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Always close Kit at the very end
        _sim.close()