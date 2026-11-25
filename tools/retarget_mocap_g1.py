"""Retarget the 007bao in-house mocap walk clips (Xsens MVN BVH) to Unitree G1.

Differs from the LAFAN1 batch (`retarget_bvh_g1.py`) in two ways:

1. **Whole-session solve.** GMR's IK warm-starts each frame from the previous
   solution, so solving a short clip from its own frame 0 can land in a bad
   basin and stay there for the whole clip (see `walk1_subject1_695-806`).
   Retargeting the whole session once and slicing the clips out of it gives
   every clip a physically consistent warm start.

2. **Per-clip ground alignment.** The IK trades off pelvis height (weight 100)
   against ankle height (weight 50), so the stance ankle height varies a little
   between clips. Each clip is shifted vertically so its 2nd-percentile lowest
   ankle sits at ``TARGET_STANCE_ANKLE_Z``, matching the LAFAN1-derived clips.
   The shift is a few millimetres now that the IK config offsets are right.

The source session is 60 fps; the output is 30 fps (stride 2), matching the
LAFAN1-derived clips and the mjlab/MimicKit motion format.

Note on the IK config: `bvh_mocap_to_g1.json` must use the mocap skeleton's
[-0.5, 0.5, 0.5, 0.5] rotation offsets. With identity offsets (as in
`bvh_xsens_to_g1.json`) the G1 root is pitched 90 deg and the robot ends up
lying on its side, with the IK jittering ~5 cm per frame on the root.

Usage (from the repo root, with the package importable):

    python tools/retarget_mocap_g1.py --session mocap_walk/NewSession-007bao.bvh \
        --out_dir tmp/mocap_g1
"""

import argparse
import math
import os
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO_ROOT / "mocap_walk" / "NewSession-007bao.bvh"
DEFAULT_OUT_DIR = REPO_ROOT / "tmp" / "mocap_g1"
FPS_OUT = 30

# (name, first_raw_frame_60fps, end_raw_frame_exclusive) -- the start values are
# `A - 1` because the per-clip files were cut with `MOTION[A:B]` semantics.
# `fwd_normal` starts 30 frames in: the session opens with a 3-frame IK startup
# transient that shows up as a 20 m/s root jump.
CLIPS = [
    ("007bao_fwd_normal_2_340", 31, 340),
    ("007bao_fwd_slow_1540_1920", 1539, 1919),
    ("007bao_fwd_fast_1920_2240", 1919, 2239),
    ("007bao_bwd_slow_3400_3760", 3399, 3759),
    ("007bao_bwd_fast_3760_4120", 3759, 4119),
    ("007bao_side_left_4520_4940", 4519, 4939),
    ("007bao_side_right_4940_5340", 4939, 5339),
    ("007bao_turn_left_5340_5920", 5339, 5919),
    ("007bao_turn_right_5920_6380", 5919, 6379),
]

# NOTE: no wrist-roll post-process here. The LAFAN1 clips needed a -30/+30 deg
# wrist_roll fix (the G1's wrist_roll zero pose has the back of the hand facing
# forward), but for this skeleton the wrist offsets are calibrated against the
# human's hand orientation, so the wrist joints come out near zero and the hand
# orientation is already correct.
TARGET_STANCE_ANKLE_Z = 0.050


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(DEFAULT_SESSION), help="session BVH (60 fps)")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="output pkl folder")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--src_human", default="bvh_mocap", help="IK_CONFIG_DICT key")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frames, human_height = load_bvh_file(args.session, format="mocap")
    frames30 = frames[::2]
    print(
        f"session {args.session}: {len(frames)} frames (60fps) -> {len(frames30)} (30fps), h={human_height}"
    )

    rt = GMR(
        src_human=args.src_human,
        tgt_robot=args.robot,
        actual_human_height=human_height,
        verbose=False,
    )
    q = np.array([rt.retarget(fr) for fr in frames30])
    print("retargeted", q.shape)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)
    foot_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
    ]

    for name, a, b in CLIPS:
        seg = q[a // 2 : b // 2]
        root_pos = seg[:, :3].copy()
        root_rot = seg[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw
        dof = seg[:, 7:]

        foot_z = []
        for i in range(len(seg)):
            quat = R.from_quat(root_rot[i]).as_quat(scalar_first=True)
            data.qpos[:] = np.concatenate([root_pos[i], quat, dof[i]])
            mj.mj_forward(model, data)
            foot_z.append(min(data.xpos[j, 2] for j in foot_ids))
        z_shift = TARGET_STANCE_ANKLE_Z - np.percentile(foot_z, 2)
        root_pos[:, 2] += z_shift

        frames_arr = np.concatenate(
            [root_pos, R.from_quat(root_rot).as_rotvec(), dof], axis=1
        )

        foot_z = np.array(foot_z) + z_shift
        speed = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1) * FPS_OUT
        jerk = np.linalg.norm(np.diff(root_pos, 2, axis=0), axis=1)

        yaw = R.from_quat(root_rot).as_euler("ZYX", degrees=True)[:, 0]
        disp = root_pos[-1, :2] - root_pos[0, :2]
        travel = math.degrees(math.atan2(disp[1], disp[0]))
        rel = float(np.median(((yaw - travel + 180) % 360) - 180))

        with open(os.path.join(args.out_dir, name + ".pkl"), "wb") as f:
            pickle.dump({"loop_mode": 0, "fps": FPS_OUT, "frames": frames_arr}, f)

        print(
            f"{name:30s} T={len(seg):4d} dz={z_shift:+.3f} rootZ[{root_pos[:, 2].min():.3f},{root_pos[:, 2].max():.3f}] "
            f"footZ[{foot_z.min():.3f},{foot_z.max():.3f}] facing-vs-travel={rel:7.1f} "
            f"rootd2_med={np.median(jerk):.4f} v_mean={speed.mean():.2f} v_max={speed.max():.2f}"
        )


if __name__ == "__main__":
    main()
