#!/usr/bin/env python
"""Fix the ENCOS-Sprint arm pose in an AMP JSON: lock shoulder roll/yaw and
elbow to the 085-approved pose while keeping shoulder PITCH varying.

ENCOS JSON frame layout (73 values):
    [0:3] root_pos, [3:7] root_rot(xyzw),
    [7:25] joint_pos (18): L_hip_yaw,L_hip_roll,L_hip_pitch,L_knee,L_ankle,
        R_hip_yaw,R_hip_roll,R_hip_pitch,R_knee,R_ankle,
        L_sh_pitch,L_sh_roll,L_sh_yaw,L_elbow,
        R_sh_pitch,R_sh_roll,R_sh_yaw,R_elbow
    [25:37] tar_toe_local(12), [37:40] lin_vel, [40:43] ang_vel,
    [43:61] joint_vel (18, same order), [61:73] toe_vel_local(12)

Fixed-arm values (from results/.../json/085runslow1_encos_motion.json):
    L_sh_roll = +0.15, R_sh_roll = -0.15   (hands slightly out)
    L_sh_yaw = 0.0,    R_sh_yaw = 0.0
    L_elbow = -0.1,    R_elbow = -0.1       (positive elbow = hand down/back,
                                             negative = hand up/bent forward)
Only L/R_sh_pitch keep varying; the fixed joints' velocities are zeroed.

Usage:
  python scripts/fix_encos_arm_pose.py --input out.json [--output out.json]
"""
import argparse
import json

import numpy as np

# joint_pos slice indices in the 73-value frame
POS = {
    "L_sh_pitch": 17, "L_sh_roll": 18, "L_sh_yaw": 19, "L_elbow": 20,
    "R_sh_pitch": 21, "R_sh_roll": 22, "R_sh_yaw": 23, "R_elbow": 24,
}
# joint_vel slice indices = POS + 36 (vel block starts at 43)
VEL = {k: v + 36 for k, v in POS.items()}

FIXED_VALUES = {
    "L_sh_roll": 0.15, "R_sh_roll": -0.15,
    "L_sh_yaw": 0.0, "R_sh_yaw": 0.0,
    "L_elbow": -0.1, "R_elbow": -0.1,
}
# fixed joints -> zero their velocities; sh_pitch velocities are kept
FIXED_VEL_ZERO = ["L_sh_roll", "L_sh_yaw", "L_elbow", "R_sh_roll", "R_sh_yaw", "R_elbow"]


def fix_arm_pose(frames):
    frames = np.asarray(frames, dtype=float)
    for name, val in FIXED_VALUES.items():
        frames[:, POS[name]] = val
    for name in FIXED_VEL_ZERO:
        frames[:, VEL[name]] = 0.0
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None,
                    help="Output path (default: in-place)")
    args = ap.parse_args()
    out = args.output or args.input

    with open(args.input) as f:
        data = json.load(f)
    frames = np.asarray(data["Frames"], dtype=float)
    fixed = fix_arm_pose(frames)

    data["Frames"] = fixed.tolist()
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print("Applied fixed-arm pose -> %s (%d frames)" % (out, len(fixed)))
    for name, i in POS.items():
        if name in FIXED_VALUES:
            print("  %-10s fixed to %+0.3f  (vel zeroed)" % (name, FIXED_VALUES[name]))
        else:
            print("  %-10s KEPT varying  (min=%.3f max=%.3f)"
                  % (name, fixed[:, i].min(), fixed[:, i].max()))


if __name__ == "__main__":
    main()
