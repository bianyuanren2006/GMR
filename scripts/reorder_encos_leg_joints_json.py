"""
Reorder the LEG joints of the ENCOS-Sprint AMP 73-value JSON frames from
    [hip_yaw, hip_roll, hip_pitch, knee, ankle]
to
    [hip_pitch, hip_roll, hip_yaw, knee, ankle]
(per leg), leaving the arm joints unchanged. Saves copies to a target folder.

73-value frame layout:
    root_pos(3) root_rot(4) joint_pos(18) toe_pos_local(12)
    lin_vel(3) ang_vel(3) joint_vel(18) toe_vel_local(12)
    joint_pos indices 7..24, joint_vel indices 43..60

Usage:
    python scripts/reorder_encos_leg_joints_json.py \
        --src results/json --dst json_modified
"""
import argparse
import glob
import json
import os

# Current joint order (from encos_spriint_v2.xml, matches pkl dof order).
CURRENT = [
    'L_hip_yaw', 'L_hip_roll', 'L_hip_pitch', 'L_knee', 'L_ankle',
    'R_hip_yaw', 'R_hip_roll', 'R_hip_pitch', 'R_knee', 'R_ankle',
    'L_sh_pitch', 'L_sh_roll', 'L_sh_yaw', 'L_elbow',
    'R_sh_pitch', 'R_sh_roll', 'R_sh_yaw', 'R_elbow',
]
# Desired order: hip_pitch, hip_roll, hip_yaw, knee, ankle per leg; arms unchanged.
TARGET = [
    'L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle',
    'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle',
    'L_sh_pitch', 'L_sh_roll', 'L_sh_yaw', 'L_elbow',
    'R_sh_pitch', 'R_sh_roll', 'R_sh_yaw', 'R_elbow',
]

JOINT_POS_SLICE = slice(7, 25)    # joint_pos indices 7..24
JOINT_VEL_SLICE = slice(43, 61)   # joint_vel indices 43..60

PERM = [CURRENT.index(name) for name in TARGET]


def reorder_frame(frame):
    """Reorder one 73-value frame (list or array-like) in place."""
    f = list(frame)
    jp = [f[JOINT_POS_SLICE][i] for i in PERM]
    jv = [f[JOINT_VEL_SLICE][i] for i in PERM]
    f[JOINT_POS_SLICE] = jp
    f[JOINT_VEL_SLICE] = jv
    return f


def reorder_json(src_path, dst_path):
    with open(src_path, 'r') as f:
        data = json.load(f)
    frames = data['Frames']
    for k, frame in enumerate(frames):
        if len(frame) != 73:
            raise ValueError(f'{src_path} frame {k} has {len(frame)} values, expected 73')
        frames[k] = reorder_frame(frame)
    with open(dst_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  {os.path.basename(src_path)} -> {dst_path}  ({len(frames)} frames)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='results/json', help='Source JSON folder')
    parser.add_argument('--dst', default='json_modified', help='Destination folder')
    parser.add_argument('--pattern', default='*.json', help='Glob pattern within src')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, args.pattern)))
    if not files:
        raise SystemExit(f'No files matched {os.path.join(args.src, args.pattern)}')
    print(f'Reordering leg joints of {len(files)} files -> {args.dst}')
    for path in files:
        reorder_json(path, os.path.join(args.dst, os.path.basename(path)))
    print('done')


if __name__ == '__main__':
    main()
