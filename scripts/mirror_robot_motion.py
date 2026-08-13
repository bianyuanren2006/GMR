"""
Mirror a retargeted robot-motion pkl (from bvh_to_robot.py) left-right.

The ENCOS-Sprint robot's hip kinematics are NOT exactly left-right symmetric, so
a simple sign-flip of the joint values does not produce a valid mirrored pose.
Instead we mirror the robot's WORLD-space body poses (position Y negated,
orientation X/Z negated) and re-solve IK with mink on the robot's own bodies.

Usage:
  python scripts/mirror_robot_motion.py --input x.pkl --output x_mirror.pkl [--xml ...]
"""
import argparse
import pickle

import numpy as np
import mujoco as mj
import mink


DEFAULT_XML = 'assets/ENCOS-Sprint/encos_spriint_v2.xml'
# Bodies tracked by the ENCOS IK config (order must match ik_match_table).
BODIES = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    'left_shoulder_pitch_link', 'left_elbow_link',
    'right_shoulder_pitch_link', 'right_elbow_link',
]

# For the mirrored pose, the LEFT body tracks the mirror of the ORIGINAL RIGHT
# body (and vice-versa). Map each body to the FK body whose mirrored pose it
# must track.
MIRROR_SRC = {
    'pelvis': 'pelvis',
    'left_hip_pitch_link': 'right_hip_pitch_link',
    'right_hip_pitch_link': 'left_hip_pitch_link',
    'left_knee_link': 'right_knee_link',
    'right_knee_link': 'left_knee_link',
    'left_ankle_link': 'right_ankle_link',
    'right_ankle_link': 'left_ankle_link',
    'left_shoulder_pitch_link': 'right_shoulder_pitch_link',
    'right_shoulder_pitch_link': 'left_shoulder_pitch_link',
    'left_elbow_link': 'right_elbow_link',
    'right_elbow_link': 'left_elbow_link',
}


def mirror_pose(pos, quat_xyzw):
    """Mirror a body pose about the sagittal (XZ) plane.
    Position: negate Y. Quaternion (xyzw): negate X and Z.
    """
    mpos = np.array([pos[0], -pos[1], pos[2]])
    mquat = np.array([-quat_xyzw[0], quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]])
    return mpos, mquat


def mirror_pkl(pkl_path, out_path, xml_path=DEFAULT_XML):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    root_pos = np.asarray(data['root_pos'], dtype=np.float64)
    root_rot = np.asarray(data['root_rot'], dtype=np.float64)   # xyzw
    dof_pos = np.asarray(data['dof_pos'], dtype=np.float64)

    model = mj.MjModel.from_xml_path(xml_path)
    config = mink.Configuration(model)
    body_ids = {b: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in BODIES}

    tasks = [
        mink.FrameTask(frame_name=b, frame_type='body',
                       position_cost=100.0, orientation_cost=100.0, lm_damping=1)
        for b in BODIES
    ]

    N = len(dof_pos)
    out_q = np.zeros((N, model.nq))

    # Warm-start helper: sign-flip joints and mirror root orientation.
    def initial_guess(i):
        q = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        # mirror root orientation: q wxyz -> negate x,z
        q[3:7] = np.array([q[3], -q[4], q[5], -q[6]])
        # swap left/right joint blocks, negate yaw/roll
        jp = q[7:].copy()
        n = len(jp) // 2
        L, Rr = jp[:n], jp[n:]
        neg = [0, 1, 6, 7]
        for j in neg:
            L[j], Rr[j] = -Rr[j], -L[j]
        q[7:] = np.concatenate([Rr, L])
        return q

    for i in range(N):
        # Set task targets from the mirrored FK of the ORIGINAL pose.
        q0 = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        config.data.qpos[:] = q0
        config.update()
        for t, b in zip(tasks, BODIES):
            src = MIRROR_SRC[b]
            bid = body_ids[src]
            pos = config.data.xpos[bid]
            orn_wxyz = config.data.xquat[bid]
            orn_xyzw = np.array([orn_wxyz[1], orn_wxyz[2], orn_wxyz[3], orn_wxyz[0]])
            mpos, mquat = mirror_pose(pos, orn_xyzw)
            mquat_wxyz = np.array([mquat[3], mquat[0], mquat[1], mquat[2]])
            t.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3(mquat_wxyz), mpos))

        # Warm start from the sign-flipped estimate (or previous solution).
        config.data.qpos[:] = initial_guess(i) if i == 0 else out_q[i - 1]
        config.update()
        for _ in range(200):
            vel = mink.solve_ik(config, tasks, 1 / 60, 'daqp', 5e-1, None)
            config.integrate_inplace(vel, 1 / 60)
        out_q[i] = config.q.copy()

    mirrored = {
        'fps': data['fps'],
        'root_pos': np.array([q[:3] for q in out_q]),
        'root_rot': np.array([q[3:7][[1, 2, 3, 0]] for q in out_q]),
        'dof_pos': np.array([q[7:] for q in out_q]),
    }
    with open(out_path, 'wb') as f:
        pickle.dump(mirrored, f)
    print(f'Mirrored {pkl_path} -> {out_path}  ({N} frames)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--xml', type=str, default=DEFAULT_XML)
    args = parser.parse_args()
    mirror_pkl(args.input, args.output, args.xml)
