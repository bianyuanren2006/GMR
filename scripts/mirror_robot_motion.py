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


DEFAULT_XML = 'assets/ENCOS-Sprint/encos_sprint_v2.xml'
# Bodies tracked by the ENCOS IK config (order must match ik_match_table).
BODIES = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    'left_shoulder_pitch_link', 'left_elbow_link',
    'right_shoulder_pitch_link', 'right_elbow_link',
]
# 19dof: adds the torso/waist link (self-mirrored) which is not tracked by the
# 18dof config but must be tracked so the IK re-solve keeps the torso pose.
BODIES_19 = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    'waist_link',
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
MIRROR_SRC_19 = dict(MIRROR_SRC)
MIRROR_SRC_19['waist_link'] = 'waist_link'


def mirror_pose(pos, quat_xyzw):
    """Mirror a body pose about the sagittal (XZ) plane.
    Position: negate Y. Quaternion (xyzw): negate X and Z.
    """
    mpos = np.array([pos[0], -pos[1], pos[2]])
    mquat = np.array([-quat_xyzw[0], quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]])
    return mpos, mquat


def mirror_pkl(pkl_path, out_path, xml_path=DEFAULT_XML, dof19=False):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    root_pos = np.asarray(data['root_pos'], dtype=np.float64)
    root_rot = np.asarray(data['root_rot'], dtype=np.float64)   # xyzw
    dof_pos = np.asarray(data['dof_pos'], dtype=np.float64)

    bodies = BODIES_19 if dof19 else BODIES
    mirror_src = MIRROR_SRC_19 if dof19 else MIRROR_SRC

    model = mj.MjModel.from_xml_path(xml_path)
    config = mink.Configuration(model)
    body_ids = {b: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies}

    tasks = [
        mink.FrameTask(frame_name=b, frame_type='body',
                       position_cost=100.0, orientation_cost=100.0, lm_damping=1)
        for b in bodies
    ]

    N = len(dof_pos)
    out_q = np.zeros((N, model.nq))

    # ENCOS-Sprint actuator/dof order is NOT [left(9) | right(9)]; it is
    #   [L_leg(5): yaw,roll,pitch,knee,ankle, R_leg(5): yaw,roll,pitch,knee,ankle,
    #    L_arm(4): sh_pitch,sh_roll,sh_yaw,elbow, R_arm(4): sh_pitch,sh_roll,sh_yaw,elbow]
    # (19dof inserts `waist` between the legs and the arms).  So a left/right
    # mirror must swap each L<->R partner explicitly. yaw/roll joints change
    # sign; pitch/knee/ankle/elbow keep sign. `waist` (idx 10 in 19dof) is a
    # torso twist around the vertical axis and keeps its sign (self-mirrored).
    # dst index -> (src index of opposite-side partner, negate)
    ENCOS_JOINT_MIRROR = {
        0: (5, True), 1: (6, True), 2: (7, False), 3: (8, False), 4: (9, False),
        5: (0, True), 6: (1, True), 7: (2, False), 8: (3, False), 9: (4, False),
        10: (14, False), 11: (15, True), 12: (16, True), 13: (17, False),
        14: (10, False), 15: (11, True), 16: (12, True), 17: (13, False),
    }
    ENCOS_JOINT_MIRROR_19 = {
        0: (5, True), 1: (6, True), 2: (7, False), 3: (8, False), 4: (9, False),
        5: (0, True), 6: (1, True), 7: (2, False), 8: (3, False), 9: (4, False),
        10: (10, False),  # waist: self-mirrored, keep sign
        11: (15, False), 12: (16, True), 13: (17, True), 14: (18, False),  # L arm
        15: (11, False), 16: (12, True), 17: (13, True), 18: (14, False),  # R arm
    }
    joint_mirror = ENCOS_JOINT_MIRROR_19 if dof19 else ENCOS_JOINT_MIRROR

    def mirror_joints(jp):
        out = jp.copy()
        for dst, (src, negate) in joint_mirror.items():
            v = jp[src]
            out[dst] = -v if negate else v
        return out

    # Warm-start helper: mirror root (position + orientation) and mirror the
    # joints with the explicit L<->R partner swap.  The root POSITION must be
    # mirrored too (negate y): the target root sits ~2x the running
    # displacement away from the un-mirrored root (e.g. +17m -> -17m), and
    # with only 200 IK iterations the root cannot travel that far from an
    # un-mirrored start -> first 1-2 frames come out misplaced (root
    # teleports / pelvis drags on the ground).
    def initial_guess(i):
        q = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        q[:3] = np.array([q[0], -q[1], q[2]])   # mirror root position (negate y)
        # mirror root orientation: q wxyz -> negate x,z
        q[3:7] = np.array([q[3], -q[4], q[5], -q[6]])
        # mirror the joints (explicit ENCOS L<->R partner swap)
        q[7:] = mirror_joints(q[7:])
        return q

    for i in range(N):
        # Set task targets from the mirrored FK of the ORIGINAL pose.
        q0 = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        config.data.qpos[:] = q0
        config.update()
        for t, b in zip(tasks, bodies):
            src = mirror_src[b]
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
    parser.add_argument('--dof19', action='store_true',
                        help='19dof ENCOS (adds waist_link body + waist dof).')
    args = parser.parse_args()
    mirror_pkl(args.input, args.output, args.xml, dof19=args.dof19)
