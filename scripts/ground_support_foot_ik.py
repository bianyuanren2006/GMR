"""
IK-based support-foot grounding for retargeted ENCOS-Sprint running motions.

Problem: the retarget follows the human ANKLE height during the stance/support
phase. In running mocap the subject runs on the forefoot, so the human ankle
naturally DESCENDS during stance while the toes stay planted. The robot has a
flat foot (sole = ankle - 0.065), so its sole hovers above the ground and keeps
descending through the support phase instead of staying planted -> looks like
the foot is sinking.

Fix (IK re-solve): re-solve the joint angles with mink so that during each foot's
support phase its ankle is constrained to a target z that keeps the sole on the
ground (z = SOLE). The pelvis/torso keep tracking their original poses, and the
leg joints (knee/hip) adapt so the foot stays planted through the whole stance.
Swing/flight phases are left untouched. The support-ankle task gets a high
position cost (must be set at FrameTask construction time) so it dominates over
the pelvis tracking and the foot really reaches the ground.

Usage:
    python scripts/ground_support_foot_ik.py --input X.pkl --output Y.pkl
        [--sole 0.065] [--xml assets/ENCOS-Sprint/encos_sprint_v2.xml]
"""
import argparse
import pickle

import numpy as np
import mujoco as mj
import mink

DEFAULT_XML = 'assets/ENCOS-Sprint/encos_sprint_v2.xml'
SOLE = 0.065  # foot sole below ankle (0818 model flat-bottom box: -0.057 - 0.008)

BODIES = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    'left_shoulder_pitch_link', 'left_elbow_link',
    'right_shoulder_pitch_link', 'right_elbow_link',
]
# support-ankle tasks get a high position cost (must be set at construction)
ANKLE_COST = 800.0
BASE_COST = 100.0
IK_ITERS = 200


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qrot(q, v):
    qc = _qconj(q)
    t = 2.0 * np.cross(qc[:3], v)
    return v + qc[3] * t + np.cross(qc[:3], t)


def _stance_intervals(lx, n):
    """Contiguous stance frames per foot = lx (forward offset) descending from
    local max to next local min (foot moving backward relative to the body)."""
    dm = np.diff(np.sign(np.diff(lx)))
    peaks = np.where(dm < 0)[0] + 1
    troughs = np.where(dm > 0)[0] + 1
    frames = set()
    for p in peaks:
        q = troughs[troughs > p]
        e = q[0] if len(q) else n - 1
        if 4 <= e - p <= 45:
            frames.update(range(p, e + 1))
    return frames


def ground_support_foot_ik(root_pos, root_rot, dof_pos, xml_path=DEFAULT_XML, sole=SOLE):
    """Re-solve IK so each support foot stays planted (sole on ground) during stance."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot = np.asarray(root_rot, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    n = len(dof_pos)

    # Detect stance intervals from the raw motion.
    model0 = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model0)
    al = mj.mj_name2id(model0, mj.mjtObj.mjOBJ_BODY, 'left_ankle_link')
    ar = mj.mj_name2id(model0, mj.mjtObj.mjOBJ_BODY, 'right_ankle_link')
    lxL = np.zeros(n)
    lxR = np.zeros(n)
    for i in range(n):
        data.qpos[:] = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        mj.mj_forward(model0, data)
        lxL[i] = _qrot(root_rot[i], data.xpos[al] - root_pos[i])[0]
        lxR[i] = _qrot(root_rot[i], data.xpos[ar] - root_pos[i])[0]
    stL = _stance_intervals(lxL, n)
    stR = _stance_intervals(lxR, n)

    # mink IK re-solve.
    model = mj.MjModel.from_xml_path(xml_path)
    config = mink.Configuration(model)
    bid = {b: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in BODIES}
    tasks = [
        mink.FrameTask(frame_name=b, frame_type='body',
                       position_cost=ANKLE_COST if b in ('left_ankle_link', 'right_ankle_link') else BASE_COST,
                       orientation_cost=BASE_COST, lm_damping=1)
        for b in BODIES
    ]
    tmap = dict(zip(BODIES, tasks))

    out_q = np.zeros((n, model.nq))
    for i in range(n):
        q0 = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        config.data.qpos[:] = q0
        config.update()
        for b in BODIES:
            bidd = bid[b]
            pos = config.data.xpos[bidd]
            orn_wxyz = config.data.xquat[bidd]  # [w,x,y,z] scalar-first (mink)
            tgt = pos
            if i in stL and b == 'left_ankle_link':
                tgt = np.array([pos[0], pos[1], sole])
            if i in stR and b == 'right_ankle_link':
                tgt = np.array([pos[0], pos[1], sole])
            tmap[b].set_target(mink.SE3.from_rotation_and_translation(mink.SO3(orn_wxyz), tgt))
        config.data.qpos[:] = q0 if i == 0 else out_q[i - 1]
        config.update()
        for _ in range(IK_ITERS):
            vel = mink.solve_ik(config, tasks, 1 / 60, 'daqp', 5e-1, None)
            config.integrate_inplace(vel, 1 / 60)
        out_q[i] = config.q.copy()

    return {
        'root_pos': out_q[:, :3],
        'root_rot': np.array([q[3:7][[1, 2, 3, 0]] for q in out_q]),
        'dof_pos': out_q[:, 7:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sole', type=float, default=SOLE,
                        help='Foot sole offset below ankle (target: ankle_z = sole keeps sole on ground)')
    parser.add_argument('--xml', type=str, default=DEFAULT_XML)
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    out = ground_support_foot_ik(data['root_pos'], data['root_rot'], data['dof_pos'],
                                 xml_path=args.xml, sole=args.sole)
    out['fps'] = data.get('fps', 40)
    with open(args.output, 'wb') as f:
        pickle.dump(out, f)
    print(f'Support-foot grounded (IK): {args.input} -> {args.output}  ({len(out["root_pos"])} frames)')


if __name__ == '__main__':
    main()
