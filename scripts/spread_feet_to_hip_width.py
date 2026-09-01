"""
Spread the retargeted robot's feet to the hip width via IK re-solve.

Problem: the human mocap has narrow hips (hip width ~16.6 cm) with the ankles
wider than the hips (ankles ~22.2 cm, legs splay OUTWARD). The ENCOS-Sprint
robot has much wider hips (~31.2 cm) - even wider than the human's ankles. When
the retarget tracks the human's ankle positions, the robot's ankles end up
NARROWER than its own hips (ankle ~25.6 cm < hip ~31.2 cm), so the legs splay
INWARD -> the feet look "pigeon-toed" (内撇).

Fix: re-solve the IK (mink) with each ankle's LATERAL position (pelvis-frame y)
set to the same side's HIP lateral position, so the ankle sits directly under
its hip (ankle width = hip width). The ankle task gets a high position cost so
it dominates. Everything else (pelvis, hips, knees, arms) tracks its original
pose.

Usage:
    python scripts/spread_feet_to_hip_width.py --input X.pkl --output Y.pkl
        [--xml assets/ENCOS-Sprint/encos_sprint_v2.xml]
"""
import argparse
import pickle

import numpy as np
import mujoco as mj
import mink

DEFAULT_XML = 'assets/ENCOS-Sprint/encos_sprint_v2.xml'
ANKLE_COST = 800.0
BASE_COST = 100.0
IK_ITERS = 200

BODIES = [
    'pelvis',
    'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    'left_shoulder_pitch_link', 'left_elbow_link',
    'right_shoulder_pitch_link', 'right_elbow_link',
]
BODIES_19 = BODIES + ['waist_link']


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qrot(q, v):
    """World -> body frame rotation."""
    qc = _qconj(q)
    t = 2.0 * np.cross(qc[:3], v)
    return v + qc[3] * t + np.cross(qc[:3], t)


def _qrot_inv(q, v):
    """Body -> world frame rotation."""
    t = 2.0 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t)


def spread_feet_to_hip_width(root_pos, root_rot, dof_pos, xml_path=DEFAULT_XML, dof19=False):
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot = np.asarray(root_rot, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    n = len(dof_pos)

    bodies = BODIES_19 if dof19 else BODIES

    model = mj.MjModel.from_xml_path(xml_path)
    config = mink.Configuration(model)
    bid = {b: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies}
    lhp = bid['left_hip_pitch_link']
    rhp = bid['right_hip_pitch_link']
    tasks = [
        mink.FrameTask(frame_name=b, frame_type='body',
                       position_cost=ANKLE_COST if b in ('left_ankle_link', 'right_ankle_link') else BASE_COST,
                       orientation_cost=BASE_COST, lm_damping=1)
        for b in bodies
    ]
    tmap = dict(zip(bodies, tasks))

    out_q = np.zeros((n, model.nq))
    for i in range(n):
        q0 = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        config.data.qpos[:] = q0
        config.update()
        # hip lateral position in the pelvis frame (same-side target for the ankle)
        hyL = _qrot(root_rot[i], config.data.xpos[lhp] - root_pos[i])[1]
        hyR = _qrot(root_rot[i], config.data.xpos[rhp] - root_pos[i])[1]
        for b in bodies:
            bidd = bid[b]
            pos = config.data.xpos[bidd]
            orn_wxyz = config.data.xquat[bidd]  # [w,x,y,z] scalar-first (mink)
            tgt = pos
            if b == 'left_ankle_link':
                rel = _qrot(root_rot[i], pos - root_pos[i]); rel[1] = hyL
                tgt = root_pos[i] + _qrot_inv(root_rot[i], rel)
            if b == 'right_ankle_link':
                rel = _qrot(root_rot[i], pos - root_pos[i]); rel[1] = hyR
                tgt = root_pos[i] + _qrot_inv(root_rot[i], rel)
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
    parser.add_argument('--xml', type=str, default=DEFAULT_XML)
    parser.add_argument('--dof19', action='store_true',
                        help='19dof ENCOS (adds waist_link body + waist dof).')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    out = spread_feet_to_hip_width(data['root_pos'], data['root_rot'], data['dof_pos'],
                                   xml_path=args.xml, dof19=args.dof19)
    out['fps'] = data.get('fps', 40)
    with open(args.output, 'wb') as f:
        pickle.dump(out, f)
    print(f'Spread feet to hip width: {args.input} -> {args.output}  ({len(out["root_pos"])} frames)')


if __name__ == '__main__':
    main()
