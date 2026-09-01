"""
Support-foot ground anchoring for retargeted ENCOS-Sprint running motions.

Problem: when retargeting running mocap (where the subject runs on the forefoot),
the robot faithfully follows the human ANKLE height, which naturally descends
during the stance/support phase. Because the robot has a flat foot (sole =
ankle - 0.07), its sole hovers above the ground and keeps DESCENDING through the
support phase instead of staying planted -> looks like the foot is sinking.

Fix: for each foot, detect its stance/support phase (foot sole within a threshold
of that foot's own minimum height). During the stance phase, adjust the ROOT
height so the support-foot sole sits on the ground (z ~ 0), keeping it planted
and roughly constant through the stance. Swing/flight phases are left untouched
(natural gait preserved). The per-frame root adjustment is smoothed so there are
no single-frame height jumps.

Usage:
    python scripts/ground_support_foot.py --input X.pkl --output Y.pkl
        [--thresh 0.12] [--smooth 15]
"""
import argparse
import pickle

import numpy as np
import mujoco as mj

DEFAULT_XML = 'assets/ENCOS-Sprint/encos_sprint_v2.xml'
SOLE_OFFSET = 0.0703  # foot sole below ankle (contact cylinder bottom)


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qrot(q, v):
    qc = _qconj(q)
    t = 2.0 * np.cross(qc[:3], v)
    return v + qc[3] * t + np.cross(qc[:3], t)


def _foot_soles(model, root_pos, root_rot, dof_pos):
    """Return left/right ankle sole world-z per frame (sole = ankle_z - SOLE_OFFSET)."""
    data = mj.MjData(model)
    al = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'left_ankle_link')
    ar = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'right_ankle_link')
    n = len(dof_pos)
    sL = np.zeros(n)
    sR = np.zeros(n)
    for i in range(n):
        q = np.concatenate([root_pos[i], root_rot[i][[3, 0, 1, 2]], dof_pos[i]])
        data.qpos[:] = q
        mj.mj_forward(model, data)
        sL[i] = data.xpos[al, 2] - SOLE_OFFSET
        sR[i] = data.xpos[ar, 2] - SOLE_OFFSET
    return sL, sR


def ground_support_foot(root_pos, root_rot, dof_pos, model=None, thresh=0.12, smooth=15):
    """Adjust root height so each support foot stays planted during its stance phase."""
    if model is None:
        model = mj.MjModel.from_xml_path(DEFAULT_XML)
    sL, sR = _foot_soles(model, root_pos, root_rot, dof_pos)
    n = len(dof_pos)

    def stance_mask(sole):
        return (sole < sole.min() + thresh).astype(float)

    # per-frame root adjustment to plant the stance foot at z=0 (only downward)
    adjL = np.where(stance_mask(sL) > 0, -sL, 0.0)
    adjR = np.where(stance_mask(sR) > 0, -sR, 0.0)
    adj = np.minimum(adjL, adjR)
    adj = np.minimum(adj, 0.0)

    # smooth the adjustment so root height doesn't jump between frames
    k = np.ones(smooth) / smooth
    adj = np.convolve(adj, k, 'same')

    root_pos = root_pos.copy()
    root_pos[:, 2] += adj
    return root_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--thresh', type=float, default=0.12,
                        help='Stance detection: foot sole within this many meters of its '
                             'own minimum is considered in stance (default 0.12)')
    parser.add_argument('--smooth', type=int, default=15,
                        help='Moving-average window for the root-height adjustment '
                             '(default 15; larger = smoother but slightly less planted)')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    root_pos = np.asarray(data['root_pos'], dtype=np.float64)
    root_rot = np.asarray(data['root_rot'], dtype=np.float64)
    dof_pos = np.asarray(data['dof_pos'], dtype=np.float64)

    root_pos2 = ground_support_foot(root_pos, root_rot, dof_pos,
                                    thresh=args.thresh, smooth=args.smooth)
    data['root_pos'] = root_pos2
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print(f'Support-foot grounded: {args.input} -> {args.output}')
    print(f'  frames={len(dof_pos)}, root_z mean {root_pos2[:, 2].mean():.3f} '
          f'(raw {root_pos[:, 2].mean():.3f})')


if __name__ == '__main__':
    main()
