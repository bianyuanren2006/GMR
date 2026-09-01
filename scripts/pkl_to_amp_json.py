"""
Convert a GMR retargeted robot-motion pkl (from bvh_to_robot.py) into an AMP-style
motion JSON consumed by humanoidGym's AMPLoader (motion_loader.py).

Frame layout (73 values, matches AMPLoader / motion_loader.py):
  root_pos(3) + root_rot(4) + joint_pos(18) + tar_toe_pos_local(12)
  + lin_vel(3) + ang_vel(3) + joint_vel(18) + tar_toe_vel_local(12)

Conventions (from humanoidGym/envs/encos_sprint/encos_amp_env.py):
  - root_rot: xyzw quaternion (body -> world), same convention as the pkl.
  - joint_pos: URDF/asset DOF order (i.e. the pkl dof_pos order as-is).
  - tar_toe_pos_local: 4 end-effector bodies
      (default [left_ankle_link, right_ankle_link, left_elbow_link, right_elbow_link])
      world position relative to pelvis, rotated into the body frame
      (exactly what get_amp_observations() computes for foot_pos).
  - lin_vel / ang_vel: BODY-frame root velocities (on reset the env does
      quat_rotate(root_orn, lin_vel) to obtain world velocity).
  - tar_toe_vel_local: end-effector body-frame velocities.
  - root_pos x,y are re-centred so the motion starts at the origin
      (frame 0 -> [0, 0, z]); --root_z_offset lowers the body height.
"""
import argparse
import json
import os
import pickle
import numpy as np
import mujoco as mj


DEFAULT_XML = 'assets/ENCOS-Sprint/encos_sprint_v2.xml'
DEFAULT_ROOT_BODY = 'pelvis'
DEFAULT_EE = ['left_ankle_link', 'right_ankle_link', 'left_elbow_link', 'right_elbow_link']


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ---------------- quaternion helpers (xyzw, Hamilton) ----------------


def quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q, v):
    """Rotate vector(s) v by quaternion q (body -> world). v can be (3,) or (...,3)."""
    v = np.asarray(v, dtype=np.float64)
    qv = q[:3]
    qw = q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q (world -> body)."""
    return quat_rotate(quat_conj(q), v)


# ---------------- finite differences ----------------


def finite_diff(pos, dt):
    vel = np.zeros_like(pos)
    if len(pos) < 2:
        return vel
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def body_angular_velocity(quats_xyzw, dt):
    """Body-frame angular velocity from a quaternion trajectory (xyzw)."""
    N = len(quats_xyzw)
    w = np.zeros((N, 3))
    for i in range(N - 1):
        delta = quat_mul(quat_conj(quats_xyzw[i]), quats_xyzw[i + 1])
        w[i] = 2.0 / dt * delta[:3]
    w[-1] = w[-2]
    return w


def compute_ee_positions(model, root_pos, root_rot_xyzw, dof_pos, ee_names):
    """Forward-kinematics the end-effector body world positions via MuJoCo."""
    data = mj.MjData(model)
    ee_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in ee_names]
    N = len(dof_pos)
    ee_world = np.zeros((N, len(ee_names), 3))
    for i in range(N):
        q = np.concatenate([root_pos[i], root_rot_xyzw[i][[3, 0, 1, 2]], dof_pos[i]])
        data.qpos[:] = q
        mj.mj_forward(model, data)
        for j, eid in enumerate(ee_ids):
            ee_world[i, j] = data.xpos[eid]
    return ee_world


def ground_clamp_positions(model, root_pos, root_rot_xyzw, dof_pos, smooth_window=1):
    """Support-leg ground clamp: adjust the root height each frame so the LOWEST
    foot contact point (the support leg) sits exactly on the ground (z = 0).
    This keeps the support leg in contact with the ground, removing both floating
    and dragging. When smooth_window > 1 the per-frame offset is smoothed with a
    moving average so the body height does not jump between frames.

    Reference: the ENCOS foot contact cylinders (radius 0.01, laid horizontally
    with their length axis along X) live on the ankle bodies at ankle-local
    z = -0.058 / -0.063, so the foot bottom is geom_z - 0.01 below the cylinder
    centre (NOT the cylinder half-length 0.13).
    """
    data = mj.MjData(model)
    ankles = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n)
              for n in ('left_ankle_link', 'right_ankle_link')]
    N = len(dof_pos)
    root_pos = root_pos.copy()
    offset = np.zeros(N)
    for i in range(N):
        q = np.concatenate([root_pos[i], root_rot_xyzw[i][[3, 0, 1, 2]], dof_pos[i]])
        data.qpos[:] = q
        mj.mj_forward(model, data)
        lows = []
        for a in ankles:
            for g in range(model.body_geomadr[a], model.body_geomadr[a] + model.body_geomnum[a]):
                if model.geom_type[g] == mj.mjtGeom.mjGEOM_CYLINDER:
                    lows.append(data.geom_xpos[g][2] - 0.01)  # contact cylinder radius
        if lows:
            # Only LIFT the root when the lowest foot is BELOW the ground
            # (penetration); never LOWER it when both feet are above ground
            # (running flight phase).  The old `-min(lows)` pulled the root
            # DOWN during flight, slamming the feet into the ground on landing
            # (penetration down to the ankle).
            offset[i] = max(0.0, -min(lows))
    if smooth_window > 1:
        # moving-average smooth the offset (pad edges by repeating first/last)
        k = np.ones(smooth_window) / smooth_window
        padded = np.concatenate([np.full(smooth_window - 1, offset[0]), offset])
        offset = np.convolve(padded, k, 'valid')[:N]
    root_pos[:, 2] += offset
    return root_pos


# ENCOS-Sprint actuator/dof order (from URDF / MuJoCo):
#   [L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle,
#    R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle,
#    L_sh_pitch, L_sh_roll, L_sh_yaw, L_elbow,
#    R_sh_pitch, R_sh_roll, R_sh_yaw, R_elbow]
# i.e. NOT [left(9) | right(9)] -> the mirror must swap each L<->R partner
# explicitly. Under a left-right flip, yaw/roll joints change sign and
# pitch joints keep sign.
# destination index -> (source index of opposite-side partner, negate)
_ENCOS_JOINT_MIRROR = {
    0: (5, True), 1: (6, True), 2: (7, False), 3: (8, False), 4: (9, False),   # L leg <- R leg
    5: (0, True), 6: (1, True), 7: (2, False), 8: (3, False), 9: (4, False),   # R leg <- L leg
    10: (14, False), 11: (15, True), 12: (16, True), 13: (17, False),          # L arm <- R arm
    14: (10, False), 15: (11, True), 16: (12, True), 17: (13, False),          # R arm <- L arm
}

# 19dof: waist (idx 10) sits between the legs and arms and is self-mirrored
# (torso twist around the vertical axis keeps its sign); the arms shift by 1.
_ENCOS_JOINT_MIRROR_19 = {
    0: (5, True), 1: (6, True), 2: (7, False), 3: (8, False), 4: (9, False),  # L leg <- R leg
    5: (0, True), 6: (1, True), 7: (2, False), 8: (3, False), 9: (4, False),  # R leg <- L leg
    10: (10, False),                                                          # waist (self)
    11: (15, False), 12: (16, True), 13: (17, True), 14: (18, False),         # L arm <- R arm
    15: (11, False), 16: (12, True), 17: (13, True), 18: (14, False),         # R arm <- L arm
}


def _mirror_encos_joints(values, njoints=18):
    """Mirror an njoint-value joint_pos/joint_vel block for the ENCOS-Sprint dof order."""
    table = _ENCOS_JOINT_MIRROR_19 if njoints == 19 else _ENCOS_JOINT_MIRROR
    out = [0.0] * len(values)
    for dst, (src, negate) in table.items():
        v = values[src]
        out[dst] = -v if negate else v
    return out


def mirror_frame(frame, njoints=18, neg_idx=(0, 1)):
    """Mirror a (4*njoints+1+4*4)-value frame (left-right flip about the XZ plane).
    Layout: root_pos(3) root_rot(4) joint_pos(n) toe_pos(12) lin(3) ang(3)
            joint_vel(n) toe_vel(12)  -> total = 7 + 2*n + 12 + 6 + 12 = 37 + 2*n.
    """
    n = njoints
    root_pos = list(frame[0:3])
    root_rot = list(frame[3:7])  # xyzw
    joint_pos = list(frame[7:7 + n])
    toe_pos = np.array(frame[7 + n:19 + n]).reshape(4, 3)
    lin_vel = list(frame[19 + n:22 + n])
    ang_vel = list(frame[22 + n:25 + n])
    joint_vel = list(frame[25 + n:25 + 2 * n])
    toe_vel = np.array(frame[25 + 2 * n:37 + 2 * n]).reshape(4, 3)

    mjoint_pos = _mirror_encos_joints(joint_pos, njoints)
    mjoint_vel = _mirror_encos_joints(joint_vel, njoints)

    # end-effectors: [L_ankle, R_ankle, L_elbow, R_elbow] -> swap pairs, negate y
    mtoe_pos = toe_pos[[1, 0, 3, 2]].copy()
    mtoe_pos[:, 1] = -mtoe_pos[:, 1]
    mtoe_vel = toe_vel[[1, 0, 3, 2]].copy()
    mtoe_vel[:, 1] = -mtoe_vel[:, 1]

    mroot_pos = [root_pos[0], -root_pos[1], root_pos[2]]
    mroot_rot = [-root_rot[0], root_rot[1], -root_rot[2], root_rot[3]]
    mlin = [lin_vel[0], -lin_vel[1], lin_vel[2]]
    mang = [-ang_vel[0], ang_vel[1], -ang_vel[2]]

    return (mroot_pos + mroot_rot + mjoint_pos
            + mtoe_pos.reshape(-1).tolist() + mlin + mang
            + mjoint_vel + mtoe_vel.reshape(-1).tolist())


def pkl_to_amp_json(pkl_path, json_path, fps=30, root_z_offset=0.0,
                    ground_clamp=False, ground_smooth=1,
                    mirror=False, mirror_neg=(0, 1, 6, 7),
                    xml_path=DEFAULT_XML, ee_names=DEFAULT_EE, njoints=18):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)   # (N,3)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)   # (N,4) xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)     # (N,njoints)
    N = len(dof_pos)
    dt = 1.0 / fps
    n = dof_pos.shape[1]
    if njoints is None:
        njoints = n
    half = njoints  # per-side joint count for the mirror layout

    # FK end-effector world positions
    model = mj.MjModel.from_xml_path(xml_path)
    if ground_clamp:
        # support-leg ground clamp BEFORE FK/velocities so ee positions and
        # velocities are computed from the corrected root height
        root_pos = ground_clamp_positions(model, root_pos, root_rot, dof_pos, ground_smooth)
    ee_world = compute_ee_positions(model, root_pos, root_rot, dof_pos, ee_names)

    # Velocities (body frame)
    root_lin_world = finite_diff(root_pos, dt)
    body_lin = np.array([quat_rotate_inverse(root_rot[i], root_lin_world[i])
                         for i in range(N)])
    body_ang = body_angular_velocity(root_rot, dt)
    dof_vel = finite_diff(dof_pos, dt)
    ee_world_vel = finite_diff(ee_world.reshape(N, -1), dt).reshape(N, -1, 3)
    ee_body = np.array([quat_rotate_inverse(root_rot[i], ee_world[i] - root_pos[i])
                        for i in range(N)])
    ee_body_vel = np.array([quat_rotate_inverse(root_rot[i], ee_world_vel[i])
                            for i in range(N)])

    # Root position: centre x,y at origin, apply z offset
    root_pos_out = root_pos.copy()
    root_pos_out[:, :2] -= root_pos_out[0, :2]
    root_pos_out[:, 2] -= root_z_offset

    frames = []
    for i in range(N):
        frame = (list(root_pos_out[i]) + list(root_rot[i]) + list(dof_pos[i])
                 + ee_body[i].reshape(-1).tolist()
                 + list(body_lin[i]) + list(body_ang[i]) + list(dof_vel[i])
                 + ee_body_vel[i].reshape(-1).tolist())
        if mirror:
            frame = mirror_frame(frame, njoints, mirror_neg)
        frames.append(frame)

    output = {
        "LoopMode": "Wrap",
        "FrameDuration": round(1.0 / fps, 6),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 1.0,
        "Frames": frames,
    }
    os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else '.', exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f'Converted {pkl_path} -> {json_path}'
          + (' (mirrored)' if mirror else ''))
    print(f'  fps={fps}, frames={N}, per frame: {len(frames[0])}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--root_z_offset', type=float, default=0.0,
                        help='Amount to lower the body height (subtract from root_z), meters')
    parser.add_argument('--ground_clamp', action='store_true',
                        help='Support-leg ground clamp: adjust root height so the lowest foot '
                             '(support leg) stays on the ground every frame')
    parser.add_argument('--ground_smooth', type=int, default=1,
                        help='Moving-average window for smoothing the ground-clamp height offset '
                             '(1 = no smoothing, e.g. 3/5 removes single-frame height jumps)')
    parser.add_argument('--mirror', action='store_true', help='Mirror left/right')
    parser.add_argument('--mirror_neg', type=str, default='0,1,6,7',
                        help='Joint indices (within each side) to negate when mirroring '
                             '(e.g. hip_yaw,hip_roll,sh_roll,sh_yaw)')
    parser.add_argument('--xml', type=str, default=DEFAULT_XML,
                        help='MuJoCo XML for FK end-effector positions')
    parser.add_argument('--ee', type=str, default=','.join(DEFAULT_EE),
                        help='Comma-separated end-effector body names')
    parser.add_argument('--njoints', type=int, default=None,
                        help='Number of joints (18=default 73-val frames, 19=75-val 19dof frames). '
                             'Auto-detected from the pkl if omitted.')
    args = parser.parse_args()
    neg_idx = tuple(int(x) for x in args.mirror_neg.split(',') if x.strip())
    ee = [e.strip() for e in args.ee.split(',') if e.strip()]
    pkl_to_amp_json(args.input, args.output, args.fps,
                    root_z_offset=args.root_z_offset,
                    ground_clamp=args.ground_clamp,
                    ground_smooth=args.ground_smooth,
                    mirror=args.mirror, mirror_neg=neg_idx,
                    xml_path=args.xml, ee_names=ee, njoints=args.njoints)
