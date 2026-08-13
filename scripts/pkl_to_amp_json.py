"""
Convert GMR pickle file to AMP-style JSON format.

Modes:
  default (no flags): [left_dof_pos(5), right_dof_pos(5), left_dof_vel(5), right_dof_vel(5)] = 20
  --gvec:             [gvec(3), body_lin_vel(3), body_ang_vel(3), root_z(1), dof_pos(10), dof_vel(10)] = 30
"""
import argparse
import json
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def compute_velocity(pos, dt):
    vel = np.zeros_like(pos)
    if len(pos) < 2:
        return vel
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def compute_angular_velocity(quats_xyzw, dt):
    """Compute angular velocity (rad/s) from quaternion time series using finite differences."""
    N = len(quats_xyzw)
    ang_vel = np.zeros((N, 3))
    if N < 2:
        return ang_vel
    for i in range(N):
        if i == 0:
            q0 = R.from_quat(quats_xyzw[0])
            q1 = R.from_quat(quats_xyzw[1])
            rel = q1 * q0.inv()
            ang_vel[i] = rel.as_rotvec() / dt
        elif i == N - 1:
            q_1 = R.from_quat(quats_xyzw[-2])
            q0 = R.from_quat(quats_xyzw[-1])
            rel = q0 * q_1.inv()
            ang_vel[i] = rel.as_rotvec() / dt
        else:
            q_1 = R.from_quat(quats_xyzw[i - 1])
            q1 = R.from_quat(quats_xyzw[i + 1])
            rel = q1 * q_1.inv()
            ang_vel[i] = rel.as_rotvec() / (2 * dt)
    return ang_vel


def _mirror_side(values, neg_idx):
    """Negate the given indices (yaw/roll joints) within one side for left-right flip."""
    out = list(values)
    for i in neg_idx:
        if i < len(out):
            out[i] = -out[i]
    return out


def mirror_frame(frame, half, neg_idx=(0, 1)):
    """Mirror basic frame: swap left/right, negate yaw/roll joints.
    frame = [lp(half), rp(half), lv(half), rv(half)]
    """
    n = half
    lp, rp = list(frame[:n]), list(frame[n:2 * n])
    lv, rv = list(frame[2 * n:3 * n]), list(frame[3 * n:4 * n])
    return (_mirror_side(rp, neg_idx) + _mirror_side(lp, neg_idx)
            + _mirror_side(rv, neg_idx) + _mirror_side(lv, neg_idx))


def mirror_frame_gvec(frame, half, neg_idx=(0, 1)):
    """Mirror extended frame: negate gvec Y, lateral vel, roll/yaw ang vel, swap dof."""
    n = half
    gvec = list(frame[:3])
    lin_vel = list(frame[3:6])
    ang_vel = list(frame[6:9])
    root_z = frame[9]  # body height, unchanged when mirroring
    lp = list(frame[10:10 + n])
    rp = list(frame[10 + n:10 + 2 * n])
    lv = list(frame[10 + 2 * n:10 + 3 * n])
    rv = list(frame[10 + 3 * n:10 + 4 * n])

    # Mirror gvec: negate Y (left-right flip)
    mgvec = [gvec[0], -gvec[1], gvec[2]]
    # Mirror linear vel: negate Y (lateral)
    mlin = [lin_vel[0], -lin_vel[1], lin_vel[2]]
    # Mirror angular vel: negate roll and yaw components
    mang = [-ang_vel[0], ang_vel[1], -ang_vel[2]]
    # Mirror dof
    mlp = _mirror_side(rp, neg_idx)
    mrp = _mirror_side(lp, neg_idx)
    mlv = _mirror_side(rv, neg_idx)
    mrv = _mirror_side(lv, neg_idx)
    return mgvec + mlin + mang + [root_z] + mlp + mrp + mlv + mrv


def pkl_to_amp_json(pkl_path, json_path, fps=50, mirror=False, use_gvec=False,
                    root_z_offset=0.0, mirror_neg=(0, 1)):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    dof_pos = data["dof_pos"]
    N = len(dof_pos)
    dt = 1.0 / fps

    # Dynamic left/right split (works for any even dof count, e.g. 10 or 18)
    half = dof_pos.shape[1] // 2
    left_pos = dof_pos[:, :half]
    right_pos = dof_pos[:, half:]
    left_vel = compute_velocity(left_pos, dt)
    right_vel = compute_velocity(right_pos, dt)

    if use_gvec:
        root_pos = data["root_pos"]
        root_rot = data["root_rot"]  # xyzw
        # Gravity vector in body frame
        body_rot = R.from_quat(root_rot)  # body -> world (xyzw)
        gvec = body_rot.inv().apply([0, 0, -1])  # (N, 3)
        # Body height above ground (optionally lowered to place feet on ground)
        root_z = root_pos[:, 2] - root_z_offset  # (N,) Z position in world
        # Linear/angular velocity in world frame, then rotate to body frame
        body_lin_vel_world = compute_velocity(root_pos, dt)
        body_ang_vel_world = compute_angular_velocity(root_rot, dt)
        body_lin_vel = body_rot.inv().apply(body_lin_vel_world)  # world -> body
        body_ang_vel = body_rot.inv().apply(body_ang_vel_world)
    else:
        gvec = body_lin_vel = body_ang_vel = root_z = None

    frames = []
    for i in range(N):
        if use_gvec:
            frame = (gvec[i].tolist() + body_lin_vel[i].tolist() + body_ang_vel[i].tolist()
                     + [root_z[i]]
                     + left_pos[i].tolist() + right_pos[i].tolist()
                     + left_vel[i].tolist() + right_vel[i].tolist())
        else:
            frame = left_pos[i].tolist() + right_pos[i].tolist() + left_vel[i].tolist() + right_vel[i].tolist()

        if mirror:
            frame = (mirror_frame_gvec(frame, half, mirror_neg) if use_gvec
                     else mirror_frame(frame, half, mirror_neg))
        frames.append(frame)

    output = {
        "LoopMode": "Wrap",
        "FrameDuration": round(1.0 / fps, 6),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "Frames": frames,
    }
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    label = " (mirrored)" if mirror else ""
    mode = "gvec+vel+dof" if use_gvec else "dof only"
    print(f'Converted {pkl_path} -> {json_path}{label}')
    print(f'  fps={fps}, frames={N}, mode={mode}, per frame: {len(frames[0])}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--mirror', action='store_true', help='Mirror left/right')
    parser.add_argument('--gvec', action='store_true',
                        help='Include gravity vector + body lin/ang vel before dof data')
    parser.add_argument('--root_z_offset', type=float, default=0.0,
                        help='Amount to lower the body height (subtract from root_z), in meters')
    parser.add_argument('--mirror_neg', type=str, default='0,1',
                        help='Comma-separated joint indices within each side to negate when mirroring '
                             '(e.g. yaw/roll). Default: 0,1 (leg hip_yaw, hip_roll)')
    args = parser.parse_args()
    neg_idx = tuple(int(x) for x in args.mirror_neg.split(',') if x.strip())
    pkl_to_amp_json(args.input, args.output, args.fps, mirror=args.mirror,
                    use_gvec=args.gvec, root_z_offset=args.root_z_offset,
                    mirror_neg=neg_idx)
