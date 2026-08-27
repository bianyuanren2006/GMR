"""
CMU-style BVH loader for the GMR retargeting pipeline.

The CMU-style skeleton (used by `mocap-0710run/`) has its own joint-frame
convention (legs along -Y in the local frame, arms along +/-X, forward +Z),
which is *different* from the LAFAN1 skeleton (legs/spine along +X).  Because of
this the LAFAN1-calibrated rot_offsets in `bvh_lafan1_to_*.json` do not transfer
to the CMU skeleton.

This loader therefore keeps the native CMU bone names and, instead of relying on
per-skeleton rotation offsets, provides foot bodies (`LeftFootMod` /
`RightFootMod`) whose orientation is **derived from the foot geometry** (the
horizontal ankle->toe direction and world-up).  This makes the foot target
skeleton-independent and keeps the robot's foot flat (no "toes pointing down").

Returned frame format (matches lafan1.load_bvh_file):
    { "Hips": [pos(3,), quat_wxyz(4,)], ..., "LeftFootMod": [...], ... }
All positions are in meters in a Z-up world frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
import general_motion_retargeting.utils.lafan_vendor.utils as utils

# Y-up (mocap) -> Z-up (world) conversion, identical to lafan1.py
_ROTATION_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
_ROTATION_QUAT = R.from_matrix(_ROTATION_MATRIX).as_quat(scalar_first=True)

# Cap on the Hips (root) forward lean reproduced on the robot, in degrees.
# The human bends up to ~70-80 deg during the block start, but the ENCOS robot
# has a rigid torso and limited leg range: reproducing that full lean makes the
# pelvis sink and the feet penetrate the ground (the pose looks like the robot
# is sitting on the ground).  Clamping the lean keeps the crouch reachable so
# the robot looks like a forward-leaning sprint start instead.
MAX_HIPS_LEAN_DEG = 45.0

# Minimum lateral distance each hand is kept from the body centerline, in
# meters.  The CMU runner's arm swing carries the wrists across the midline
# (up to ~-0.15 m), which makes the robot look like its hands "cross the
# body".  We clamp the wrist so it stays at least this far outside the body.
MIN_HAND_LAT = 0.10

# Lateral widening factor applied to the FOOT positions, relative to the body
# centerline.  The ENCOS robot's hips are wider than the mocap subject's
# (robot hip-yaw offset ~0.11 m vs the subject's ~0.085 m).  Scaling each foot
# outward by this factor maps the subject's stance onto the robot's hip width
# (0.085 * 1.30 ~= 0.11), so the feet land under the robot's own hips instead
# of being pinched inward -> avoids "打腿" / knock-kneed stance while keeping
# the natural stride shape.
FOOT_LAT_SCALE = 1.30

# Minimum lateral distance each FOOT (ankle) is kept from the body centerline,
# in meters.  Safety floor on top of FOOT_LAT_SCALE: a human runner swings the
# trailing foot close to (or across) the midline, which after scaling could
# still dip too close on the narrower robot.  Keeps the swing-over gap safe.
MIN_FOOT_LAT = 0.07


def _lateral_knee_quat(hip_pos, knee_pos, ankle_pos, lateral_ref):
    """World-frame (scalar-first) quaternion for a knee whose flexion axis is
    lateral (perpendicular to the leg plane, pointing outward).

    The ENCOS robot knee body frame is: +Y = flexion axis, +Z up the thigh
    (toward the hip), +X completes the frame.  We map:
        body +Y -> normal of the hip-knee-ankle plane, oriented along
                   `lateral_ref` (the outward direction),
        body +Z -> knee -> hip direction (up the thigh).
    """
    thigh = np.array(hip_pos, dtype=np.float64) - np.array(knee_pos, dtype=np.float64)
    shin = np.array(ankle_pos, dtype=np.float64) - np.array(knee_pos, dtype=np.float64)
    n = np.cross(thigh, shin)
    nn = np.linalg.norm(n)
    z = thigh / (np.linalg.norm(thigh) + 1e-9)
    if nn < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    y = n / nn
    if np.dot(y, lateral_ref) < 0.0:
        y = -y
    x = np.cross(y, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    x = x / nx
    y = np.cross(z, x)          # re-orthonormalize (Y = Z x X keeps X x Y = Z)
    rot_mat = np.stack([x, y, z], axis=1)
    return R.from_matrix(rot_mat).as_quat(scalar_first=True)


def _level_foot_quat(ankle_pos, toe_pos, fallback_quat):
    """Build a world-frame (scalar-first) quaternion for a flat, forward foot.

    The robot ankle body frame is assumed to be: +X forward (toe), +Y pitch
    axis, +Z up (see encos_spriint_v2.xml).  We map:
        body +X (forward) -> horizontal(ankle->toe)
        body +Z (up)      -> world up
    """
    fwd = np.array(toe_pos, dtype=np.float64) - np.array(ankle_pos, dtype=np.float64)
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        # degenerate horizontal forward: fall back to the ankle orientation
        return np.asarray(fallback_quat, dtype=np.float64)
    fwd = fwd / n
    up = np.array([0.0, 0.0, 1.0])
    y = np.cross(up, fwd)          # body +Y
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return np.asarray(fallback_quat, dtype=np.float64)
    y = y / ny
    x = np.cross(y, up)            # re-orthonormalize body +X
    rot_mat = np.stack([x, y, up], axis=1)  # columns = body axes in world
    return R.from_matrix(rot_mat).as_quat(scalar_first=True)


def load_cmu_bvh(bvh_file, drop_first_frame=True):
    """Load a CMU-style BVH into LAFAN1-format frames (native CMU bone names).

    Returns:
        frames:        list of {bone_name: [pos(3,), quat_wxyz(4,)]}
        human_height:  robust subject height in meters (max head - lowest foot)
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # Smoothed horizontal direction of travel of the hips, used as the robot's
    # facing.  The raw CMU Hips forward (+Z body axis) does NOT align with the
    # direction of travel (it can be off by 0-30+ deg), which makes the robot
    # run "crab-style" (body turned sideways).  Facing the direction of motion
    # is the natural choice for a robot base.
    start = 1 if drop_first_frame else 0
    hip_idx = data.bones.index("Hips")
    hips_pos = global_data[1][start:, hip_idx] @ _ROTATION_MATRIX.T / 100.0  # (N,3)
    n_hip = hips_pos.shape[0]
    vel = np.zeros_like(hips_pos)
    if n_hip >= 5:
        vel[2:-2] = hips_pos[4:] - hips_pos[:-4]      # central diff over 4 frames
        vel[:2] = vel[2]; vel[-2:] = vel[-3]
    vel[:, 2] = 0.0                                   # horizontal only
    if n_hip >= 5:
        for c in (0, 1):                              # smooth velocity (5-tap)
            pad = np.concatenate([np.full(2, vel[0, c]), vel[:, c], np.full(2, vel[-1, c])])
            vel[:, c] = np.convolve(pad, np.ones(5) / 5.0, "valid")[:n_hip]
    travel_speed = np.linalg.norm(vel[:, :2], axis=1)  # m per 4 frames
    travel_dir = np.zeros_like(vel[:, :2])
    nz = travel_speed > 1e-6
    travel_dir[nz] = vel[nz, :2] / travel_speed[nz, None]

    frames = []
    for frame in range(start, data.pos.shape[0]):
        fi = frame - start
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(_ROTATION_QUAT, global_data[0][frame, i])
            position = global_data[1][frame, i] @ _ROTATION_MATRIX.T / 100.0  # cm -> m
            result[bone] = [position, orientation]

        # Rebuild the Hips (root) orientation so the robot faces the way the
        # human actually faces.  The CMU skeleton's Hips frame has a baked-in
        # ~90 deg roll (its forward is the +Z body axis), so feeding the raw
        # quaternion to the yaw-only retargeting makes the robot run sideways.
        #
        # We aim the robot's +X (forward) at the human's horizontal forward and
        # the robot's +Z (up) at the human's torso-up axis (the raw Hips +Y
        # axis after the Y-up->Z-up conversion).  Using the torso-up axis (not
        # world-up) PRESERVES the forward lean: the human bends at the waist
        # during the crouch/block start (raw Hips lean goes from ~104 deg to
        # ~171 deg), and forcing z = world-up here previously made the robot
        # unable to lean forward, so the legs got twisted / floated sideways.
        # The yaw-only retargeting (yaw_only_keep_pitch) then keeps this pitch.
        hq_mat = R.from_quat(result["Hips"][1], scalar_first=True).as_matrix()
        # Aim the robot's +X (forward) at the direction of travel when the
        # human is actually moving (>= ~1.5 m/s), otherwise fall back to the
        # raw human forward (+Z body axis projected horizontal).  Facing the
        # motion direction keeps the robot from running "crab-style".
        if travel_speed[fi] > 0.15:
            fwd = np.array([travel_dir[fi][0], travel_dir[fi][1], 0.0])
        else:
            fwd = hq_mat @ np.array([0.0, 0.0, 1.0])
            fwd[2] = 0.0
            nf = np.linalg.norm(fwd)
            if nf < 1e-6:
                fwd = np.array([1.0, 0.0, 0.0])
            else:
                fwd = fwd / nf
        if True:
            # ENCOS robot: forward = +X body axis, left side = +Y (from the
            # default-pose geometry).  Aim the robot's +X (forward) at the
            # human's forward so the robot runs forward with uncrossed legs.
            x = fwd
            z = hq_mat @ np.array([0.0, 1.0, 0.0])  # robot +Z = human torso-up
            nz = np.linalg.norm(z)
            if nz > 1e-6:
                z = z / nz
                # Clamp the forward lean to keep the crouch reachable for the
                # rigid-torso robot (otherwise the pelvis sinks and the feet
                # penetrate the ground -> looks like sitting on the ground).
                up = np.array([0.0, 0.0, 1.0])
                lean = np.arccos(np.clip(np.dot(z, up), -1.0, 1.0))
                if lean > np.radians(MAX_HIPS_LEAN_DEG):
                    axis = np.cross(z, up)
                    na = np.linalg.norm(axis)
                    if na > 1e-8:
                        axis = axis / na
                        z = R.from_rotvec(
                            axis * (lean - np.radians(MAX_HIPS_LEAN_DEG))
                        ).apply(z)
                        z = z / np.linalg.norm(z)
                y = np.cross(z, x)
                ny = np.linalg.norm(y)
                if ny > 1e-6:
                    y = y / ny
                    x = np.cross(y, z)  # re-orthonormalize
                    rot_mat = np.stack([x, y, z], axis=1)
                    result["Hips"][1] = R.from_matrix(rot_mat).as_quat(scalar_first=True)

        # Keep the hands from swinging across the body.  The CMU runner's arm
        # swing carries the wrists across the midline, which looks like the
        # robot's hands "cross the body"; clamp each wrist to stay outside a
        # minimum lateral distance from the body centerline.
        hips_mat = R.from_quat(result["Hips"][1], scalar_first=True).as_matrix()
        lat_axis = hips_mat @ np.array([0.0, 1.0, 0.0])
        lat_axis[2] = 0.0
        nla = np.linalg.norm(lat_axis)
        if nla > 1e-9:
            lat_axis = lat_axis / nla
            hip_pos = result["Hips"][0]
            for side, sgn in (("Left", 1.0), ("Right", -1.0)):
                w = result[f"{side}Wrist"]
                rel = np.array(w[0]) - hip_pos
                lat = np.dot(rel, lat_axis)
                target_lat = sgn * MIN_HAND_LAT
                if sgn * lat < MIN_HAND_LAT:
                    w[0] = hip_pos + rel + lat_axis * (target_lat - lat)

        # Skeleton-independent leg orientation (flexion axis kept lateral so the
        # robot's knees do not point inward -> avoids X-legs / knock-knees).
        # IMPORTANT: both ENCOS knee/hip body frames use +Y = robot-left (the
        # XML mirror is done via position/geometry, NOT a frame flip; both knee
        # joints are `axis="0 1 0"`), so the lateral reference is the corrected
        # Hips +Y axis for BOTH legs.  Using robot-right for the right leg made
        # the right patella point backward (shin twisted outward / knock-knee).
        hips_mat = R.from_quat(result["Hips"][1], scalar_first=True).as_matrix()
        left_world = hips_mat @ np.array([0.0, 1.0, 0.0])
        for side in ("Left", "Right"):
            # hip, knee and ankle share the thigh orientation; the derived quat
            # (Z up the thigh, Y lateral, X forward) is the same for hip & knee.
            q_leg = _lateral_knee_quat(
                result[f"{side}Hip"][0], result[f"{side}Knee"][0],
                result[f"{side}Ankle"][0], left_world,
            )
            result[f"{side}Hip"][1] = q_leg
            result[f"{side}Knee"][1] = q_leg

        # Skeleton-independent foot orientation (ankle position + derived quat).
        for side in ("Left", "Right"):
            ankle = result[f"{side}Ankle"]
            toe = result[f"{side}Toe"]
            result[f"{side}FootMod"] = [
                np.array(ankle[0]),
                _level_foot_quat(ankle[0], toe[0], ankle[1]),
            ]

        # Keep the two feet from coming too close together ("打腿"): widen the
        # whole stance to match the robot's hips (FOOT_LAT_SCALE) and then
        # clamp each ankle to a minimum lateral distance from the body
        # centerline (MIN_FOOT_LAT), in the same corrected-Hips lateral frame
        # used for the hand clamp above.
        hips_mat = R.from_quat(result["Hips"][1], scalar_first=True).as_matrix()
        lat_axis = hips_mat @ np.array([0.0, 1.0, 0.0])
        lat_axis[2] = 0.0
        nla = np.linalg.norm(lat_axis)
        if nla > 1e-9:
            lat_axis = lat_axis / nla
            hip_pos = result["Hips"][0]
            for side, sgn in (("Left", 1.0), ("Right", -1.0)):
                foot = result[f"{side}FootMod"]
                rel = np.array(foot[0]) - hip_pos
                lat = np.dot(rel, lat_axis)
                # widen the stance proportionally to the robot's hips
                new_lat = lat * FOOT_LAT_SCALE
                # then enforce the safety floor (prevent swing-over clipping)
                if sgn * new_lat < MIN_FOOT_LAT:
                    new_lat = sgn * MIN_FOOT_LAT
                if abs(new_lat - lat) > 1e-9:
                    foot[0] = hip_pos + rel + lat_axis * (new_lat - lat)
        frames.append(result)

    heights = [
        f["Head"][0][2] - min(f["LeftFootMod"][0][2], f["RightFootMod"][0][2])
        for f in frames
    ]
    human_height = float(max(heights))
    return frames, human_height


if __name__ == "__main__":
    import sys
    frames, h = load_cmu_bvh(sys.argv[1] if len(sys.argv) > 1 else "mocap-0710run/run3.bvh")
    print(f"frames={len(frames)} height={h:.3f} m")
    print("keys:", list(frames[0].keys()))
    # verify the derived foot orientation is level (pitch ~ 0) on a stance frame
    for i in [0, 10, 20]:
        lf = frames[i]["LeftFootMod"][1]
        euler = R.from_quat(lf, scalar_first=True).as_euler("ZYX", degrees=True)
        print(f"frame {i} LeftFootMod euler(ZYX) = {euler.round(1)} deg")
