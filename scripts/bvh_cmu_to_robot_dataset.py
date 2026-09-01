"""
Batch retarget a folder of CMU-style (mocap) BVH files to a humanoid robot.

Dedicated path for the CMU skeleton (native bone names kept, no LAFAN1
renaming) using the skeleton-independent ik_config
`bvh_cmu_to_encos_sprint.json`:
  - limbs are tracked by POSITION only (no reliance on per-skeleton rotation
    offsets),
  - the foot orientation is derived from the ankle->toe geometry in the loader
    (keeps the robot's feet flat, no "toes pointing down"),
  - the root (Hips) orientation is tracked yaw-only with forward lean kept.

The first frame of each clip is dropped: these mocap files start with a
corrupted initialization frame (the subject teleports from the origin), which
otherwise injects a large spike into the retarget.

Output pkl format matches scripts/bvh_to_robot_dataset.py:
    {
        "root_pos":       (N,3)   world root translation,
        "root_rot":       (N,4)   world root quaternion (xyzw),
        "dof_pos":        (N,K)   joint positions,
        "local_body_pos": (N,B,3) body positions in a root-identity frame,
        "fps":            int,
        "link_body_list": [B],
    }
"""
import argparse
import os
import pickle

import mujoco as mj
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.cmu import load_cmu_bvh


def pick_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src_folder",
        help="Folder containing CMU-style BVH motion files to load.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../results/mocap_0710run_encos_sprint",
    )
    parser.add_argument(
        "--robot",
        default="encos_sprint_hands",
        type=str,
        help="Robot key in params.py (encos_sprint_hands adds virtual hand "
             "bodies so the human wrist can be tracked).",
    )
    parser.add_argument(
        "--fps",
        default=40,
        type=int,
        help="Motion framerate of the source BVH clips (default 40 for mocap-0710run).",
    )
    parser.add_argument(
        "--drop_first_frame",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Drop the corrupted first frame of each clip (default: drop).",
    )
    parser.add_argument(
        "--no_hip_smooth",
        default=False,
        action="store_true",
        help="Disable the Savitzky-Golay smoothing applied to the hip yaw/roll "
             "joints (keeps the raw retargeted hip values).",
    )
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    if args.robot not in IK_CONFIG_DICT.get("bvh_cmu", {}):
        raise ValueError(
            f"Robot '{args.robot}' has no 'bvh_cmu' IK config. "
            f"Available: {list(IK_CONFIG_DICT.get('bvh_cmu', {}).keys())}"
        )

    device = pick_device()
    print(f"Using device: {device}")

    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue

            bvh_file_path = os.path.join(dirpath, filename)
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(
                ".bvh", ".pkl"
            )

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue

            # Load the CMU-style BVH (native bone names + derived foot orientation)
            try:
                frames, actual_human_height = load_cmu_bvh(
                    bvh_file_path, drop_first_frame=args.drop_first_frame
                )
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue
            print(
                f"\nLoaded {bvh_file_path}: {len(frames)} frames, "
                f"human_height={actual_human_height:.3f} m"
            )

            # Initialize the retargeting system (bvh_cmu -> robot config)
            retarget = GMR(
                src_human="bvh_cmu",
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            mj.MjModel.from_xml_path(retarget.xml_file)

            # Warm-start the root at the human's starting position.  The mocap
            # clips begin far from the world origin, so starting the robot from
            # the default pose would produce a large first-frame divergence.
            q0 = retarget.configuration.q.copy()
            q0[:3] = frames[0]["Hips"][0]
            q0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation (wxyz)
            retarget.configuration.data.qpos[:] = q0
            retarget.configuration.update()

            # retarget to get all qpos
            qpos_list = []
            for curr_frame in tqdm(range(len(frames)), desc=f"Retargeting {filename}"):
                qpos = retarget.retarget(frames[curr_frame])
                qpos_list.append(qpos.copy())
            qpos_list = np.array(qpos_list)

            # Initialize the forward kinematics (for local_body_pos)
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)

            root_pos = qpos_list[:, :3]
            root_rot = qpos_list[:, 3:7]
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
            dof_pos = qpos_list[:, 7:]
            num_frames = root_pos.shape[0]

            # Smooth the hip yaw/roll joints: the under-constrained hip yaw can
            # spike by 0.2-0.6 rad in a single frame (visible "twitching" during
            # running).  Savitzky-Golay removes these spikes while preserving the
            # stride-frequency oscillation.  ENCOS dof order: [L_hip_yaw, L_hip_roll,
            # L_hip_pitch, L_knee, L_ankle, R_hip_yaw, R_hip_roll, ...].
            # Window 21 (0.5 s @ 40 fps) is long enough to kill the residual
            # 0.2-0.3 rad/frame hip-yaw spikes without blurring the stride.
            if not args.no_hip_smooth and num_frames >= 21:
                for j in (0, 1, 5, 6):  # hip yaw + hip roll, both legs
                    dof_pos[:, j] = savgol_filter(
                        dof_pos[:, j], window_length=21, polyorder=3
                    )

            # obtain local body pos (root identity frame)
            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0
            local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos,
                identity_root_rot,
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
            )
            body_names = kinematics_model.body_names

            motion_data = {
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos.detach().cpu().numpy(),
                "fps": args.fps,
                "link_body_list": body_names,
            }

            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved to {tgt_file_path}")

    print("Done. saved to ", tgt_folder)


if __name__ == "__main__":
    main()
