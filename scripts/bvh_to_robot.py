import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np


LEFT_RIGHT_SWAP = {
    'LeftUpLeg': 'RightUpLeg', 'RightUpLeg': 'LeftUpLeg',
    'LeftLeg': 'RightLeg', 'RightLeg': 'LeftLeg',
    'LeftFoot': 'RightFoot', 'RightFoot': 'LeftFoot',
    'LeftToe': 'RightToe', 'RightToe': 'LeftToe',
    'LeftFootMod': 'RightFootMod', 'RightFootMod': 'LeftFootMod',
    'LeftShoulder': 'RightShoulder', 'RightShoulder': 'LeftShoulder',
    'LeftArm': 'RightArm', 'RightArm': 'LeftArm',
    'LeftForeArm': 'RightForeArm', 'RightForeArm': 'LeftForeArm',
    'LeftHand': 'RightHand', 'RightHand': 'LeftHand',
}


def mirror_human_frame(frame):
    """Mirror a human frame left-right about the sagittal plane (lateral Y axis).

    Positions: negate Y. Quaternions (scalar-first, wxyz): negate X and Z
    (a reflection of a rotation is a rotation about the reflected axis by the
    negated angle -> q' = (-x, y, -z, w)). Left/right body names are swapped.
    """
    out = {}
    for name, (pos, quat) in frame.items():
        mname = LEFT_RIGHT_SWAP.get(name, name)
        mpos = np.array([pos[0], -pos[1], pos[2]], dtype=np.float64)
        mquat = np.array([quat[0], -quat[1], quat[2], -quat[3]], dtype=np.float64)
        out[mname] = [mpos, mquat]
    return out

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "tinker_sdu", "mini", "encos_sprint"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--mirror",
        action="store_true",
        default=False,
        help="Mirror the human motion left-right before retargeting.",
    )

    parser.add_argument(
        "--yaw_only_keep_pitch",
        action="store_true",
        default=False,
        help="For yaw-only bodies, zero roll (side sway) but keep pitch (forward lean).",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )
    
    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )
    if args.yaw_only_keep_pitch:
        retargeter.yaw_only_keep_pitch = True

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    
    # Warm up the IK on the first frame so the starting pose is consistent with
    # the rest of the trajectory (avoids a frame-0 discontinuity from the default
    # pose initialization).
    if len(lafan1_data_frames) > 0:
        warm = mirror_human_frame(lafan1_data_frames[0]) if args.mirror else lafan1_data_frames[0]
        retargeter.retarget(warm)

    while True:
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]
        if args.mirror:
            smplx_data = mirror_human_frame(smplx_data)

        # retarget
        qpos = retargeter.retarget(smplx_data)
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.save_path is not None:
            qpos_list.append(qpos)

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break

    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
