"""
Extract a range of frames from a BVH file and save as a new BVH file.
"""
import argparse
import os


def extract_bvh_frames(src_path, dst_path, start_frame, end_frame):
    """
    Extract frames [start_frame, end_frame] (1-indexed, inclusive)
    from a BVH file and write to dst_path.
    """
    with open(src_path, 'r') as f:
        lines = f.readlines()

    # Find the MOTION section
    motion_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("MOTION"):
            motion_line = i
            break

    if motion_line is None:
        raise ValueError("Could not find MOTION section in BVH file")

    # Parse original frame count
    frames_line = motion_line + 1
    frame_time_line = motion_line + 2

    # Data starts at motion_line + 3
    data_start = motion_line + 3

    # Extract header (everything up to end of hierarchy + MOTION header)
    header = lines[:data_start]

    # Extract the requested frame range (1-indexed)
    src_total_frames = len(lines) - data_start
    frame_data = lines[data_start:data_start + src_total_frames]

    # Validate frame range
    if start_frame < 1 or end_frame > len(frame_data):
        raise ValueError(
            f"Frame range [{start_frame}, {end_frame}] out of bounds "
            f"(total frames: {len(frame_data)})"
        )

    selected = frame_data[start_frame - 1 : end_frame]  # inclusive
    num_frames = len(selected)

    # Update frames count in the header
    # Replace "Frames: XXX" with new count
    header[frames_line] = f"Frames: {num_frames}\n"

    # Create output dir
    os.makedirs(os.path.dirname(dst_path) if os.path.dirname(dst_path) else '.', exist_ok=True)

    # Write output
    with open(dst_path, 'w') as f:
        f.writelines(header)
        f.writelines(selected)

    print(f"Extracted frames {start_frame}-{end_frame} ({num_frames} frames)")
    print(f"  From: {src_path}")
    print(f"  To:   {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frame range from BVH file")
    parser.add_argument("--src", required=True, help="Source BVH file")
    parser.add_argument("--dst", required=True, help="Destination BVH file")
    parser.add_argument("--start", type=int, required=True, help="Start frame (1-indexed)")
    parser.add_argument("--end", type=int, required=True, help="End frame (inclusive)")
    args = parser.parse_args()

    extract_bvh_frames(args.src, args.dst, args.start, args.end)
