"""
Extract a range of frames from a BVH file and save as a new BVH file.

--start/--end are 1-indexed inclusive.  --fastest N auto-selects the N-frame
window with the highest mean hip speed (drops the first raw frame as a bad
frame, same convention as load_cmu_bvh; raw positions are in cm).
"""
import argparse
import os

import numpy as np


def parse_frame_data(src_path):
    """Return (lines, data_start, frame_lines)."""
    with open(src_path, 'r') as f:
        lines = f.readlines()
    motion_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("MOTION"):
            motion_line = i
            break
    if motion_line is None:
        raise ValueError("Could not find MOTION section in BVH file")
    data_start = motion_line + 3
    frame_lines = [l for l in lines[data_start:] if l.strip()]
    return lines, data_start, frame_lines


def hip_speed(frame_lines, dt=0.025):
    """Per-frame horizontal hip speed (m/s). First raw frame dropped as bad;
    positions assumed cm -> m; travel in the X-Z plane."""
    data = np.array([[float(x) for x in l.split()] for l in frame_lines])
    hip = data[1:, :3] / 100.0  # Hips X,Y,Z
    v = np.linalg.norm(np.diff(hip[:, [0, 2]], axis=0), axis=1) / dt
    return v


def find_fastest_window(frame_lines, n, dt=0.025):
    """Return (start_1idx, end_1idx) inclusive of the n-frame window with the
    highest mean hip speed. v[i] corresponds to raw 0-based frame i+1."""
    v = hip_speed(frame_lines, dt)
    best = None
    for s in range(len(v) - n + 1):
        m = v[s:s + n].mean()
        if best is None or m > best[0]:
            best = (m, s)
    m, s = best
    seg = v[s:s + n]
    print("fastest %d-frame window: raw 0-based [%d, %d)  "
          "mean=%.2f  median=%.2f  max=%.2f m/s"
          % (n, s + 1, s + 1 + n, seg.mean(), np.median(seg), seg.max()))
    return s + 2, s + 1 + n  # convert to 1-indexed inclusive


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
    parser.add_argument("--start", type=int, default=None, help="Start frame (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End frame (inclusive)")
    parser.add_argument("--fastest", type=int, default=None,
                        help="Auto-select the N-frame fastest window (overrides --start/--end)")
    args = parser.parse_args()

    if args.fastest:
        _, _, frame_lines = parse_frame_data(args.src)
        start, end = find_fastest_window(frame_lines, args.fastest)
    else:
        if args.start is None or args.end is None:
            parser.error("--start/--end or --fastest is required")
        start, end = args.start, args.end
    extract_bvh_frames(args.src, args.dst, start, end)
