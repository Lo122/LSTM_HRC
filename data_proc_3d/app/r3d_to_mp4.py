"""Extract just the colour video from a Record3D .r3d capture as an .mp4.

A .r3d is a zip holding, per frame, rgbd/{i}.jpg (colour), rgbd/{i}.depth and
rgbd/{i}.conf (LZFSE-compressed), plus a "metadata" JSON. This script reads
only the JPEGs, so it needs no pyliblzfse and does not touch the depth data --
use iphone_lidar_test.py instead when you want depth.npz/intrinsics.json as
well.

FRAME RATE: taken from metadata["frameTimestamps"], NOT metadata["fps"].
Those two disagree, and the timestamps are the truthful one. On
2026-09-02--19-27-46.r3d the "fps" field reads 60 while the timestamps are a
perfectly regular 0.03334s apart (1st and 99th percentile identical) -- i.e.
30fps, and 442s of footage rather than the 221s a 60fps file would claim.
Writing the mp4 at the metadata rate makes it play at double speed and puts
every derived timestamp out by 2x, which matters because everything
downstream of this (velocity/acceleration features, the MotionBERT window,
the tracker's frame-counted thresholds) is frame-rate sensitive. Pass --fps to
override if a capture's timestamps are unusable.

Frames are written in numeric index order, not zip order: the archive lists
entries arbitrarily (rgbd/9733.jpg can precede rgbd/72.depth), so iterating
the namelist would scramble the video.

Usage:
    cd data_proc_3d/app
    uv run python r3d_to_mp4.py "G://...//2026-09-02--19-27-46.r3d"
    uv run python r3d_to_mp4.py <path.r3d> --output out.mp4 --max-frames 300
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("r3d_path", type=str, help="Path to the .r3d capture.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .mp4 (default: <capture name>.mp4 next to the .r3d).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the frame rate. By default it is derived from "
                             "metadata['frameTimestamps']; see the module docstring on "
                             "why metadata['fps'] is not trusted.")
    parser.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270),
                        help="Rotate frames clockwise, for captures recorded in a "
                             "different device orientation.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (quick check on a long capture).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output if it already exists.")
    return parser.parse_args()


def resolve_fps(metadata, override=None):
    """Return (fps, how_it_was_decided).

    Prefers the median spacing of frameTimestamps over metadata["fps"], which
    reports the capture setting rather than the rate frames were delivered at.
    """
    if override is not None:
        return float(override), "--fps override"

    timestamps = np.asarray(metadata.get("frameTimestamps", []), dtype=float)
    if timestamps.size >= 2:
        median_delta = float(np.median(np.diff(timestamps)))
        if median_delta > 1e-6:
            declared = metadata.get("fps")
            derived = 1.0 / median_delta
            note = "from frameTimestamps"
            if declared and abs(float(declared) - derived) > 0.5:
                note += f" (metadata['fps'] says {float(declared):g} -- ignored)"
            return derived, note

    fallback = float(metadata.get("fps", 30.0))
    return fallback, "from metadata['fps'] (no usable timestamps)"


def frame_indices(zip_file):
    """Colour-frame indices in numeric order."""
    indices = []
    for name in zip_file.namelist():
        if name.startswith("rgbd/") and name.endswith(".jpg"):
            stem = name[len("rgbd/"):-len(".jpg")]
            if stem.isdigit():
                indices.append(int(stem))
    return sorted(indices)


ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def r3d_to_mp4(r3d_path, output_path, fps_override=None, rotate=0, max_frames=None):
    r3d_path = Path(r3d_path)
    output_path = Path(output_path)

    with zipfile.ZipFile(r3d_path) as zip_file:
        metadata = json.loads(zip_file.read("metadata"))
        fps, fps_note = resolve_fps(metadata, fps_override)
        indices = frame_indices(zip_file)
        if not indices:
            raise ValueError(f"No rgbd/*.jpg colour frames in {r3d_path}")
        if max_frames is not None:
            indices = indices[:max_frames]

        expected = len(metadata.get("frameTimestamps", []))
        if expected and expected != len(frame_indices(zip_file)):
            print(f"  NOTE: {expected} frame timestamps but "
                  f"{len(frame_indices(zip_file))} colour frames")

        first = cv2.imdecode(
            np.frombuffer(zip_file.read(f"rgbd/{indices[0]}.jpg"), np.uint8),
            cv2.IMREAD_COLOR)
        if first is None:
            raise IOError(f"Could not decode rgbd/{indices[0]}.jpg")
        if rotate:
            first = cv2.rotate(first, ROTATIONS[rotate])
        height, width = first.shape[:2]

        print(f"{r3d_path.name}")
        print(f"  {len(indices)} frames, {width}x{height}, {fps:.3f} fps  ({fps_note})")
        print(f"  duration {len(indices) / fps:.1f}s -> {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Could not open VideoWriter for {output_path}")

        start = time.time()
        try:
            for position, index in enumerate(indices):
                # cv2.imdecode returns BGR, which is what VideoWriter wants --
                # no colour conversion (the live Record3DStream path in
                # camera_utils/iphone_connection.py does need one, since it
                # yields literal RGB arrays).
                frame = cv2.imdecode(
                    np.frombuffer(zip_file.read(f"rgbd/{index}.jpg"), np.uint8),
                    cv2.IMREAD_COLOR)
                if frame is None:
                    raise IOError(f"Could not decode rgbd/{index}.jpg")
                if rotate:
                    frame = cv2.rotate(frame, ROTATIONS[rotate])
                writer.write(frame)

                if (position + 1) % 500 == 0:
                    elapsed = time.time() - start
                    print(f"    {position + 1}/{len(indices)} "
                          f"({(position + 1) / elapsed:.0f} frames/s)")
        finally:
            writer.release()

    print(f"  done in {time.time() - start:.1f}s "
          f"({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def main():
    args = parse_args()
    r3d_path = Path(args.r3d_path)
    if not r3d_path.is_file():
        raise FileNotFoundError(r3d_path)

    output_path = Path(args.output) if args.output else r3d_path.with_suffix(".mp4")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")

    r3d_to_mp4(r3d_path, output_path, fps_override=args.fps,
               rotate=args.rotate, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
