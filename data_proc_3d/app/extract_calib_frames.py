"""Pull calibration frames out of a video file (e.g. an .mp4 recorded on the
iPhone's own Camera app) so they can be fed to calibrate_camera.py, which
takes still images rather than video.

Two modes:

  board    (default) Harvest a well-spread set of ChArUco views for
           INTRINSIC calibration -> a folder of .png files for
           `calibrate_camera.py intrinsic --images-dir`.

  marker   Pick the single best frame containing a plain ArUco marker for
           EXTRINSIC calibration -> one .png for
           `calibrate_camera.py extrinsic --method marker --image`.

WHY THIS ISN'T JUST "SAVE EVERY Nth FRAME": a 60s handheld sweep is ~1800
frames, and cv2.calibrateCamera does not benefit from being handed all of
them. What it needs is ~20-30 views that are SHARP and that DISAGREE with
each other about where the board sits. Dumping every Nth frame instead
gives you hundreds of near-duplicates biased toward whatever pose you
happened to linger on, which is slow to solve and quietly weights the fit
toward that one viewpoint. So frames are filtered three ways:

  1. Detected -- the board must actually resolve at least --min-corners
     ChArUco corners (same detector the calibration itself will use, from
     camera_utils/charuco_board.py, so a frame accepted here is a frame
     the calibration can use).

  2. Sharp -- variance-of-Laplacian above --min-sharpness. Handheld video
     is full of motion-blurred frames; blurred checkerboard corners still
     "detect" but localize badly, and a few of them bias K more than a
     dozen good views help it. The threshold is scene-dependent: run once
     and read the reported sharpness distribution, then set it.

  3. Different -- a candidate is kept only if its corners sit at least
     --min-shift px away (mean displacement over shared corner ids) from
     EVERY frame already kept. This is what actually buys pose diversity:
     translating, tilting or approaching the board all move the corners,
     so one threshold covers all three without having to reason about
     pose explicitly.

RESOLUTION IS PART OF THE CALIBRATION. K is only valid for the exact
capture mode it was solved in -- a 4K30 recording, a 1080p recording and a
DroidCam stream all have different sensor crops and therefore different K,
and there is no scale factor that converts between modes with different
aspect ratios. Record the calibration video in the SAME mode you will
record production footage in.

STABILIZATION: the stock Camera app applies electronic stabilization to
video, which warps frames non-rigidly and breaks the single-fixed-K model
this calibration assumes. Turn off enhanced/action stabilization in
Settings > Camera > Record Video, or record with an app that can disable
it. Optical (OIS) is much less harmful than electronic and is usually
tolerable.

Usage:
    cd data_proc_3d/app

    # intrinsics: harvest board views, then calibrate
    uv run python extract_calib_frames.py board_sweep.mp4 -o frames/board `
        --squares-x 7 --squares-y 9 --square-length-mm 38 --marker-length-mm 28
    uv run python calibrate_camera.py intrinsic --images-dir frames/board `
        --squares-x 7 --squares-y 9 --square-length-mm 38 --marker-length-mm 28 `
        --min-corners 20 --output ../dataset/calib_data/rear_intrinsics.json

    # extrinsics: grab the one best marker frame
    uv run python extract_calib_frames.py marker_shot.mp4 -o frames/marker `
        --mode marker --aruco-dict DICT_4X4_50 --marker-id 0
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_utils.charuco_board import detect_charuco, make_charuco_board
from camera_utils.intrinsic_calibration import imwrite_unicode


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=str, help="Input video (.mp4/.mov).")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Folder to write the extracted .png frames into.")
    parser.add_argument("--mode", choices=("board", "marker"), default="board",
                        help="'board': many ChArUco views for intrinsics (default). "
                             "'marker': the single best ArUco frame for extrinsics.")
    parser.add_argument("--stride", type=int, default=3,
                        help="Only examine every Nth frame (default 3). Detection is the "
                             "slow part; consecutive video frames are near-identical anyway.")
    parser.add_argument("--target", type=int, default=25,
                        help="Stop once this many board views have been kept (default 25). "
                             "0 = keep going to the end of the video.")
    parser.add_argument("--min-sharpness", type=float, default=100.0,
                        help="Reject frames whose variance-of-Laplacian is below this "
                             "(default 100). Scene-dependent -- see the reported "
                             "distribution and retune.")
    parser.add_argument("--min-shift", type=float, default=40.0,
                        help="A frame is kept only if its corners are at least this many "
                             "px (mean, over shared corner ids) from every frame already "
                             "kept (default 40). Raise for stricter pose diversity.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete any existing .png files in --output-dir first.")

    board = parser.add_argument_group("board geometry (must match the printed sheet)")
    board.add_argument("--squares-x", type=int, default=7)
    board.add_argument("--squares-y", type=int, default=9)
    board.add_argument("--square-length-mm", type=float, default=38.0)
    board.add_argument("--marker-length-mm", type=float, default=28.0)
    board.add_argument("--aruco-dict", type=str, default="DICT_5X5_50",
                       help="DICT_5X5_50 for this project's ChArUco board; "
                            "DICT_4X4_50 for its standalone 200mm marker.")
    board.add_argument("--min-corners", type=int, default=20,
                       help="Minimum ChArUco corners for a view to be usable (default 20). "
                            "Deliberately stricter than the calibration scripts' default of "
                            "6, which is tuned for pose recovery rather than intrinsics.")
    board.add_argument("--marker-id", type=int, default=None,
                       help="--mode marker only. Required marker id; any marker if omitted.")
    return parser.parse_args()


def open_video(path):
    """cv2.VideoCapture can fail on Windows paths with non-ASCII characters
    (this repo lives under ".../Universität Stuttgart/..."), the same issue
    intrinsic_calibration.imread_unicode works around for stills. There is no
    buffer-based equivalent for video, so fall back to reading through a
    temporary all-ASCII copy."""
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture, None

    temp_dir = Path(tempfile.mkdtemp(prefix="calib_video_"))
    temp_path = temp_dir / f"input{Path(path).suffix}"
    print(f"Could not open the video directly (likely a non-ASCII path); "
          f"retrying via a temporary copy at {temp_path}")
    shutil.copy2(path, temp_path)

    capture = cv2.VideoCapture(str(temp_path))
    if not capture.isOpened():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise IOError(f"Could not open {path} even via a temporary copy.")
    return capture, temp_dir


def sharpness(gray):
    """Variance of the Laplacian -- the standard cheap blur score. Higher is
    sharper. Absolute values depend on scene texture and resolution, so it is
    only meaningful compared against other frames of the same recording."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def corners_by_id(charuco_corners, charuco_ids):
    return {int(i): c.ravel() for i, c in zip(charuco_ids, charuco_corners)}


def min_shift_from(candidate, kept):
    """Smallest mean corner displacement between `candidate` and any view in
    `kept`, in px. inf when nothing shares enough corners to compare, which
    counts as "clearly a different view"."""
    smallest = float("inf")
    for previous in kept:
        shared = candidate.keys() & previous.keys()
        if len(shared) < 4:
            continue
        mean_shift = float(np.mean([
            np.linalg.norm(candidate[i] - previous[i]) for i in shared]))
        smallest = min(smallest, mean_shift)
    return smallest


def extract_board_frames(capture, output_dir, args):
    _board, detector = make_charuco_board(
        args.squares_x, args.squares_y,
        args.square_length_mm / 1000.0, args.marker_length_mm / 1000.0,
        args.aruco_dict)

    kept_corners = []
    sharpness_seen = []
    examined = detected = 0
    rejected_blur = rejected_duplicate = 0
    frame_index = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.stride:
            continue
        examined += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids = detect_charuco(
            detector, gray, min_corners=args.min_corners)
        if charuco_corners is None:
            continue
        detected += 1

        score = sharpness(gray)
        sharpness_seen.append(score)
        if score < args.min_sharpness:
            rejected_blur += 1
            continue

        candidate = corners_by_id(charuco_corners, charuco_ids)
        if min_shift_from(candidate, kept_corners) < args.min_shift:
            rejected_duplicate += 1
            continue

        kept_corners.append(candidate)
        out_path = output_dir / f"frame_{frame_index:06d}.png"
        imwrite_unicode(out_path, frame)
        print(f"  kept {len(kept_corners):3d}  frame {frame_index:6d}  "
              f"{len(charuco_ids):2d} corners  sharpness {score:7.1f}")

        if args.target and len(kept_corners) >= args.target:
            print(f"  reached --target {args.target}, stopping early.")
            break

    print(f"\nExamined {examined} frames (stride {args.stride}): "
          f"{detected} with the board, {len(kept_corners)} kept.")
    print(f"  rejected {rejected_blur} as blurred, {rejected_duplicate} as "
          f"too similar to a kept view.")
    if sharpness_seen:
        percentiles = np.percentile(sharpness_seen, [10, 50, 90])
        print(f"  sharpness of detected frames: p10={percentiles[0]:.0f} "
              f"median={percentiles[1]:.0f} p90={percentiles[2]:.0f} "
              f"(--min-sharpness is {args.min_sharpness:g})")

    if len(kept_corners) < 10:
        print("\nWARNING: fewer than 10 views. calibrateCamera needs varied "
              "viewpoints to separate focal length from distance -- reshoot with "
              "more tilt and more distance variation, or relax --min-shift/"
              "--min-sharpness.")
    return len(kept_corners)


def extract_marker_frame(capture, output_dir, args):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    best_score = -1.0
    best_frame = None
    best_index = -1
    examined = detected = 0
    frame_index = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.stride:
            continue
        examined += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _corners, ids, _rejected = detector.detectMarkers(gray)
        if ids is None:
            continue
        if args.marker_id is not None and args.marker_id not in ids.ravel():
            continue
        detected += 1

        # Sharpest frame wins: the extrinsic is a single solvePnP, so its
        # accuracy rests entirely on how well this one frame's four marker
        # corners localize.
        score = sharpness(gray)
        if score > best_score:
            best_score, best_frame, best_index = score, frame.copy(), frame_index

    print(f"Examined {examined} frames (stride {args.stride}): "
          f"{detected} containing the marker.")
    if best_frame is None:
        wanted = "any marker" if args.marker_id is None else f"marker id {args.marker_id}"
        print(f"\nNo frame contained {wanted} from {args.aruco_dict}. Check that "
              f"--aruco-dict matches the printed sheet -- this project's 200mm "
              f"marker is DICT_4X4_50, which is NOT the scripts' DICT_5X5_50 default.")
        return 0

    out_path = output_dir / f"marker_frame_{best_index:06d}.png"
    imwrite_unicode(out_path, best_frame)
    print(f"  best frame {best_index} (sharpness {best_score:.1f}) -> {out_path}")
    return 1


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.png"))
    if existing:
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already holds {len(existing)} .png file(s); pass "
                f"--overwrite to replace them. (Mixing frames from two recordings "
                f"would calibrate across two capture modes at once.)")
        for path in existing:
            path.unlink()

    capture, temp_dir = open_video(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{video_path.name}: {width}x{height}, {total} frames")
        print(f"  NOTE: the K you solve from these frames is valid ONLY for "
              f"{width}x{height} capture in this same mode.\n")

        if args.mode == "board":
            written = extract_board_frames(capture, output_dir, args)
        else:
            written = extract_marker_frame(capture, output_dir, args)
    finally:
        capture.release()
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nWrote {written} frame(s) to {output_dir}")


if __name__ == "__main__":
    main()
