"""Find and validate a live camera index before calibrating against it --
written for an OBS Virtual Camera fed by DroidCam, but it works for any
cv2.VideoCapture source.

calibrate_camera.py's live subcommands take --camera-index, and on Windows
that index is just a position in DirectShow's enumeration: it carries no
name, it shifts when devices are plugged/unplugged or when OBS's virtual
camera is started/stopped, and several unrelated devices commonly report the
same resolution. So "which number is my phone" is not answerable by
guessing. --list answers it visually.

--preview then checks the thing that actually matters: whether the board or
marker still DETECTS after passing through OBS. Three OBS-specific failure
modes this catches:

  MIRRORED SOURCE. The single most confusing one. ArUco bit patterns are
  chirality-sensitive, so a horizontally flipped marker is not a valid
  marker -- detection drops to exactly zero with no error message, and the
  preview looks perfectly fine to a human. If a scene has "Flip Horizontal"
  applied (common, because it makes a webcam feel like a mirror), nothing
  will ever calibrate. --preview detects on both the frame and its mirror
  and says so explicitly when only the mirror works.

  UNEXPECTED RESOLUTION. OBS Virtual Camera emits at the CANVAS resolution
  (Settings > Video > Output/Scaled Resolution), not at DroidCam's native
  resolution. Whatever scaling, cropping or letterboxing OBS applies to fit
  the source onto that canvas is baked into the images you calibrate from.
  That is fine -- it is the very reason to calibrate through OBS rather than
  through DroidCam directly -- but it makes K a property of the whole OBS
  scene, not of the phone. See the warning --preview prints.

  BLACK / PLACEHOLDER FRAMES. An index can open successfully and still
  deliver nothing usable when the OBS virtual camera has not been started,
  which otherwise surfaces much later as "the board is never detected".

ONCE IT DETECTS, LOCK THE SCENE. K is only valid for the exact pipeline it
was solved through. Changing the OBS canvas resolution, the source's
scale/crop/transform, any filter that resamples, or DroidCam's own
resolution setting all silently invalidate the calibration. Recalibrate
after touching any of them.

Usage:
    cd data_proc_3d/app

    # 1. which index is the OBS virtual camera?
    uv run python check_video_source.py --list

    # 2. does the ChArUco board detect through it?
    uv run python check_video_source.py --preview 2 `
        --squares-x 7 --squares-y 9 --square-length-mm 38 --marker-length-mm 28

    # 3. does the 200mm extrinsic marker detect through it?
    uv run python check_video_source.py --preview 2 --mode marker `
        --aruco-dict DICT_4X4_50 --marker-id 0

Then hand that same index to the real calibration:
    uv run python calibrate_camera.py intrinsic --camera-index 2 ...
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Probing indices that hold no device makes DirectShow log a C++ exception
# and a "can't be used to capture by index" warning per miss. That is the
# expected result of probing, not a fault, and it buries the actual report.
# The env var has to be set before cv2 is imported to take effect.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

cv2.setLogLevel(2)  # 2 == ERROR; cv2.utils.logging is absent in some builds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_utils.charuco_board import detect_charuco, draw_charuco_detection, make_charuco_board
from camera_utils.intrinsic_calibration import imwrite_unicode
from camera_utils.video_source import add_capture_args, open_camera, resolve_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="Probe camera indices and write a contact sheet showing what "
                             "each one sees, so the right index can be identified by eye.")
    parser.add_argument("--preview", type=int, default=None, metavar="INDEX",
                        help="Open this index live with detection overlaid. Q or ESC to quit.")
    parser.add_argument("--max-index", type=int, default=8,
                        help="Highest index to probe with --list (default 8).")
    add_capture_args(parser)
    parser.add_argument("--contact-sheet", type=str, default="camera_indices.png",
                        help="Where --list writes its contact sheet (default "
                             "./camera_indices.png).")
    parser.add_argument("--mode", choices=("board", "marker"), default="board",
                        help="What --preview looks for (default board).")

    target = parser.add_argument_group("target geometry (must match the printed sheet)")
    target.add_argument("--squares-x", type=int, default=7)
    target.add_argument("--squares-y", type=int, default=9)
    target.add_argument("--square-length-mm", type=float, default=38.0)
    target.add_argument("--marker-length-mm", type=float, default=28.0)
    target.add_argument("--aruco-dict", type=str, default="DICT_5X5_50",
                        help="DICT_5X5_50 for the ChArUco board, DICT_4X4_50 for the "
                             "standalone 200mm marker.")
    target.add_argument("--min-corners", type=int, default=20)
    target.add_argument("--marker-id", type=int, default=None)
    return parser.parse_args()


def probe(index, backend, width=None, height=None):
    """Open one index just long enough to learn what it delivers. Returns
    (frame, info) with frame None when the index is unusable.

    When width/height are given they are REQUESTED, and what comes back is
    what the device actually agreed to -- which is the only way to find out
    whether a device will honour a mode. OpenCV otherwise negotiates its own
    default (often 640x480) no matter what the device can deliver."""
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        return None, "not available"
    try:
        if width and height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None, "opened but delivered no frame"
        # Off the decoded frame, not off CAP_PROP_FRAME_WIDTH -- the property
        # often echoes back whatever was just set, even when it was ignored.
        actual_w, actual_h = frame.shape[1], frame.shape[0]
        fps = capture.get(cv2.CAP_PROP_FPS)
        refused = (" [REQUESTED %dx%d - REFUSED]" % (width, height)
                   if width and height and (actual_w, actual_h) != (width, height) else "")

        # A virtual camera that has not been started still opens fine and
        # still delivers frames -- it just serves a placeholder. Two kinds
        # show up, and both otherwise look like a working device:
        #   flat black            -> near-zero pixel variance
        #   a vendor logo card    -> plenty of variance, but STATIC. OBS's
        #                            "camera off" logo is exactly this, so
        #                            variance alone reports it as healthy.
        # Comparing two frames a few frame-periods apart separates a live
        # feed (sensor noise guarantees they differ) from either placeholder.
        time.sleep(0.2)
        ok2, frame2 = capture.read()
        if frame.std() < 1.0:
            note = " (BLANK - is the OBS virtual camera started?)"
        elif ok2 and frame2 is not None and np.array_equal(frame, frame2):
            note = " (STATIC placeholder - start the virtual camera in OBS)"
        else:
            note = ""
        return frame, f"{actual_w}x{actual_h} @ {fps:.0f}fps{refused}{note}"
    finally:
        capture.release()


def list_devices(args):
    backend = resolve_backend(args.backend)
    print(f"Probing indices 0..{args.max_index} on the {args.backend} backend.")
    if args.capture_width and args.capture_height:
        print(f"Requesting {args.capture_width}x{args.capture_height} from each device; "
              f"anything marked REFUSED cannot deliver it.")
    print()

    thumbnails = []
    for index in range(args.max_index + 1):
        frame, info = probe(index, backend, args.capture_width, args.capture_height)
        print(f"  index {index}: {info}")
        if frame is None:
            continue
        thumb = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        cv2.rectangle(thumb, (0, 0), (thumb.shape[1] - 1, 28), (0, 0, 0), -1)
        cv2.putText(thumb, f"index {index}  {info.split(' (')[0]}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        thumbnails.append(thumb)

    if not thumbnails:
        print("\nNo usable camera indices. If OBS is running, start Virtual Camera "
              "(Controls > Start Virtual Camera) and try again.")
        return

    height = max(t.shape[0] for t in thumbnails)
    padded = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 4,
                                 cv2.BORDER_CONSTANT, value=(40, 40, 40))
              for t in thumbnails]
    sheet_path = Path(args.contact_sheet)
    imwrite_unicode(sheet_path, np.hstack(padded))
    print(f"\nContact sheet -> {sheet_path.resolve()}")
    print("Open it, find the panel showing your phone, and use that index.")


def make_detector(args):
    """Returns detect(gray) -> (n_found, overlay_fn) for whichever target."""
    if args.mode == "board":
        _board, detector = make_charuco_board(
            args.squares_x, args.squares_y,
            args.square_length_mm / 1000.0, args.marker_length_mm / 1000.0,
            args.aruco_dict)
        total = (args.squares_x - 1) * (args.squares_y - 1)

        def detect(gray, min_corners):
            corners, ids = detect_charuco(detector, gray, min_corners=min_corners)
            return (0 if ids is None else len(ids)), (corners, ids)

        def draw(display, payload):
            draw_charuco_detection(display, *payload)

        return detect, draw, f"ChArUco corners (of {total})"

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    aruco = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def detect(gray, _min_corners):
        corners, ids, _rejected = aruco.detectMarkers(gray)
        if ids is not None and args.marker_id is not None:
            keep = [i for i, mid in enumerate(ids.ravel()) if mid == args.marker_id]
            corners = [corners[i] for i in keep]
            ids = ids[keep] if keep else None
        return (0 if ids is None else len(ids)), (corners, ids)

    def draw(display, payload):
        corners, ids = payload
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

    wanted = "any id" if args.marker_id is None else f"id {args.marker_id}"
    return detect, draw, f"{args.aruco_dict} markers ({wanted})"


def preview(args):
    capture, (width, height) = open_camera(
        args.preview, args.capture_width, args.capture_height, args.backend)

    detect, draw, label = make_detector(args)
    print(f"Index {args.preview}: {width}x{height}, looking for {label}.")
    print(f"Any K solved through this source is valid ONLY for {width}x{height} "
          f"with the OBS scene exactly as it is now.")
    print("Q or ESC to quit.\n")

    mirror_warned = False
    frames_seen = 0
    frames_detected = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Frame read failed -- source stopped?")
                break
            frames_seen += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, payload = detect(gray, args.min_corners)
            if found:
                frames_detected += 1

            # Mirrored sources detect on the flip and never on the original.
            if not found and not mirror_warned and frames_seen % 15 == 0:
                mirrored_found, _ = detect(cv2.flip(gray, 1), args.min_corners)
                if mirrored_found:
                    mirror_warned = True
                    print("*** SOURCE IS MIRRORED ***")
                    print("    The target detects only in the horizontally flipped image.")
                    print("    ArUco patterns are chirality-sensitive, so nothing will")
                    print("    ever calibrate this way. In OBS, right-click the source >")
                    print("    Transform > uncheck Flip Horizontal, then re-run.")

            display = frame.copy()
            draw(display, payload)
            colour = (0, 255, 0) if found else (0, 0, 255)
            cv2.rectangle(display, (0, 0), (display.shape[1], 64), (0, 0, 0), -1)
            cv2.putText(display, f"index {args.preview}  {width}x{height}  {label}: {found}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(display, "MIRRORED - fix the OBS transform" if mirror_warned
                        else f"detected in {frames_detected}/{frames_seen} frames",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if mirror_warned else (200, 200, 200), 2)
            cv2.imshow(f"check_video_source - index {args.preview} - Q to quit", display)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"\nDetected the target in {frames_detected} of {frames_seen} frames.")
    if frames_detected == 0:
        print("Nothing detected. Check, in order: the OBS virtual camera is started; "
              "the source is not flipped (see above); --aruco-dict matches the printed "
              "sheet; the target is lit and in focus.")
    else:
        print(f"Source looks good. Calibrate with:\n"
              f"  uv run python calibrate_camera.py intrinsic --camera-index {args.preview} "
              f"--squares-x {args.squares_x} --squares-y {args.squares_y} "
              f"--square-length-mm {args.square_length_mm:g} "
              f"--marker-length-mm {args.marker_length_mm:g} --min-corners {args.min_corners}")


def main():
    args = parse_args()
    if args.preview is not None:
        preview(args)
    elif args.list:
        list_devices(args)
    else:
        raise SystemExit("Pass --list to find your camera index, or --preview INDEX "
                         "to validate one. See --help.")


if __name__ == "__main__":
    main()
