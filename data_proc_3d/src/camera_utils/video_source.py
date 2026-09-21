"""Opening a live camera at a KNOWN resolution.

cv2.VideoCapture does not ask a device for its best mode. On Windows the
DirectShow backend negotiates whatever format it settles on first --
routinely 640x480 -- regardless of what the device can actually deliver.
An OBS Virtual Camera emitting 1920x1080 or 3840x2160 therefore arrives in
OpenCV as 640x480 unless the resolution is requested explicitly, and
nothing in the capture path reports that anything was lost.

That matters here more than it would elsewhere, because resolution is part
of the calibration: K is only valid for the exact mode it was solved in. A
calibration that silently ran at 640x480 while the live pipeline later runs
at 1920x1080 is wrong by a factor of 3 in fx, fy, cx and cy, and nothing
about either run looks abnormal.

So every live capture in this package goes through open_camera(), which
requests a resolution and then VERIFIES it against a decoded frame. The
verification is the point: cap.get(CAP_PROP_FRAME_WIDTH) frequently echoes
back the value that was just set even when the device ignored it, so the
properties cannot be trusted and only a real frame's shape can.
"""
import sys

import cv2

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,   # DirectShow -- how OBS Virtual Camera registers on Windows
    "msmf": cv2.CAP_MSMF,     # Media Foundation -- OpenCV's Windows default, slower to open
    "any": cv2.CAP_ANY,
}


def resolve_backend(name="auto"):
    """'auto' picks DirectShow on Windows, where it is markedly more
    reliable than Media Foundation for virtual cameras, and leaves the
    choice to OpenCV everywhere else."""
    if name == "auto":
        return BACKENDS["dshow"] if sys.platform == "win32" else BACKENDS["any"]
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {sorted(BACKENDS)} or 'auto'.")
    return BACKENDS[name]


def open_camera(index, width=None, height=None, backend="auto", quiet=False):
    """Open camera `index`, optionally requesting width x height.

    Returns (capture, (actual_width, actual_height)). The actual size is
    read off a decoded frame, not off the capture properties. Raises IOError
    if the device will not open; prints a warning (rather than raising) if
    it opens but refuses the requested size, since a different-but-usable
    mode is still worth calibrating in as long as the operator knows which
    one they got.
    """
    capture = cv2.VideoCapture(index, resolve_backend(backend))
    if not capture.isOpened():
        capture.release()
        raise IOError(
            f"Could not open camera index {index} on the {backend} backend. "
            f"Run app/check_video_source.py --list to see which indices exist.")

    if width and height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise IOError(
            f"Camera index {index} opened but delivered no frame. If this is an OBS "
            f"Virtual Camera, start it in OBS (Controls > Start Virtual Camera).")
    actual = (frame.shape[1], frame.shape[0])

    if width and height and actual != (width, height):
        print(f"  WARNING: requested {width}x{height} but the device delivered "
              f"{actual[0]}x{actual[1]}.")
        print(f"           Calibrating anyway -- the intrinsics will be valid for "
              f"{actual[0]}x{actual[1]}, so the live pipeline must run at that size too.")
        print(f"           For an OBS Virtual Camera, set Settings > Video > Output "
              f"(Scaled) Resolution, then STOP and START the virtual camera.")
    elif not quiet:
        print(f"  Capturing at {actual[0]}x{actual[1]}.")

    return capture, actual


def add_capture_args(parser):
    """Shared --capture-width/--capture-height/--backend flags, so every
    entry point that opens a live camera spells them the same way."""
    parser.add_argument("--capture-width", type=int, default=None,
                        help="Request this capture width. Without it OpenCV picks a "
                             "default that is often 640 regardless of what the device "
                             "can deliver -- see camera_utils/video_source.py.")
    parser.add_argument("--capture-height", type=int, default=None,
                        help="Request this capture height. Use together with "
                             "--capture-width.")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "any"), default="auto",
                        help="Capture backend (default auto: DirectShow on Windows).")
