"""
viewer/view/video.py
---------------
Video frame utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def grab_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Return the video frame at *frame_idx* as an RGB numpy array.

    Returns *None* on any failure (file missing, codec error, etc.).
    """
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def resolve_video_path(metadata: dict) -> tuple[str, bool]:
    """Extract the video file path from pose metadata.

    Returns
    -------
    path : str    Ethe resolved path string (may be empty)
    ok   : bool   ETrue if *path* points to an existing file
    """
    path = str(metadata.get("video") or metadata.get("video_path") or "")
    ok   = bool(path) and Path(path).exists()
    return path, ok
