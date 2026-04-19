"""
viewer/view/overlay.py
----------------------
OpenCV skeleton-on-video overlay renderer (View layer).

Pure function — no Streamlit dependency, no side-effects.

Coordinate contract
-------------------
``kpts_xyn`` must be in YOLO's ``xyn`` format: float32 (17, 2) where each
value is in [0, 1] relative to the frame dimensions.  Multiply by (W, H) to
get pixel coordinates.

These are the ``raw_landmarks`` stored by the extractor (before the
mean-centre / scale normalisation applied for LSTM training).
"""

from __future__ import annotations

import numpy as np


# Segment group for each COCO keypoint pair — mirrors skeleton.py
_PAIR_TO_GROUP: dict[tuple[int, int], str] = {
    (0, 1): "head",  (0, 2): "head",
    (1, 3): "head",  (2, 4): "head",
    (5, 6): "torso",
    (5, 7): "arm",   (7, 9):  "arm",
    (6, 8): "arm",   (8, 10): "arm",
    (5, 11): "torso", (6, 12): "torso",
    (11, 12): "torso",
    (11, 13): "leg",  (13, 15): "leg",
    (12, 14): "leg",  (14, 16): "leg",
}


def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` hex string to an OpenCV BGR tuple."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_pose_overlay(
    frame_rgb:       np.ndarray,                  # (H, W, 3) uint8 RGB
    kpts_xyn:        np.ndarray,                  # (17, 2) float in [0, 1]
    seg_colours:     dict[str, str],              # {"head": "#hex", ...}
    joint_colour:    str,                         # "#hex"
    body_center_cfg: list[tuple[str, str]],       # [(label, "#hex"), ...]
) -> np.ndarray:
    """Draw skeleton joints and limbs onto *frame_rgb*.

    Parameters
    ----------
    frame_rgb       : Source video frame in RGB order.  Copied — not mutated.
    kpts_xyn        : (17, 2) keypoints in [0, 1] normalised pixel coords.
    seg_colours     : Hex colour per body-segment group (head/arm/torso/leg).
    joint_colour    : Hex colour for joint circle fill.
    body_center_cfg : Body-centre markers as [(label, hex_colour), ...] in
                      order mid-hip, mid-shoulder, half-body.

    Returns
    -------
    Annotated frame as a new RGB numpy array.
    """
    import cv2  # imported here to keep the module importable without OpenCV

    frame = frame_rgb.copy()
    H, W  = frame.shape[:2]

    # (17, 2) → pixel coords
    pts = (kpts_xyn * np.array([W, H], dtype=np.float32)).round().astype(int)

    # ── limb segments ──────────────────────────────────────────────────────
    for (a, b), group in _PAIR_TO_GROUP.items():
        # Skip joints that were not detected (stored as [0, 0])
        if pts[a].sum() == 0 or pts[b].sum() == 0:
            continue
        cv2.line(
            frame,
            tuple(pts[a]), tuple(pts[b]),
            _hex_to_bgr(seg_colours[group]),
            thickness=2, lineType=cv2.LINE_AA,
        )

    # ── joint circles ──────────────────────────────────────────────────────
    jt_bgr = _hex_to_bgr(joint_colour)
    for pt in pts:
        if pt.sum() == 0:
            continue
        cv2.circle(frame, tuple(pt), radius=5, color=jt_bgr, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, tuple(pt), radius=5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # ── body-centre diamond markers ────────────────────────────────────────
    mid_hip      = ((pts[11].astype(float) + pts[12].astype(float)) / 2).astype(int)
    mid_shoulder = ((pts[5].astype(float)  + pts[6].astype(float))  / 2).astype(int)
    half_body    = ((pts[11].astype(float) + pts[12].astype(float)
                   + pts[5].astype(float)  + pts[6].astype(float)) / 4).astype(int)

    for (label, hex_colour), cpt in zip(body_center_cfg, [mid_hip, mid_shoulder, half_body]):
        bgr  = _hex_to_bgr(hex_colour)
        size = 8
        diamond = np.array([
            [cpt[0],        cpt[1] - size],
            [cpt[0] + size, cpt[1]       ],
            [cpt[0],        cpt[1] + size],
            [cpt[0] - size, cpt[1]       ],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [diamond], bgr)
        cv2.polylines(frame, [diamond], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame, label,
            (cpt[0] + 10, cpt[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1, cv2.LINE_AA,
        )

    return frame
