"""
yolo_pose_config.py
-------------------
Shared constants for YOLO-pose (17-keypoint COCO layout) analysis.
Import from this module wherever keypoint names or joint definitions are needed.
"""


# ── Keypoint names (index 0-16, COCO order) ──────────────────────────────────
KEYPOINT_NAMES: list[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)


# ── Skeleton connectivity (pairs of keypoint indices) ────────────────────────
SKELETON_PAIRS: list[tuple[int, int]] = [
    (0, 1), (0, 2),            # nose – eyes
    (1, 3), (2, 4),            # eyes – ears
    (5, 6),                    # shoulders
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 11), (6, 12),          # torso sides
    (11, 12),                  # hips
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
]


# ── Joint-angle triplets (name, a_idx, vertex_idx, c_idx) ────────────────────
# Angle is measured at *vertex* between vectors vertex→a and vertex→c.
JOINT_ANGLE_TRIPLETS: list[tuple[str, int, int, int]] = [
    ("left_elbow",      5,  7,  9),
    ("right_elbow",     6,  8, 10),
    ("left_shoulder",   7,  5, 11),
    ("right_shoulder",  8,  6, 12),
    # ("left_knee",      11, 13, 15),
    # ("right_knee",     12, 14, 16),
]

JOINT_ANGLE_TRIPLETS_CAL: list[tuple[str, int, int, int]] = [
    ("neck",     5, 6, 0),   # angle at neck between shoulders and nose
    
]


RATIO_BETWEEN_DISTS: list[tuple[str, tuple[str, str], tuple[str, str]]] = [
    # ("elbow/shoulder",      ("left_elbow", "right_elbow"), ("left_shoulder", "right_shoulder")),
    ("wrist/shoulder",     ("right_wrist", "left_wrist"),  ("left_shoulder", "right_shoulder")),
    # ("wrist/elbow",       ("right_wrist", "left_wrist"),  ("left_elbow", "right_elbow")),
]


# ── Body-centre definitions ───────────────────────────────────────────────────
# Each entry maps a centre name to (compute_fn, display_label).
# compute_fn: (T, 17, 2) tensor → (T, 2) tensor
CENTER_CFG: dict[str, tuple] = {
    "hip": (
        lambda lm: (lm[:, 11, :] + lm[:, 12, :]) / 2.0,
        "mid-hip",
    ),
    "shoulder": (
        lambda lm: (lm[:, 5, :] + lm[:, 6, :]) / 2.0,
        "mid-shoulder",
    ),
    "half_body": (
        lambda lm: (lm[:, 11, :] + lm[:, 12, :] + lm[:, 5, :] + lm[:, 6, :]) / 4.0,
        "half-body",
    ),
}



STEP_ID_LABEL: dict[int, str] = {
    0: "Put spacer",
    1: "Align piece",
    2: "Screwing",
    3: "Move spacer",
    }