import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))
    
from utilities.yolo_pose_config import (
    KEYPOINT_NAMES,
    JOINT_ANGLE_TRIPLETS,
    CENTER_CFG,
)



def extract_features(landmarks: torch.Tensor) -> dict:
    """Extract pose features from a sequence of normalised keypoints.

    Args:
        landmarks: (T, 17, 2) tensor of normalised 2-D keypoints.

    Returns:
        Dictionary with velocity statistics and joint-angle statistics.
    """
    T = landmarks.shape[0]

    # ---- frame-to-frame speed per keypoint ----
    velocity  = landmarks[1:] - landmarks[:-1]       # (T-1, 17, 2)
    speed     = velocity.norm(dim=2)                 # (T-1, 17)
    mean_speed_per_joint = {
        KEYPOINT_NAMES[i]: round(float(speed[:, i].mean()), 6)
        for i in range(17)
    }

    # ---- overall body speed summary ----
    body_speed_mean = round(float(speed.mean()), 6)
    body_speed_max  = round(float(speed.max()),  6)

    # ---- frame-to-frame acceleration per keypoint ----
    accel = speed[1:] - speed[:-1]                   # (T-2, 17)  signed
    accel_abs = accel.abs()                           # (T-2, 17)
    mean_accel_per_joint = {
        KEYPOINT_NAMES[i]: round(float(accel_abs[:, i].mean()), 6)
        for i in range(17)
    }
    body_accel_mean = round(float(accel_abs.mean()), 6)
    body_accel_max  = round(float(accel_abs.max()),  6)

    # ---- joint angles ----
    joint_angles: dict = {}
    for name, a, vertex, c in JOINT_ANGLE_TRIPLETS:
        ang = _angle_at_joint(landmarks, a, vertex, c)  # (T,)
        joint_angles[name] = {
            "mean_deg": round(float(ang.mean()), 4),
            "std_deg":  round(float(ang.std()),  4),
            "min_deg":  round(float(ang.min()),  4),
            "max_deg":  round(float(ang.max()),  4),
        }

    return {
        "num_frames":            T,
        "body_speed_mean":       body_speed_mean,
        "body_speed_max":        body_speed_max,
        "mean_speed_per_joint":  mean_speed_per_joint,
        "body_accel_mean":       body_accel_mean,
        "body_accel_max":        body_accel_max,
        "mean_accel_per_joint":  mean_accel_per_joint,
        "joint_angles":          joint_angles,
    }
    
    
def _angle_at_joint(
    kpts: torch.Tensor, a: int, vertex: int, c: int
) -> torch.Tensor:
    """Compute the angle (degrees) at *vertex* across T frames.

    Args:
        kpts:   (T, 17, 2) normalised keypoints.
        a:      index of the first arm keypoint.
        vertex: index of the joint at which the angle is measured.
        c:      index of the second arm keypoint.

    Returns:
        (T,) tensor of angles in degrees.
    """
    va = kpts[:, a, :]      - kpts[:, vertex, :]   # (T, 2)
    vc = kpts[:, c, :]      - kpts[:, vertex, :]   # (T, 2)
    dot  = (va * vc).sum(dim=1)                    # (T,)
    norm = va.norm(dim=1) * vc.norm(dim=1) + 1e-8  # (T,)
    cos_angle = (dot / norm).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).rad2deg()          # (T,)


# ============================================================
# Per-frame feature computation  →  DataFrame / CSV
# ============================================================

def compute_joint_speeds_df(landmarks: torch.Tensor) -> pd.DataFrame:
    """Compute per-frame joint speed and return as a DataFrame.

    Columns: ``frame``, ``{joint}_speed`` × 17.
    Row *i* corresponds to the transition from frame *i* to frame *i+1*
    (so the DataFrame has T-1 rows for T input frames).

    Args:
        landmarks: (T, 17, 2) normalised keypoints.

    Returns:
        DataFrame with shape (T-1, 18).
    """
    velocity = landmarks[1:] - landmarks[:-1]   # (T-1, 17, 2)
    speed    = velocity.norm(dim=2).numpy()      # (T-1, 17)
    data = {"frame": np.arange(1, speed.shape[0] + 1)}
    for i, name in enumerate(KEYPOINT_NAMES):
        data[f"{name}_speed"] = speed[:, i]
    return pd.DataFrame(data)


_CENTER_CFG = CENTER_CFG


def compute_joint_acceleration_df(landmarks: torch.Tensor) -> pd.DataFrame:
    """Compute per-frame joint acceleration magnitude and return as a DataFrame.

    Acceleration = |speed[i+1] - speed[i]|, so the DataFrame has T-2 rows.
    Columns: ``frame``, ``{joint}_accel`` × 17.

    Args:
        landmarks: (T, 17, 2) normalised keypoints.

    Returns:
        DataFrame with shape (T-2, 18).
    """
    velocity = landmarks[1:] - landmarks[:-1]        # (T-1, 17, 2)
    speed    = velocity.norm(dim=2)                  # (T-1, 17)
    accel    = (speed[1:] - speed[:-1]).abs().numpy()  # (T-2, 17)
    data = {"frame": np.arange(2, accel.shape[0] + 2)}
    for i, name in enumerate(KEYPOINT_NAMES):
        data[f"{name}_accel"] = accel[:, i]
    return pd.DataFrame(data)


def compute_joint_distances_df(landmarks: torch.Tensor, center: str = "hip") -> pd.DataFrame:
    """Compute per-frame distance of every keypoint from a body-centre point.

    Columns: ``frame``, ``{joint}_dist`` × 17.

    Args:
        landmarks: (T, 17, 2) normalised keypoints.
        center:    ``"hip"``, ``"shoulder"``, or ``"half_body"``.

    Returns:
        DataFrame with shape (T, 18).
    """
    if center not in _CENTER_CFG:
        raise ValueError(f"Unknown center '{center}'. Choose from: {list(_CENTER_CFG)}.")
    center_fn, _ = _CENTER_CFG[center]
    body_center = center_fn(landmarks).numpy()    # (T, 2)
    kpts = landmarks.numpy()                       # (T, 17, 2)
    diff = kpts - body_center[:, None, :]          # (T, 17, 2)
    dist = np.linalg.norm(diff, axis=2)            # (T, 17)
    data = {"frame": np.arange(dist.shape[0])}
    for i, name in enumerate(KEYPOINT_NAMES):
        data[f"{name}_dist_{center}"] = dist[:, i]
    return pd.DataFrame(data)


def merge_pose_dfs(
    speed_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    dist_hip_df: pd.DataFrame,
    dist_shoulder_df: pd.DataFrame,
    dist_half_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all per-frame pose DataFrames on the ``frame`` column.

    Because speed (T-1 rows), acceleration (T-2 rows), and distances (T rows)
    have different lengths, an outer join is used and missing values are left as
    ``NaN``.

    Column groups in the result:
        * ``frame``
        * ``{joint}_speed``          × 17
        * ``{joint}_accel``          × 17
        * ``{joint}_dist_hip``       × 17
        * ``{joint}_dist_shoulder``  × 17
        * ``{joint}_dist_half_body`` × 17

    Args:
        speed_df:         output of :func:`compute_joint_speeds_df`.
        accel_df:         output of :func:`compute_joint_acceleration_df`.
        dist_hip_df:      output of :func:`compute_joint_distances_df` with ``center=\"hip\"``.
        dist_shoulder_df: output of :func:`compute_joint_distances_df` with ``center=\"shoulder\"``.
        dist_half_df:     output of :func:`compute_joint_distances_df` with ``center=\"half_body\"``.

    Returns:
        Merged DataFrame sorted by ``frame``.
    """
    dfs = [speed_df, accel_df, dist_hip_df, dist_shoulder_df, dist_half_df]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="frame", how="outer")
    return merged.sort_values("frame").reset_index(drop=True)


def compute_body_center_speed_df(landmarks: torch.Tensor) -> pd.DataFrame:
    """Compute per-frame speed of each body-centre point (hip, shoulder, half_body).

    Columns: ``frame``, ``hip_center_speed``, ``shoulder_center_speed``,
    ``half_body_center_speed``.  Shape: (T-1, 4).

    Args:
        landmarks: (T, 17, 2) normalised keypoints.

    Returns:
        DataFrame with shape (T-1, 4).
    """
    data: dict = {"frame": np.arange(1, landmarks.shape[0])}
    for name, (fn, _) in _CENTER_CFG.items():
        center = fn(landmarks)                    # (T, 2)
        vel    = center[1:] - center[:-1]         # (T-1, 2)
        data[f"{name}_center_speed"] = vel.norm(dim=1).numpy()
    return pd.DataFrame(data)


def compute_body_center_accel_df(landmarks: torch.Tensor) -> pd.DataFrame:
    """Compute per-frame acceleration of each body-centre point.

    Columns: ``frame``, ``hip_center_accel``, ``shoulder_center_accel``,
    ``half_body_center_accel``.  Shape: (T-2, 4).

    Args:
        landmarks: (T, 17, 2) normalised keypoints.

    Returns:
        DataFrame with shape (T-2, 4).
    """
    data: dict = {"frame": np.arange(2, landmarks.shape[0])}
    for name, (fn, _) in _CENTER_CFG.items():
        center = fn(landmarks)                         # (T, 2)
        speed  = (center[1:] - center[:-1]).norm(dim=1)  # (T-1,)
        accel  = (speed[1:] - speed[:-1]).abs().numpy()   # (T-2,)
        data[f"{name}_center_accel"] = accel
    return pd.DataFrame(data)


