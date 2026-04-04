"""
viewer/model/pose_data.py
-------------------------
Cached pose-data loader (Model layer).

Loads a .pt file and pre-computes all pandas DataFrames needed by the viewer.
Results are cached per unique file path by Streamlit so re-selecting the same
file never recomputes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── ensure src/ is importable ────────────────────────────────────────────────
from ..config import SRC_ROOT

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utilities.file_io import load_torch
from utilities         import pose_analysis_yolo


@st.cache_data(show_spinner="Loading pose data …")
def load_pose_data(path: str) -> tuple:
    """Load a .pt pose file and pre-compute all feature DataFrames.

    Returns
    -------
    landmarks_np     : ndarray (T, 17, 2)
    metadata         : dict
    speed_df         : DataFrame – per-joint speed
    accel_df         : DataFrame – per-joint acceleration
    dist_hip_df      : DataFrame – joint distances from hip centre
    dist_shoulder_df : DataFrame – joint distances from shoulder centre
    dist_half_df     : DataFrame – joint distances from half-body centre
    center_speed_df  : DataFrame – body-centre speeds
    center_accel_df  : DataFrame – body-centre accelerations
    """
    data      = load_torch(path)
    landmarks = data["landmarks"]          # (T, 17, 2) tensor
    metadata  = data.get("metadata", {})

    landmarks_np = (
        landmarks.numpy() if hasattr(landmarks, "numpy") else np.array(landmarks)
    )

    # raw_landmarks: xyn in [0,1] pixel-relative coords saved by the extractor.
    # Older .pt files won't have this key — callers must handle None.
    raw = data.get("raw_landmarks")
    raw_landmarks_np: np.ndarray | None = (
        raw.numpy() if hasattr(raw, "numpy") else (np.array(raw) if raw is not None else None)
    )

    speed_df         = pose_analysis_yolo.compute_joint_speeds_df(landmarks)
    accel_df         = pose_analysis_yolo.compute_joint_acceleration_df(landmarks)
    dist_hip_df      = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="hip")
    dist_shoulder_df = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="shoulder")
    dist_half_df     = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="half_body")
    center_speed_df  = pose_analysis_yolo.compute_body_center_speed_df(landmarks)
    center_accel_df  = pose_analysis_yolo.compute_body_center_accel_df(landmarks)

    return (
        landmarks_np, metadata,
        speed_df, accel_df,
        dist_hip_df, dist_shoulder_df, dist_half_df,
        center_speed_df, center_accel_df,
        raw_landmarks_np,
    )


def extract_and_save_raw_landmarks(
    video_path: str,
    pt_path: str,
    yolo_model_path: str,
    on_progress=None,
) -> "np.ndarray | None":
    """Re-run YOLO on *video_path* to extract raw xyn keypoints and patch *pt_path*.

    The extracted ``raw_landmarks`` tensor is saved back into the .pt file so
    subsequent loads find it automatically without re-running this function.
    Clears the ``load_pose_data`` Streamlit cache after saving.

    Parameters
    ----------
    on_progress : callable(current_frame: int, total_frames: int), optional
        Invoked after every frame so the caller can update a progress bar.

    Returns
    -------
    ndarray (T, 17, 2) float32, or ``None`` on any failure.
    """
    try:
        import cv2
        import torch as _torch
        from ultralytics import YOLO

        model = YOLO(yolo_model_path)
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_list: list = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False)
            result  = results[0]
            if result.keypoints is not None and len(result.keypoints.xyn) > 0:
                kpts = result.keypoints.xyn[0].cpu().float().numpy()
                kpts = kpts[:17, :2] if kpts.shape[0] >= 17 else np.zeros((17, 2), dtype=np.float32)
            else:
                kpts = np.zeros((17, 2), dtype=np.float32)
            raw_list.append(kpts)
            i += 1
            if on_progress:
                on_progress(i, max(total, i))

        cap.release()

        if not raw_list:
            return None

        raw_np = np.stack(raw_list).astype(np.float32)

        # Patch the .pt file so future loads find raw_landmarks automatically.
        data = _torch.load(pt_path, map_location="cpu", weights_only=False)
        data["raw_landmarks"] = _torch.from_numpy(raw_np)
        _torch.save(data, pt_path)

        # Invalidate the cache so the next load_pose_data call re-reads the file.
        load_pose_data.clear()

        return raw_np

    except Exception:
        return None
