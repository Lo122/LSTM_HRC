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
from pathlib import Path

import numpy as np
import streamlit as st

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── ensure src/ is importable ────────────────────────────────────────────────
from config import SRC_ROOT, cfg

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from file_io_utils import load_elan_label_data, load_step_ids_from_json, load_torch
from data_proc_2d.src         import pose_analysis_yolo


def _get_optional_tensor(data: dict, *names: str):
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return None


def _load_labels_for_video(
    video_file_name: str | None,
    label_root: Path,
    extension: str,
    loader,
) -> list[dict]:
    if not video_file_name or not label_root.exists():
        return []

    for label_path in sorted(label_root.rglob(f"*{extension}")):
        label_info, referenced_video_name = loader(label_path)
        if referenced_video_name == video_file_name:
            return label_info or []

    return []


@st.cache_data(show_spinner="Loading pose data …")
def load_pose_data(path: str) -> tuple:
    """Load a .pt pose file and pre-compute all feature DataFrames.

    Returns
    -------
    landmarks_np     : ndarray (T, 17, 2)
    metadata         : dict
    speed_df         : DataFrame – per-joint speed
    accel_df         : DataFrame – per-joint acceleration
    angle_df         : DataFrame – per-joint angles
    ratio_df         : DataFrame – configured distance ratios
    dist_half_df     : DataFrame – joint distances from half-body centre
    dist_hip_df      : DataFrame – joint distances from hip centre
    dist_shoulder_df : DataFrame – joint distances from shoulder centre
    center_speed_df  : DataFrame – body-centre speeds
    center_accel_df  : DataFrame – body-centre accelerations
    step_labels      : list[dict] – step labels loaded from JSON
    elan_step_labels : list[dict] – step labels loaded from ELAN CSV
    """
    data      = load_torch(path)
    landmarks = data["landmarks"]          # (T, 17, 2) tensor
    speed     = _get_optional_tensor(data, "speed", "feat_speed")
    acceleration = _get_optional_tensor(data, "acceleration", "feat_acc")
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
    video_path = metadata.get("video")
    video_file_name = os.path.basename(video_path) if video_path else None

    speed_df         = pose_analysis_yolo.compute_joint_speeds_df(landmarks, speed=speed)
    accel_df         = pose_analysis_yolo.compute_joint_acceleration_df(
        landmarks,
        acceleration=acceleration,
        speed=speed,
    )
    angle_df         = pose_analysis_yolo.compute_joint_angles_df(landmarks)
    ratio_df         = pose_analysis_yolo.compute_distance_ratios_df(landmarks)
    dist_half_df     = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="half_body")
    dist_hip_df      = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="hip")
    dist_shoulder_df = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="shoulder")
    center_speed_df  = pose_analysis_yolo.compute_body_center_speed_df(landmarks)
    center_accel_df  = pose_analysis_yolo.compute_body_center_accel_df(landmarks)
    step_labels      = _load_labels_for_video(
        video_file_name,
        cfg.step_label_root,
        ".json",
        load_step_ids_from_json,
    )
    elan_step_labels = _load_labels_for_video(
        video_file_name,
        cfg.elan_label_root,
        ".csv",
        load_elan_label_data,
    )

    return (
        landmarks_np, metadata,
        speed_df, accel_df, angle_df, ratio_df,
        dist_half_df, dist_hip_df, dist_shoulder_df,
        center_speed_df, center_accel_df,
        step_labels, elan_step_labels,
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
