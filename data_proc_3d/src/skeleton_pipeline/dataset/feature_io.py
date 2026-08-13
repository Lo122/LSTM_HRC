"""Turns one generate_lstm_training_data.py .npz output into the
{panel_key: (T, n_cols) tensor} feature dict saved by app/build_training_pairs.py
-- thin glue over skeleton_pipeline.features.h36m_features.compute_all_features
(the actual kinematic-feature math already lives there)."""
import re
from pathlib import Path

import numpy as np

from ..features.h36m_features import compute_all_features


def extract_features(npz_file: Path, logger) -> tuple[dict, dict]:
    """Returns (metadata, features): features is {panel_key: torch.Tensor
    of shape (T, n_cols_in_panel)}; metadata carries fps/frame count/bone
    lengths plus panel_columns (panel_key -> column names, in tensor
    column order) so downstream code can recover individual feature names."""
    data = np.load(npz_file, allow_pickle=True)
    positions = data["keypoints_3d"]          # (T, 17, 3), root-relative, meters, may contain NaN rows
    fps = float(data["fps"])
    total_frames = positions.shape[0]

    n_valid = int(np.sum(~np.isnan(positions).any(axis=(1, 2))))
    logger.info("  %s frames, %.1f%% with a valid (non-NaN) skeleton, fps=%.2f",
                total_frames, 100.0 * n_valid / max(total_frames, 1), fps)

    features, panel_columns = features_from_positions(positions, fps)

    metadata = {
        "source_npz": str(npz_file),
        "total_frames": total_frames,
        "fps": fps,
        "bone_length_edges": data["bone_length_edges"].tolist(),
        "bone_length_targets": data["bone_length_targets"].tolist(),
        "body_scale_m": float(data["body_scale_m"]),
        "gravity_aligned": bool(data["gravity_aligned"]),
        "panel_columns": panel_columns,
    }
    return metadata, features


def features_from_positions(positions: np.ndarray, fps: float) -> tuple[dict, dict]:
    """The reusable core of extract_features(): (T, 17, 3) positions + fps
    -> ({panel_key: torch.Tensor (T, n_cols)}, {panel_key: [column names]}).
    Split out so skeleton_pipeline.dataset.augment can recompute features on
    AUGMENTED positions (rotated/noised) without going back through an .npz
    file -- augmentation must happen on raw positions, before these
    (rotation/noise-sensitive) kinematic features are derived, not after."""
    import torch

    feature_dict, panel_groups = compute_all_features(positions, fps)

    features = {}
    for panel_title, columns in panel_groups.items():
        key = _slugify(panel_title)
        stacked = np.stack([feature_dict[col] for col in columns], axis=1)  # (T, n_cols)
        features[key] = torch.from_numpy(stacked.astype(np.float32))

    panel_columns = {_slugify(title): cols for title, cols in panel_groups.items()}
    return features, panel_columns


def to_feature_dict(features: dict, panel_columns: dict) -> tuple[dict, dict]:
    """Inverse of the stacking step in features_from_positions(): unpacks
    the {panel_key: (T, n_cols) tensor} dict a saved .pt's "features" holds
    back into a flat {column_name: (T,) ndarray} feature_dict + {panel_key:
    [column names]} panel_groups -- the shapes
    skeleton_pipeline.plotting.feature_plots.plot_panels() expects. Lets
    app/plot_features.py re-plot features straight from a .pt (including an
    AUGMENTED one, whose feature values differ from the source .npz's --
    hence plotting from the .pt, not re-deriving from the .npz again)."""
    feature_dict = {}
    panel_groups = {}
    for panel_key, tensor in features.items():
        columns = panel_columns[panel_key]
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        panel_groups[panel_key] = list(columns)
        for i, column in enumerate(columns):
            feature_dict[column] = array[:, i]
    return feature_dict, panel_groups


def _slugify(title: str) -> str:
    """"Joint Velocity X" -> "joint_velocity_x"; "Position X (relative to
    pelvis)" -> "position_x_relative_to_pelvis"."""
    title = re.sub(r"[()]", "", title)
    title = re.sub(r"[^0-9a-zA-Z]+", "_", title.strip().lower())
    return title.strip("_")
