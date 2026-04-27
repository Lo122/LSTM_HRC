import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parent
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))
from plot_utils import PanelData 
from yolo_pose_config import (
    KEYPOINT_NAMES,
    JOINT_ANGLE_TRIPLETS,
    JOINT_ANGLE_TRIPLETS_CAL,
    CENTER_CFG,
    RATIO_BETWEEN_DISTS,
)

# Plot Configuration 
XY_AXES = ("x", "y")


def build_feature_dataframes(features: dict) -> dict[str, pd.DataFrame]:
    feature_dataframes: dict[str, pd.DataFrame] = {}

    for feature_name, value in features.items():
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            array = value
        else:
            continue

        array = np.asarray(array)

        if array.ndim == 0:
            continue

        if array.ndim == 1:
            feature_dataframes[f"{feature_name}_db"] = pd.DataFrame({feature_name: array})
            continue

        part = pd.DataFrame(array.reshape(array.shape[0], -1))
        part.columns = _feature_column_names(feature_name, part.shape[1])
        feature_dataframes[f"{feature_name}_db"] = part

    return feature_dataframes


def _feature_column_names(feature_name: str, width: int) -> list[str]:
    named_columns = {
        "velocity_scale": _keypoint_feature_names("velocity"),
        "acceleration_scale": _keypoint_feature_names("acceleration"),
        "velocity_x": _keypoint_feature_names("velocity_x"),
        "velocity_y": _keypoint_feature_names("velocity_y"),
        "acceleration_x": _keypoint_feature_names("acceleration_x"),
        "acceleration_y": _keypoint_feature_names("acceleration_y"),
        "velocity_xy": _keypoint_xy_feature_names("velocity"),
        "acceleration_xy": _keypoint_xy_feature_names("acceleration"),
        "pol_vectors_x": _keypoint_feature_names("polar_vector_x"),
        "pol_vectors_y": _keypoint_feature_names("polar_vector_y"),
        "pol_vectors": _keypoint_xy_feature_names("polar_vector"),
        "pol_angles": _keypoint_feature_names("polar_angle_deg"),
        "joint_angles": _joint_angle_feature_names(),
        "ratios": _ratio_feature_names(),
        "dist_ratios": _keypoint_feature_names("distance_from_center_ratio"),
    }
    columns = named_columns.get(feature_name)
    if columns is None or len(columns) != width:
        return [f"{feature_name}_{index}" for index in range(width)]
    return columns


def _sanitize_feature_label(name: str) -> str:
    return name.lower().replace("/", "_over_").replace(" ", "_").replace("-", "_")


def _keypoint_feature_names(suffix: str) -> list[str]:
    return [f"{keypoint}_{suffix}" for keypoint in KEYPOINT_NAMES]


def _keypoint_xy_feature_names(suffix: str) -> list[str]:
    return [
        f"{keypoint}_{suffix}__{axis}"
        for keypoint in KEYPOINT_NAMES
        for axis in XY_AXES
    ]


def _joint_angle_feature_names() -> list[str]:
    joint_names = [name for name, *_ in JOINT_ANGLE_TRIPLETS]
    joint_names.extend(name for name, *_ in JOINT_ANGLE_TRIPLETS_CAL)
    return [f"{_sanitize_feature_label(name)}_angle_deg" for name in joint_names]


def _ratio_feature_names() -> list[str]:
    return [
        f"{_sanitize_feature_label(name)}_ratio"
        for name, *_ in RATIO_BETWEEN_DISTS
    ]



def build_panel_data(
    feature_dataframes: dict[str, pd.DataFrame], panel_config: list[tuple[str, str, str, str]]
) -> list[PanelData]:
    panel_data: list[PanelData] = []

    for dataframe_name, ylabel, panel_title, yscale in panel_config:
        feature_df = feature_dataframes.get(dataframe_name)
        if feature_df is None or feature_df.empty:
            continue

        columns = list(feature_df.columns)
        panel_data.append(PanelData(df=feature_df, names=columns, cols=columns, ylabel=ylabel, title=panel_title, yscale=yscale))

    return panel_data































def _coerce_kinematic_tensor(values: torch.Tensor, feature_name: str) -> torch.Tensor:
    """Validate a stored kinematic tensor before it is used downstream."""
    if values.ndim != 2:
        raise ValueError(
            f"Expected {feature_name} to be a 2-D tensor, got shape {tuple(values.shape)}."
        )
    if values.shape[1] != len(KEYPOINT_NAMES):
        raise ValueError(
            f"Expected {feature_name} to have {len(KEYPOINT_NAMES)} joints, got {values.shape[1]}."
        )
    return values


def _resolve_speed_tensor(
    landmarks: torch.Tensor,
    speed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return stored joint speed when available, otherwise derive it from landmarks."""
    if speed is not None:
        return _coerce_kinematic_tensor(speed, "speed")

    velocity = landmarks[1:] - landmarks[:-1]
    return velocity.norm(dim=2)


def _resolve_acceleration_tensor(
    landmarks: torch.Tensor,
    acceleration: torch.Tensor | None = None,
    speed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return stored joint acceleration when available, otherwise derive it."""
    if acceleration is not None:
        return _coerce_kinematic_tensor(acceleration, "acceleration").abs()

    speed_tensor = _resolve_speed_tensor(landmarks, speed)
    if speed_tensor.shape[0] == landmarks.shape[0]:
        if speed_tensor.shape[0] <= 1:
            return torch.zeros_like(speed_tensor)
        accel = (speed_tensor[1:] - speed_tensor[:-1]).abs()
        pad = torch.zeros((1, accel.shape[1]), dtype=accel.dtype, device=accel.device)
        return torch.cat([pad, accel], dim=0)

    return (speed_tensor[1:] - speed_tensor[:-1]).abs()


def _summarize_kinematic_tensor(values: torch.Tensor) -> tuple[dict[str, float], float, float]:
    """Compute per-joint mean plus overall mean/max for a 2-D kinematic tensor."""
    if values.numel() == 0:
        zeros = {name: 0.0 for name in KEYPOINT_NAMES}
        return zeros, 0.0, 0.0

    mean_per_joint = {
        KEYPOINT_NAMES[i]: round(float(values[:, i].mean()), 6)
        for i in range(len(KEYPOINT_NAMES))
    }
    return mean_per_joint, round(float(values.mean()), 6), round(float(values.max()), 6)


def _frame_numbers(base_frames: int, feature_frames: int, feature_name: str) -> np.ndarray:
    """Infer frame numbers for tensors aligned to either T, T-1, or T-2 frames."""
    candidates: list[tuple[int, int]] = [(base_frames, 0)]
    if feature_name == "speed":
        candidates.append((max(base_frames - 1, 0), 1))
    elif feature_name == "acceleration":
        candidates.extend([
            (max(base_frames - 1, 0), 1),
            (max(base_frames - 2, 0), 2),
        ])
    else:
        raise ValueError(f"Unknown feature name '{feature_name}'.")

    for expected_frames, start_frame in candidates:
        if feature_frames == expected_frames:
            return np.arange(start_frame, start_frame + feature_frames)

    raise ValueError(
        f"Cannot align {feature_name} with {feature_frames} frames to landmarks with {base_frames} frames."
    )


def _tensor_to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


_KEYPOINT_INDEX_BY_NAME = {name: index for index, name in enumerate(KEYPOINT_NAMES)}



def extract_features(
    landmarks: torch.Tensor,
    speed: torch.Tensor | None = None,
    acceleration: torch.Tensor | None = None,
) -> dict:
    """Extract pose features from a sequence of normalised keypoints.

    Args:
        landmarks:     (T, 17, 2) tensor of normalised 2-D keypoints.
        speed:         optional stored joint speed tensor.
        acceleration:  optional stored joint acceleration tensor.

    Returns:
        Dictionary with velocity statistics, joint-angle statistics, and
        configured distance-ratio statistics.
    """
    T = landmarks.shape[0]

    speed_tensor = _resolve_speed_tensor(landmarks, speed)
    mean_speed_per_joint, body_speed_mean, body_speed_max = _summarize_kinematic_tensor(speed_tensor)

    accel_tensor = _resolve_acceleration_tensor(landmarks, acceleration, speed_tensor)
    mean_accel_per_joint, body_accel_mean, body_accel_max = _summarize_kinematic_tensor(accel_tensor)

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

    distance_ratios: dict = {}
    for name, first_dist_feature, second_dist_feature in RATIO_BETWEEN_DISTS:
        ratio = _distance_ratio(landmarks, first_dist_feature, second_dist_feature)
        distance_ratios[name] = {
            "mean": round(float(ratio.mean()), 4),
            "std":  round(float(ratio.std()), 4),
            "min":  round(float(ratio.min()), 4),
            "max":  round(float(ratio.max()), 4),
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
        "distance_ratios":       distance_ratios,
    }
    
    
def _angle_at_joint(
    kpts: torch.Tensor,
    a: int,
    vertex: int | torch.Tensor,
    c: int,
) -> torch.Tensor:
    """Compute the angle (degrees) at *vertex* across T frames.

    Args:
        kpts:   (T, 17, 2) normalised keypoints.
        a:      index of the first arm keypoint.
        vertex: index of the joint at which the angle is measured, or a
                per-frame tensor with shape ``(T, 2)``.
        c:      index of the second arm keypoint.

    Returns:
        (T,) tensor of angles in degrees.
    """
    if isinstance(vertex, int):
        vertex_points = kpts[:, vertex, :]
    elif torch.is_tensor(vertex):
        if vertex.ndim != 2 or vertex.shape != (kpts.shape[0], 2):
            raise ValueError(
                "Expected vertex tensor to have shape "
                f"({kpts.shape[0]}, 2), got {tuple(vertex.shape)}."
            )
        vertex_points = vertex.to(dtype=kpts.dtype, device=kpts.device)
    else:
        raise TypeError(
            "vertex must be either an integer keypoint index or a tensor "
            f"with shape ({kpts.shape[0]}, 2), got {type(vertex).__name__}."
        )

    va = kpts[:, a, :] - vertex_points   # (T, 2)
    vc = kpts[:, c, :] - vertex_points   # (T, 2)
    dot  = (va * vc).sum(dim=1)                    # (T,)
    norm = va.norm(dim=1) * vc.norm(dim=1) + 1e-8  # (T,)
    cos_angle = (dot / norm).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).rad2deg()          # (T,)


def _distance_between_keypoints(
    kpts: torch.Tensor,
    point_a_name: str,
    point_b_name: str,
) -> torch.Tensor:
    """Compute per-frame Euclidean distance between two named keypoints."""
    try:
        point_a_idx = _KEYPOINT_INDEX_BY_NAME[point_a_name]
        point_b_idx = _KEYPOINT_INDEX_BY_NAME[point_b_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown keypoint name in ratio config: {error.args[0]!r}."
        ) from error

    delta = kpts[:, point_a_idx, :] - kpts[:, point_b_idx, :]
    return delta.norm(dim=1)


def _distance_ratio(
    kpts: torch.Tensor,
    first_dist_feature: tuple[str, str],
    second_dist_feature: tuple[str, str],
) -> torch.Tensor:
    """Compute per-frame ratio between two configured segment distances."""
    first_distance = _distance_between_keypoints(kpts, *first_dist_feature)
    second_distance = _distance_between_keypoints(kpts, *second_dist_feature)
    return first_distance / second_distance.clamp_min(1e-8)


# ============================================================
# Per-frame feature computation  →  DataFrame / CSV
# ============================================================

def compute_joint_speeds_df(
    landmarks: torch.Tensor,
    speed: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Return per-frame joint speed as a DataFrame.

    If a stored speed tensor is provided, it is used directly. Otherwise speed
    is derived from the landmark sequence.

    Args:
        landmarks: (T, 17, 2) normalised keypoints.
        speed:     optional stored joint speed tensor.

    Returns:
        DataFrame with either T or T-1 rows, depending on input alignment.
    """
    speed_tensor = _resolve_speed_tensor(landmarks, speed)
    speed_array = _tensor_to_numpy(speed_tensor)
    data = {"frame": _frame_numbers(landmarks.shape[0], speed_array.shape[0], "speed")}
    for i, name in enumerate(KEYPOINT_NAMES):
        data[f"{name}_speed"] = speed_array[:, i]
    return pd.DataFrame(data)


_CENTER_CFG = CENTER_CFG


def compute_joint_acceleration_df(
    landmarks: torch.Tensor,
    acceleration: torch.Tensor | None = None,
    speed: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Return per-frame joint acceleration magnitude as a DataFrame.

    If a stored acceleration tensor is provided, it is used directly.
    Otherwise acceleration is derived from the provided speed tensor or,
    failing that, from landmarks.

    Args:
        landmarks:     (T, 17, 2) normalised keypoints.
        acceleration:  optional stored joint acceleration tensor.
        speed:         optional stored joint speed tensor.

    Returns:
        DataFrame with T, T-1, or T-2 rows, depending on input alignment.
    """
    accel_tensor = _resolve_acceleration_tensor(landmarks, acceleration, speed)
    accel_array = _tensor_to_numpy(accel_tensor)
    data = {
        "frame": _frame_numbers(landmarks.shape[0], accel_array.shape[0], "acceleration")
    }
    for i, name in enumerate(KEYPOINT_NAMES):
        data[f"{name}_accel"] = accel_array[:, i]
    return pd.DataFrame(data)


def compute_joint_angles_df(landmarks: torch.Tensor) -> pd.DataFrame:
    """Compute per-frame joint angles and return them as a DataFrame.

    Columns: ``frame``, ``{joint}_angle_deg`` for each configured joint angle.

    Args:
        landmarks: (T, 17, 2) normalised keypoints.

    Returns:
        DataFrame with shape (T, 1 + len(JOINT_ANGLE_TRIPLETS)).
    """
    data = {"frame": np.arange(landmarks.shape[0])}
    for name, a, vertex, c in JOINT_ANGLE_TRIPLETS:
        angle_values = _angle_at_joint(landmarks, a, vertex, c)
        data[f"{name}_angle_deg"] = _tensor_to_numpy(angle_values)
    
    for name, a, b, c in JOINT_ANGLE_TRIPLETS_CAL:
        middle_point = (landmarks[:, a, :] + landmarks[:, b, :]) / 2
        angle_values = _angle_at_joint(landmarks, a, middle_point, c)
        data[f"{name}_angle_deg"] = _tensor_to_numpy(angle_values)
    
    return pd.DataFrame(data)


def compute_distance_ratios_df(landmarks: torch.Tensor) -> pd.DataFrame:
    """Compute per-frame configured distance ratios and return them as a DataFrame.

    Columns: ``frame``, ``{ratio_name}_ratio`` for each configured entry in
    :data:`RATIO_BETWEEN_DISTS`.

    Args:
        landmarks: (T, 17, 2) normalised keypoints.

    Returns:
        DataFrame with shape (T, 1 + len(RATIO_BETWEEN_DISTS)).
    """
    data = {"frame": np.arange(landmarks.shape[0])}
    for name, first_dist_feature, second_dist_feature in RATIO_BETWEEN_DISTS:
        ratio_values = _distance_ratio(landmarks, first_dist_feature, second_dist_feature)
        data[f"{name}_ratio"] = _tensor_to_numpy(ratio_values)
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
    angle_df: pd.DataFrame | None = None,
    ratio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all per-frame pose DataFrames on the ``frame`` column.

    Because speed, acceleration, and distance tensors may use different frame
    alignments, an outer join is used and missing values are left as ``NaN``.

    Column groups in the result:
        * ``frame``
        * ``{joint}_speed``          × 17
        * ``{joint}_accel``          × 17
        * ``{joint}_dist_hip``       × 17
        * ``{joint}_dist_shoulder``  × 17
        * ``{joint}_dist_half_body`` × 17
        * ``{joint}_angle_deg``      × number of configured joint angles
        * ``{ratio_name}_ratio``     × number of configured distance ratios

    Args:
        speed_df:         output of :func:`compute_joint_speeds_df`.
        accel_df:         output of :func:`compute_joint_acceleration_df`.
        dist_hip_df:      output of :func:`compute_joint_distances_df` with ``center=\"hip\"``.
        dist_shoulder_df: output of :func:`compute_joint_distances_df` with ``center=\"shoulder\"``.
        dist_half_df:     output of :func:`compute_joint_distances_df` with ``center=\"half_body\"``.
        angle_df:         optional output of :func:`compute_joint_angles_df`.
        ratio_df:         optional output of :func:`compute_distance_ratios_df`.

    Returns:
        Merged DataFrame sorted by ``frame``.
    """
    dfs = [speed_df, accel_df, dist_hip_df, dist_shoulder_df, dist_half_df]
    if angle_df is not None:
        dfs.append(angle_df)
    if ratio_df is not None:
        dfs.append(ratio_df)
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


