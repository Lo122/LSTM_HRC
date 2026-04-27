
import os
import sys
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

if str(PROJECT_SRC_ROOT / "data_proc_2d") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT / "data_proc_2d"))
from utilities import log_utils
from src.annotation_config import ANNOTATION_CONFIG
from src.file_io_utils import load_torch, save_torch, iter_files
from src import feature_extraction
from src import labelling_utils
from src.yolo_pose_config import JOINT_ANGLE_TRIPLETS, JOINT_ANGLE_TRIPLETS_CAL, RATIO_BETWEEN_DISTS
from utilities.file_io import load_json
from src import plot_utils



label_config = labelling_utils.LabelConfiguration(
    buffer=100.0,
    function_type="bezier"
)

EXCLUDE_NODES = [13, 14, 15, 16]  # exclude knees and ankles for now due to frequent occlusion and noise in the dataset
EXCLUDE_STEPS = [7, 8, 9, 10, 11]
ACTIVE_LABEL_IDS = sorted(
    annotation_id
    for annotation_id in set(ANNOTATION_CONFIG.values())
    if annotation_id not in EXCLUDE_STEPS
)
SAVE_PLOTS = True
SHOW_PLOTS = False


def main():
    # setup logger and file paths
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("feature_extraction")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/feature_extraction.log"))
    
    # I/O paths
    google_drive_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
    pt_root = google_drive_path / "dataset" / "keypoints"
    pt_files = list(iter_files(pt_root, extension=".pt"))
    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)
    
    out_pt_dir = google_drive_path / "dataset" / "original"
    label_folder_root = google_drive_path / "annotations" / "cam-04"
    plot_folder_root = folder_root/ "data_proc_2d" / "results" / "feature_label_plots" 
    
    # main loop to process each .pt file and extract features
    for pt_file in pt_files:
        if "cam-04" not in pt_file.stem:
            logger.info("SKIP %s: not in cam-04 folder", pt_file)
            continue
        logger.info("Processing %s", pt_file.name)
        try:
            metadata, features = _extract_features(pt_file, logger, exclude_nodes=EXCLUDE_NODES)
            metadata['label_config'] = label_config.__dict__
            user_id = pt_file.stem.split("__")[1]  # extract user_id from file name
            labels = _extract_labels(user_id, label_folder_root,
                                     frame_size=metadata.get("total_frames", 0),
                                     exclude_steps=EXCLUDE_STEPS, logger=logger,
                                     save_plots=SAVE_PLOTS, show_plots=SHOW_PLOTS,
                                     save_file_path=plot_folder_root / f"label_plot__{user_id}.png",
                                     )
            output_data = {
                "metadata": metadata,
                "features": features,
                "labels": labels
            }
            # _save_features(output_data, out_pt_dir / pt_file.name, logger)

        except Exception as error:
            logger.exception("Failed to process %s: %s", pt_file, error)



def _extract_features(pt_file: Path, logger, exclude_nodes: list[int]) -> tuple[dict, dict]:

    data = load_torch(str(pt_file), logger=logger)
    
    metadata = data.get("metadata", {})
    landmarks = data["smoothed_landmarks"]         # (T, 17, 2)
    
    features = {}
    
    # clean up NaN, inf, zero or any malicious values for tranning models in landmarks
    landmarks = feature_extraction._log_malicious_tensor(landmarks, "landmarks")
    landmarks = feature_extraction._fill_zero_frames_with_previous(landmarks, "landmarks")
    
    # velocity and acceleration (magnitude)
    include_node_list = [i for i in range(landmarks.shape[1]) if i not in exclude_nodes]
    velocity_scale, acceleration_scale, velocity_xy, acceleration_xy = feature_extraction.veclocity_acceleration_magnitude(landmarks)
    features["velocity_scale"] = velocity_scale[:, include_node_list]
    features["acceleration_scale"] = acceleration_scale[:, include_node_list]
    features["velocity_xy"] = _flatten_features(velocity_xy[:, include_node_list, :])
    features["acceleration_xy"] = _flatten_features(acceleration_xy[:, include_node_list, :])
    
    
    vectors, angluer_velocity, distance_velocity, angles, distance = feature_extraction.polar_coordinate_features(landmarks[:, include_node_list, :])
    features["pol_vectors"] = _flatten_features(vectors)
    features["pol_distance"] = distance
    features["pol_angles"] = angles
    features["pol_distance_velocity"] = distance_velocity
    features["pol_angluer_velocity"] = angluer_velocity
        
    # angles at joints
    angles = []
    for _, a, vertex, c in JOINT_ANGLE_TRIPLETS:
        ang = feature_extraction.angle_at_joint(landmarks, a, vertex, c)  # (T,)
        angles.append(ang)
    
    for _, a, b, c in JOINT_ANGLE_TRIPLETS_CAL:
        middle_point = (landmarks[:, a, :] + landmarks[:, b, :]) / 2
        angle_values = feature_extraction.angle_at_joint(landmarks, a, middle_point, c)
        angles.append(angle_values)
        
    features["joint_angles"] = torch.stack(angles, dim=1)  # (T, num_angles)
    
    # ratios between distances of keypoint pairs
    distance_ratios = []
    for _, first_dist_feature, second_dist_feature in RATIO_BETWEEN_DISTS:
        ratio = feature_extraction.distance_ratio(landmarks, first_dist_feature, second_dist_feature)
        distance_ratios.append(ratio)
    features["ratios"] = torch.stack(distance_ratios, dim=1)  # (T, num_ratios)
    
    features['dist_ratios'] \
        = feature_extraction.distance_from_center(landmarks[:, include_node_list, :],
                                                ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
                                                ("left_hip", "right_hip"))
    
    return metadata, features
    

def _extract_labels(extract_user_id: str, folder_path: Path, frame_size: int,
                    exclude_steps: list, logger,
                    save_plots: bool = False, show_plots: bool = False,
                    save_file_path: Path = Path()) -> dict:

    json_path = folder_path / f"label__{extract_user_id}.json"
    detail_json_path = folder_path / f"label_detail__{extract_user_id}.json"
    step_id_data = load_json(str(json_path), logger)
    status_id_data = load_json(str(detail_json_path), logger)
    frame_index = pd.RangeIndex(frame_size)
    step_labels = step_id_data.get("labels", [])
    status_labels = status_id_data.get("labels", [])
    
    
    label_db = pd.DataFrame(index=frame_index)
    step_id_db, step_id_plateau_db = _build_label_variants(
        step_labels,
        frame_size,
        frame_index,
        exclude_steps=exclude_steps,
    )
    status_id_db, status_id_plateau_db = _build_label_variants(
        status_labels,
        frame_size,
        frame_index,
        exclude_steps=exclude_steps,
    )
    task_progress_db = _build_progress_matrix(
        status_labels,
        frame_size,
        frame_index,
        exclude_steps=exclude_steps,
    )
        
    
    # single step id label (num_frames,) int in [0, num_steps-1]
    label_db["step_id"] = _highest_value_label_per_frame(step_id_db, frame_index)
    label_db["step_id_prob"] = _highest_value_value_per_frame(step_id_db, frame_index)
    label_db["step_id_plateau"] = _highest_value_label_per_frame(step_id_plateau_db, frame_index)
    label_db["step_id_plateau_prob"] = _highest_value_value_per_frame(step_id_plateau_db, frame_index)
    
    
    # single status id label (num_frames,) int in [0, num_status-1]
    label_db["status_id"] = _highest_value_label_per_frame(status_id_db, frame_index)
    label_db["status_id_prob"] = _highest_value_value_per_frame(status_id_db, frame_index)
    label_db["status_id_plateau"] = _highest_value_label_per_frame(status_id_plateau_db, frame_index)
    label_db["status_id_plateau_prob"] = _highest_value_value_per_frame(status_id_plateau_db, frame_index)
    label_db["task_progress"] = _highest_value_value_per_frame(task_progress_db, frame_index)
    
    # convert dataframe to dict of tensors for training
    labels = {}
    labels['step_id'] = torch.from_numpy(label_db["step_id"].values.astype(int))
    labels['step_id_prob'] = torch.from_numpy(label_db["step_id_prob"].values.astype(float))
    labels['status_id'] = torch.from_numpy(label_db["status_id"].values.astype(int))
    labels['status_id_prob'] = torch.from_numpy(label_db["status_id_prob"].values.astype(float))
    labels['task_progress'] = torch.from_numpy(label_db["task_progress"].values.astype(float))

    
    labels['step_id_plateau'] = torch.from_numpy(label_db["step_id_plateau"].values.astype(int))
    labels['step_id_plateau_prob'] = torch.from_numpy(label_db["step_id_plateau_prob"].values.astype(float))
    labels['status_id_plateau'] = torch.from_numpy(label_db["status_id_plateau"].values.astype(int))
    labels['status_id_plateau_prob'] = torch.from_numpy(label_db["status_id_plateau_prob"].values.astype(float))
    
    labels['step_id_vector']  = torch.from_numpy(step_id_db.values.astype(float))
    labels['status_id_vector']  = torch.from_numpy(status_id_db.values.astype(float))
    
    labels['step_id_plateau_vector']  = torch.from_numpy(step_id_plateau_db.values.astype(float))
    labels['status_id_plateau_vector']  = torch.from_numpy(status_id_plateau_db.values.astype(float))

    labels['task_progress_vector']  = torch.from_numpy(task_progress_db.values.astype(float))

    
    # plot labels for debugging
    plot_utils._plot_label_debug(
        extract_user_id,
        step_id_db,
        step_id_plateau_db,
        status_id_db,
        status_id_plateau_db,
        task_progress_db,
        logger,
        show_debug_plots=show_plots,
        save_debug_plots=save_plots,
        save_file_path=save_file_path,
    )
    
    return labels


def _accumulate_label_series(
    label_db: pd.DataFrame,
    label_name: int,
    label_values: pd.Series,
    frame_index: pd.RangeIndex,
    max_value: float | None = None,
) -> pd.DataFrame:
    """Add one per-label frame series into the matching dataframe column."""
    result = label_db.copy()
    aligned_values = label_values.reindex(frame_index, fill_value=0.0)

    if label_name not in result.columns:
        result[label_name] = pd.Series(0.0, index=frame_index, dtype=float)

    result[label_name] = result[label_name].add(aligned_values, fill_value=0.0)
    if max_value is not None:
        result[label_name] = result[label_name].clip(upper=max_value)
    return result


def _build_label_variants(
    label_entries: list[dict],
    frame_size: int,
    frame_index: pd.RangeIndex,
    exclude_steps: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build asymmetric-peak and plateau label matrices for the same entries."""
    return (
        _build_label_matrix(
            label_entries,
            frame_size,
            frame_index,
            smooth_type="asymmetric_peak",
            exclude_steps=exclude_steps,
        ),
        _build_label_matrix(
            label_entries,
            frame_size,
            frame_index,
            smooth_type="plateau",
            exclude_steps=exclude_steps,
        ),
    )


def _build_label_matrix(
    label_entries: list[dict],
    frame_size: int,
    frame_index: pd.RangeIndex,
    smooth_type: str,
    exclude_steps: list | None = None,
) -> pd.DataFrame:
    """Build one label-score matrix for a specific smoothing mode."""
    label_db = pd.DataFrame(index=frame_index)

    for label_info in label_entries:
        step_id = label_info.get("step_id")
        start_frame = label_info.get("start_frame")
        end_frame = label_info.get("end_frame")
        if step_id is None or start_frame is None or end_frame is None:
            continue
        if exclude_steps is not None and step_id in exclude_steps:
            continue

        label_db = _accumulate_label_series(
            label_db,
            step_id,
            labelling_utils.define_step_label_entry(
                step_id,
                frame_size,
                start_frame,
                end_frame,
                buffer=label_config.buffer,
                smooth_type=smooth_type,
                function_type=label_config.function_type,
            ),
            frame_index,
            max_value=1.0,
        )

    return _ensure_db_columns(label_db, frame_index)


def _build_progress_matrix(
    label_entries: list[dict],
    frame_size: int,
    frame_index: pd.RangeIndex,
    exclude_steps: list | None = None,
) -> pd.DataFrame:
    """Build the per-step progress matrix from status label entries."""
    progress_db = pd.DataFrame(index=frame_index)
    grouped_entries: dict[tuple[int, int | None], list[dict]] = {}

    for label_info in label_entries:
        step_id = label_info.get("step_id")
        piece_id = label_info.get("piece_id")
        start_frame = label_info.get("start_frame")
        end_frame = label_info.get("end_frame")
        if step_id is None or start_frame is None or end_frame is None:
            continue
        if exclude_steps is not None and step_id in exclude_steps:
            continue

        group_key = (int(step_id), None if piece_id is None else int(piece_id))
        grouped_entries.setdefault(group_key, []).append(label_info)

    for (step_id, _piece_id), group_entries in grouped_entries.items():
        progress_db = _accumulate_label_series(
            progress_db,
            step_id,
            _build_progress_series_for_piece_group(
                group_entries,
                frame_size,
                frame_index,
            ),
            frame_index,
        )

    return _ensure_db_columns(progress_db, frame_index)


def _ensure_db_columns(label_db: pd.DataFrame, frame_index: pd.RangeIndex) -> pd.DataFrame:
    """Force a stable set of label columns so vectors are always (num_frames, 7)."""
    return label_db.reindex(index=frame_index, columns=ACTIVE_LABEL_IDS, fill_value=0.0)


def _build_progress_series_for_piece_group(
    label_entries: list[dict],
    frame_size: int,
    frame_index: pd.RangeIndex,
) -> pd.Series:
    """Build one progress series that carries the last progress through gaps."""
    progress_series = pd.Series(0.0, index=frame_index, dtype=float)
    previous_end_frame: int | None = None
    previous_end_progress = 0.0

    ordered_entries = sorted(
        label_entries,
        key=lambda entry: (
            float(entry.get("start_frame", 0.0)),
            float(entry.get("end_frame", 0.0)),
            float(entry.get("timestamp", 0.0)),
        ),
    )

    for label_info in ordered_entries:
        start_frame = int(round(float(label_info["start_frame"])))
        end_frame = int(round(float(label_info["end_frame"])))
        start_progress = float(label_info.get("start_progress", 0.0))
        end_progress = float(label_info.get("end_progress", 100.0))

        if previous_end_frame is not None and previous_end_progress < 100.0:
            gap_start = max(previous_end_frame + 1, 0)
            gap_end = min(start_frame - 1, frame_size - 1)
            if gap_start <= gap_end:
                progress_series.loc[gap_start:gap_end] = previous_end_progress

        # Some label segments share the boundary frame (end == next start). Clamp the
        # next segment to start on the following frame so task progress stays <= 100.
        effective_start_frame = start_frame
        if previous_end_frame is not None:
            effective_start_frame = max(start_frame, previous_end_frame + 1)

        if effective_start_frame > end_frame:
            continue

        segment_series = labelling_utils.cal_status_progress(
            label_info.get("step_id"),
            frame_size,
            effective_start_frame,
            end_frame,
            start_progress,
            end_progress,
        )
        progress_series.loc[effective_start_frame:end_frame] = segment_series.loc[effective_start_frame:end_frame]

        previous_end_frame = end_frame
        previous_end_progress = end_progress

    return progress_series


def _highest_value_label_per_frame(
    label_db: pd.DataFrame,
    frame_index: pd.RangeIndex,
) -> pd.Series:
    """Return the column label with the highest score for each frame."""
    if label_db.empty:
        return pd.Series(-1, index=frame_index, dtype="int64")

    return label_db.idxmax(axis=1).astype(int)


def _highest_value_value_per_frame(
    label_db: pd.DataFrame,
    frame_index: pd.RangeIndex,
) -> pd.Series:
    """Return the highest label value for each frame."""
    if label_db.empty:
        return pd.Series(0.0, index=frame_index, dtype="float64")

    return label_db.max(axis=1)





def _save_features(features: dict, file_path: Path, logger) -> None:
    
    output_pt = file_path.parent / _format_output_file_name(file_path.name)
    num_frames = features.get("num_frames", 0)
    save_torch(features, str(output_pt), logger=logger)
    logger.info("Saved %s frames to %s", num_frames, output_pt)



def _flatten_features(feature: torch.Tensor) -> torch.Tensor:
    """Flatten the last two dimensions of *feature* into one."""
    T = feature.shape[0]
    return feature.view(T, -1)


def _format_output_file_name(input_file_name: str) -> str:
    pattern = r"^keypoints__(cam-\d{2}_uid-\d{2}_take-\d{2})\.pt$"
    replacement = r"features__\1.pt"
    return re.sub(pattern, replacement, input_file_name)



if __name__ == "__main__":
    main()