"""Task-progress label handling for the training-data pipeline
(app/build_training_pairs.py): turns the raw per-piece
label__cam-04_uid-XX_take-YY.json / label_detail__...json annotation
entries (step_id, piece_id, start_frame, end_frame[, start_progress,
end_progress] -- produced by data_proc_2d/app/proc_annotations.py from
ELAN exports, see that script + data_proc_2d/src/file_io_utils.py's
load_elan_label_data/_assign_group_progress_ranges) into smooth per-frame
label tensors.

*** Reimplementation note ***: data_proc_2d/app/build_training_pairs.py (the
2D counterpart of this pipeline) built these same label tensors via
labelling_utils.LabelConfiguration/define_step_label_entry/cal_status_progress,
which no longer exist in this repo (only unrecoverable .pyc bytecode
remains -- same situation skeleton_pipeline/features/h36m_features.py's own
docstring documents and works around for the lost 2D *feature* code). So
_define_step_label_entry()/_cal_status_progress() below are a fresh,
self-contained reimplementation (smoothstep/"bezier" ramps) against the
still-present raw label JSON schema, functionally mirroring
build_training_pairs.py's own still-readable orchestration
(_build_label_matrix/_build_progress_matrix/etc., kept close to here) --
not a byte-for-byte port of the lost helpers. Cross-check against the
original if it resurfaces.

ANNOTATION_CONFIG below is copied from data_proc_2d/src/annotation_config.py
(that file DOES still exist) rather than imported cross-project, since
data_proc_3d/app deliberately runs in its own separate, minimal venv (see
generate_lstm_training_data.py's module docstring) -- keep the two in sync
by hand if the label taxonomy changes.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils

# Mirrors data_proc_2d/src/annotation_config.py's ANNOTATION_CONFIG values
# (ELAN tier label -> step_id). The "- BL" (baseline?) variants 7-11 are
# excluded from training, same as data_proc_2d/app/build_training_pairs.py's
# EXCLUDE_STEPS.
ANNOTATION_STEP_IDS = {
    "Place Spacer": 0,
    "Move Spacer": 1,
    "Remove spacer": 2,
    "Align the Piece": 3,
    "Place the Piece": 4,
    "Screw": 5,
    "Mistake": 6,
    "Place Spacer - BL": 7,
    "Remove Spacer - BL": 8,
    "Align - BL": 9,
    "Place the Piece - BL": 10,
    "Screw - BL": 11,
}
EXCLUDE_STEPS = [step_id for step_id in ANNOTATION_STEP_IDS.values() if step_id >= 7]
ACTIVE_LABEL_IDS = sorted(set(ANNOTATION_STEP_IDS.values()) - set(EXCLUDE_STEPS))


@dataclass
class LabelConfiguration:
    """Mirrors data_proc_2d/app/build_training_pairs.py's
    labelling_utils.LabelConfiguration(buffer=100.0, function_type="bezier")."""
    buffer: float = 100.0          # ramp-in/out width around each labeled span, in frames
    function_type: str = "bezier"  # only "bezier" (smoothstep) is implemented below


DEFAULT_LABEL_CONFIG = LabelConfiguration()


def extract_labels(
    take_id: str,
    annotations_dir: Path,
    frame_size: int,
    logger,
    label_config: LabelConfiguration = DEFAULT_LABEL_CONFIG,
    exclude_steps: list | None = None,
) -> dict:
    """Builds the same label set data_proc_2d/app/build_training_pairs.py's
    _extract_labels() did, from label__{take_id}.json / label_detail__{take_id}.json
    under *annotations_dir* (take_id e.g. "cam-04_uid-01_take-01"). Returns a
    dict of torch tensors: step_id/status_id(_plateau)(_prob), task_progress,
    and their (T, len(ACTIVE_LABEL_IDS)) *_vector soft-label forms."""
    import torch

    exclude_steps = EXCLUDE_STEPS if exclude_steps is None else exclude_steps

    step_id_data = io_utils.load_json(annotations_dir / f"label__{take_id}.json", logger)
    status_id_data = io_utils.load_json(annotations_dir / f"label_detail__{take_id}.json", logger)
    frame_index = pd.RangeIndex(frame_size)
    step_labels = step_id_data.get("labels", []) if step_id_data else []
    status_labels = status_id_data.get("labels", []) if status_id_data else []

    label_db = pd.DataFrame(index=frame_index)
    step_id_db, step_id_plateau_db = _build_label_variants(
        step_labels, frame_size, frame_index, label_config, exclude_steps)
    status_id_db, status_id_plateau_db = _build_label_variants(
        status_labels, frame_size, frame_index, label_config, exclude_steps)
    task_progress_db = _build_progress_matrix(status_labels, frame_size, frame_index, exclude_steps)

    label_db["step_id"] = _highest_value_label_per_frame(step_id_db, frame_index)
    label_db["step_id_prob"] = _highest_value_value_per_frame(step_id_db, frame_index)
    label_db["step_id_plateau"] = _highest_value_label_per_frame(step_id_plateau_db, frame_index)
    label_db["step_id_plateau_prob"] = _highest_value_value_per_frame(step_id_plateau_db, frame_index)

    label_db["status_id"] = _highest_value_label_per_frame(status_id_db, frame_index)
    label_db["status_id_prob"] = _highest_value_value_per_frame(status_id_db, frame_index)
    label_db["status_id_plateau"] = _highest_value_label_per_frame(status_id_plateau_db, frame_index)
    label_db["status_id_plateau_prob"] = _highest_value_value_per_frame(status_id_plateau_db, frame_index)
    label_db["task_progress"] = _highest_value_value_per_frame(task_progress_db, frame_index)

    labels = {
        "step_id": torch.from_numpy(label_db["step_id"].values.astype(np.int64)),
        "step_id_prob": torch.from_numpy(label_db["step_id_prob"].values.astype(np.float32)),
        "status_id": torch.from_numpy(label_db["status_id"].values.astype(np.int64)),
        "status_id_prob": torch.from_numpy(label_db["status_id_prob"].values.astype(np.float32)),
        "task_progress": torch.from_numpy(label_db["task_progress"].values.astype(np.float32)),
        "step_id_plateau": torch.from_numpy(label_db["step_id_plateau"].values.astype(np.int64)),
        "step_id_plateau_prob": torch.from_numpy(label_db["step_id_plateau_prob"].values.astype(np.float32)),
        "status_id_plateau": torch.from_numpy(label_db["status_id_plateau"].values.astype(np.int64)),
        "status_id_plateau_prob": torch.from_numpy(label_db["status_id_plateau_prob"].values.astype(np.float32)),
        "step_id_vector": torch.from_numpy(step_id_db.values.astype(np.float32)),
        "status_id_vector": torch.from_numpy(status_id_db.values.astype(np.float32)),
        "step_id_plateau_vector": torch.from_numpy(step_id_plateau_db.values.astype(np.float32)),
        "status_id_plateau_vector": torch.from_numpy(status_id_plateau_db.values.astype(np.float32)),
        "task_progress_vector": torch.from_numpy(task_progress_db.values.astype(np.float32)),
    }

    debug_frames = {
        "step_id (asymmetric_peak)": step_id_db,
        "step_id (plateau)": step_id_plateau_db,
        "status_id (asymmetric_peak)": status_id_db,
        "status_id (plateau)": status_id_plateau_db,
        "task_progress": task_progress_db,
    }
    return labels, debug_frames


# ---------------------------------------------------------------------------
# Smoothing primitives
# ---------------------------------------------------------------------------
def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Cubic ease (3t^2-2t^3), t clipped to [0, 1] first -- the "bezier"
    ramp shape LabelConfiguration.function_type refers to."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _define_step_label_entry(frame_size: int, start_frame: float, end_frame: float,
                              buffer: float, smooth_type: str) -> pd.Series:
    """Returns a (frame_size,) score curve in [0, 1] for one labeled span.

    - "plateau": smoothstep ramp 0->1 over [start-buffer, start], flat 1
      over [start, end], smoothstep ramp 1->0 over [end, end+buffer].
    - "asymmetric_peak": same ramps but meeting at a single apex (the
      span's temporal midpoint) instead of a flat top -- avoids a tie
      across the whole plateau when _highest_value_label_per_frame later
      picks one label per frame via idxmax.
    """
    frames = np.arange(frame_size, dtype=np.float64)
    start_frame = float(start_frame)
    end_frame = float(end_frame)
    buffer = max(float(buffer), 1e-6)

    if smooth_type == "plateau":
        ramp_up = _smoothstep((frames - (start_frame - buffer)) / buffer)
        ramp_down = _smoothstep(((end_frame + buffer) - frames) / buffer)
        values = np.minimum(ramp_up, ramp_down)
        values = np.where((frames >= start_frame) & (frames <= end_frame), 1.0, values)
    elif smooth_type == "asymmetric_peak":
        apex = (start_frame + end_frame) / 2.0
        up_span = max(apex - (start_frame - buffer), 1e-6)
        down_span = max((end_frame + buffer) - apex, 1e-6)
        ramp_up = _smoothstep((frames - (start_frame - buffer)) / up_span)
        ramp_down = _smoothstep(((end_frame + buffer) - frames) / down_span)
        values = np.where(frames <= apex, ramp_up, ramp_down)
    else:
        raise ValueError(f"Unknown smooth_type: {smooth_type!r}")

    values = np.clip(values, 0.0, 1.0)
    values[(frames < start_frame - buffer) | (frames > end_frame + buffer)] = 0.0
    return pd.Series(values, index=pd.RangeIndex(frame_size))


def _cal_status_progress(frame_size: int, start_frame: float, end_frame: float,
                          start_progress: float, end_progress: float) -> pd.Series:
    """(frame_size,) series: smoothstep-interpolated progress in
    [start_progress, end_progress] over [start_frame, end_frame];
    start_progress before the span, end_progress after it."""
    frames = np.arange(frame_size, dtype=np.float64)
    start_frame = float(start_frame)
    end_frame = float(end_frame)
    span = max(end_frame - start_frame, 1e-6)
    t = _smoothstep((frames - start_frame) / span)
    values = start_progress + t * (end_progress - start_progress)
    values = np.where(frames < start_frame, start_progress, values)
    values = np.where(frames > end_frame, end_progress, values)
    return pd.Series(values, index=pd.RangeIndex(frame_size))


# ---------------------------------------------------------------------------
# Label-matrix building (mirrors data_proc_2d/app/build_training_pairs.py)
# ---------------------------------------------------------------------------
def _accumulate_label_series(label_db: pd.DataFrame, label_name: int, label_values: pd.Series,
                              frame_index: pd.RangeIndex, max_value: float | None = None) -> pd.DataFrame:
    result = label_db.copy()
    aligned_values = label_values.reindex(frame_index, fill_value=0.0)
    if label_name not in result.columns:
        result[label_name] = pd.Series(0.0, index=frame_index, dtype=float)
    result[label_name] = result[label_name].add(aligned_values, fill_value=0.0)
    if max_value is not None:
        result[label_name] = result[label_name].clip(upper=max_value)
    return result


def _build_label_variants(label_entries: list[dict], frame_size: int, frame_index: pd.RangeIndex,
                           label_config: LabelConfiguration,
                           exclude_steps: list | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        _build_label_matrix(label_entries, frame_size, frame_index, label_config,
                             smooth_type="asymmetric_peak", exclude_steps=exclude_steps),
        _build_label_matrix(label_entries, frame_size, frame_index, label_config,
                             smooth_type="plateau", exclude_steps=exclude_steps),
    )


def _build_label_matrix(label_entries: list[dict], frame_size: int, frame_index: pd.RangeIndex,
                         label_config: LabelConfiguration, smooth_type: str,
                         exclude_steps: list | None) -> pd.DataFrame:
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
            label_db, step_id,
            _define_step_label_entry(frame_size, start_frame, end_frame,
                                      buffer=label_config.buffer, smooth_type=smooth_type),
            frame_index, max_value=1.0,
        )
    return _ensure_db_columns(label_db, frame_index)


def _build_progress_matrix(label_entries: list[dict], frame_size: int, frame_index: pd.RangeIndex,
                            exclude_steps: list | None) -> pd.DataFrame:
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
            progress_db, step_id,
            _build_progress_series_for_piece_group(group_entries, frame_size, frame_index),
            frame_index,
        )
    return _ensure_db_columns(progress_db, frame_index)


def _build_progress_series_for_piece_group(label_entries: list[dict], frame_size: int,
                                            frame_index: pd.RangeIndex) -> pd.Series:
    """One progress series that carries the last progress through gaps
    between consecutive entries of the same (step_id, piece_id) -- e.g.
    label_detail's split "0-50% then 51-100%" entries for one piece."""
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

        effective_start_frame = start_frame
        if previous_end_frame is not None:
            effective_start_frame = max(start_frame, previous_end_frame + 1)
        if effective_start_frame > end_frame:
            continue

        segment_series = _cal_status_progress(
            frame_size, effective_start_frame, end_frame, start_progress, end_progress)
        progress_series.loc[effective_start_frame:end_frame] = segment_series.loc[effective_start_frame:end_frame]

        previous_end_frame = end_frame
        previous_end_progress = end_progress

    return progress_series


def _ensure_db_columns(label_db: pd.DataFrame, frame_index: pd.RangeIndex) -> pd.DataFrame:
    """Forces a stable set of label columns so vectors are always
    (num_frames, len(ACTIVE_LABEL_IDS))."""
    return label_db.reindex(index=frame_index, columns=ACTIVE_LABEL_IDS, fill_value=0.0)


def _highest_value_label_per_frame(label_db: pd.DataFrame, frame_index: pd.RangeIndex) -> pd.Series:
    if label_db.empty:
        return pd.Series(-1, index=frame_index, dtype="int64")
    return label_db.idxmax(axis=1).astype(int)


def _highest_value_value_per_frame(label_db: pd.DataFrame, frame_index: pd.RangeIndex) -> pd.Series:
    if label_db.empty:
        return pd.Series(0.0, index=frame_index, dtype="float64")
    return label_db.max(axis=1)
