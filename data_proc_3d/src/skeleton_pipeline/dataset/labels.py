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
# (ELAN tier label -> step_id).
#
# "M - X" is a MISTAKE variant of step X: the subject performing that step
# incorrectly. It is not a second person and not a separate activity, so each
# one pairs with its correct counterpart ("M - Screw" <-> "Screw", "M - Lift
# Main" <-> "Lifting Main", and so on; "Pull the Cables - 2" has no variant).
#
# The ids are NOT contiguous by theme, and 6 sorts away from the other
# "Pull the Cables", so read this dict rather than assuming ranges.
#
# (An older data_proc_2d taxonomy had "- BL" (body language) variants at ids
# 7-11 which this file's comment used to say were excluded. Those tiers are
# no longer annotated and no longer appear above; ids 7-11 are now Lifting
# Main and four mistake variants, and they are NOT excluded. EXCLUDE_STEPS
# below drops only "T Pose" (15), the calibration pose.)
ANNOTATION_STEP_IDS = {
    "Pull the Cables - 1": 0,
    "Pull the Cables - 2": 6,
     "Lift": 1,
     "Align": 2,
     "Screw": 3,
     "Connect Cables": 4,
     "Clamping": 5,
     "Lifting Main": 7,
     "M - Pull the Cables": 8,
     "M - Lift": 9,
     "M - Align": 10,
     "M - Screw": 11,
     "M - Connect Cables": 12,
     "M - Lift Main": 13,
     "M - Clamping": 14,
     "T Pose": 15,
}
EXCLUDE_STEPS = [step_id for step_id in ANNOTATION_STEP_IDS.values() if step_id >= 15]
ACTIVE_LABEL_IDS = sorted(set(ANNOTATION_STEP_IDS.values()) - set(EXCLUDE_STEPS))

# The model's actual output taxonomy, collapsing the annotation tiers above.
# {output_id: {name: [annotation step_ids that map to it]}}.
#
# Two heads, not one: entries 0-6 are the mutually exclusive TASK classes, and
# the last entry is the separate binary MISTAKE flag. A mistake is not a class
# of its own -- "M - Screw" is still the Screw task, done wrong -- so each M-
# tier appears twice here: once under its task, once under Mistake. That is why
# the flag needs its own output rather than an eighth class: predicting "Screw"
# and "this is a mistake" are different questions about the same frame.
LABEL_MAP_DICT = {
    0: {"Pull Cables": [0, 6, 8]},
    1: {"Lift": [1, 9]},
    2: {"Align": [2, 10]},
    3: {"Screw": [3, 11]},
    4: {"Connect Cables": [4, 12]},
    5: {"Clamp Coupling": [5, 14]},
    6: {"Place": [7, 13]},
    7: {"Mistake": [8, 9, 10, 11, 12, 13, 14]}
}
MISTAKE_ENTRY_NAME = "Mistake"


def _unpack_label_map(label_map: dict) -> tuple[dict, dict, set]:
    """LABEL_MAP_DICT -> ({task_id: name}, {step_id: task_id}, {mistake step_ids})."""
    task_names, step_to_task, mistake_step_ids = {}, {}, set()
    for output_id, entry in label_map.items():
        (name, step_ids), = entry.items()
        if name == MISTAKE_ENTRY_NAME:
            mistake_step_ids = set(step_ids)
            continue
        task_names[output_id] = name
        for step_id in step_ids:
            step_to_task[step_id] = output_id
    return task_names, step_to_task, mistake_step_ids


TASK_NAMES, STEP_TO_TASK, MISTAKE_STEP_IDS = _unpack_label_map(LABEL_MAP_DICT)
TASK_IDS = sorted(TASK_NAMES)

# Every annotated step the pipeline keeps must land in a task, or its frames
# would silently train as "no task" while still carrying real motion.
_UNMAPPED_STEPS = sorted(set(ACTIVE_LABEL_IDS) - set(STEP_TO_TASK))
if _UNMAPPED_STEPS:
    raise ValueError(
        f"LABEL_MAP_DICT does not map annotation step id(s) {_UNMAPPED_STEPS}; "
        f"every id in ACTIVE_LABEL_IDS ({ACTIVE_LABEL_IDS}) needs a task.")



def to_task_entries(label_entries: list[dict], exclude_steps: list | None = None) -> list[dict]:
    """Annotation entries -> model-output entries, via LABEL_MAP_DICT.

    Each returned entry gains "task_id" (the collapsed class), "is_mistake",
    and "mistake_id" (0 for mistakes, absent otherwise -- the single column of
    the binary mistake matrix).

    Mistake spans also get their progress REVERSED, 100 -> 0, where a correct
    span runs 0 -> 100. A mistake undoes work rather than advancing it, so
    within one (task, piece) the curve falls back while the error is being made
    and climbs again over the redo. Both land on the same task column, which is
    the point of folding "M - Screw" into "Screw": the progress signal for a
    piece stays continuous across the mistake instead of splitting into two
    unrelated classes.
    """
    exclude_steps = EXCLUDE_STEPS if exclude_steps is None else exclude_steps
    task_entries = []
    for entry in label_entries:
        step_id = entry.get("step_id")
        if step_id is None or step_id in exclude_steps:
            continue
        task_id = STEP_TO_TASK.get(int(step_id))
        if task_id is None:
            continue

        is_mistake = int(step_id) in MISTAKE_STEP_IDS
        task_entry = dict(entry)
        task_entry["task_id"] = task_id
        task_entry["is_mistake"] = is_mistake
        if is_mistake:
            task_entry["mistake_id"] = 0
            task_entry["start_progress"] = float(entry.get("end_progress", 100.0))
            task_entry["end_progress"] = float(entry.get("start_progress", 0.0))
        task_entries.append(task_entry)
    return task_entries


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
    """Per-frame label tensors from label__{take_id}.json under
    *annotations_dir* (take_id e.g. "uid-01_take-01").

    Returns (labels, debug_frames). The model's three targets are:

      task_id        (T,)  int64, one of TASK_IDS -- LABEL_MAP_DICT's classes
      mistake        (T,)  int64 0/1, the separate binary flag
      task_progress  (T,)  float 0-100 within the frame's task, counting DOWN
                           across a mistake (see to_task_entries)

    each with a soft *_prob companion and, for task_id, a
    (T, len(TASK_IDS)) *_vector of per-class scores for soft-label training.
    The *_plateau variants score a flat 1.0 across the whole annotated span
    rather than peaking at its midpoint. The raw step_id* tensors (the 15
    un-collapsed annotation tiers) come along for debugging and plots; they are
    not model outputs."""
    import torch

    exclude_steps = EXCLUDE_STEPS if exclude_steps is None else exclude_steps

    step_id_data = io_utils.load_json(annotations_dir / f"label__{take_id}.json", logger)
    frame_index = pd.RangeIndex(frame_size)
    step_labels = step_id_data.get("labels", []) if step_id_data else []

    task_labels = to_task_entries(step_labels, exclude_steps)
    mistake_labels = [entry for entry in task_labels if entry["is_mistake"]]

    label_db = pd.DataFrame(index=frame_index)
    step_id_db, step_id_plateau_db = _build_label_variants(
        step_labels, frame_size, frame_index, label_config, exclude_steps)
    task_id_db, task_id_plateau_db = _build_label_variants(
        task_labels, frame_size, frame_index, label_config, exclude_steps,
        label_key="task_id", columns=TASK_IDS)
    mistake_db, mistake_plateau_db = _build_label_variants(
        mistake_labels, frame_size, frame_index, label_config, exclude_steps,
        label_key="mistake_id", columns=[0])
    task_progress_db = _build_progress_matrix(
        task_labels, frame_size, frame_index, exclude_steps,
        label_key="task_id", columns=TASK_IDS)

    label_db["step_id"] = _highest_value_label_per_frame(step_id_db, frame_index)
    label_db["step_id_prob"] = _highest_value_value_per_frame(step_id_db, frame_index)
    label_db["step_id_plateau"] = _highest_value_label_per_frame(step_id_plateau_db, frame_index)
    label_db["step_id_plateau_prob"] = _highest_value_value_per_frame(step_id_plateau_db, frame_index)

    label_db["task_id"] = _highest_value_label_per_frame(task_id_db, frame_index)
    label_db["task_id_prob"] = _highest_value_value_per_frame(task_id_db, frame_index)
    label_db["task_id_plateau"] = _highest_value_label_per_frame(task_id_plateau_db, frame_index)
    label_db["task_id_plateau_prob"] = _highest_value_value_per_frame(task_id_plateau_db, frame_index)

    label_db["mistake_prob"] = mistake_db[0]
    label_db["mistake"] = (mistake_plateau_db[0] >= 0.5).astype(np.int64)

    # Progress of the task that task_id names for that frame, NOT the largest
    # progress over all tasks: 21.5% of labelled frames have two steps annotated
    # at once (Lift + Lifting Main, Lift + Align, ...), and taking the max there
    # reports a concurrent task's near-100% while the selected one has barely
    # started.
    label_db["task_progress"] = _value_at_selected_label(
        task_progress_db, label_db["task_id"], frame_index)

    labels = {
        # --- model outputs ---------------------------------------------------
        "task_id": torch.from_numpy(label_db["task_id"].values.astype(np.int64)),
        "task_id_prob": torch.from_numpy(label_db["task_id_prob"].values.astype(np.float32)),
        "task_id_vector": torch.from_numpy(task_id_db.values.astype(np.float32)),
        "mistake": torch.from_numpy(label_db["mistake"].values.astype(np.int64)),
        "mistake_prob": torch.from_numpy(label_db["mistake_prob"].values.astype(np.float32)),
        "task_progress": torch.from_numpy(label_db["task_progress"].values.astype(np.float32)),
        "task_progress_vector": torch.from_numpy(task_progress_db.values.astype(np.float32)),
        "task_id_plateau": torch.from_numpy(label_db["task_id_plateau"].values.astype(np.int64)),
        "task_id_plateau_prob": torch.from_numpy(label_db["task_id_plateau_prob"].values.astype(np.float32)),
        "task_id_plateau_vector": torch.from_numpy(task_id_plateau_db.values.astype(np.float32)),
        # --- raw annotation tiers, kept for debugging/plots ------------------
        "step_id": torch.from_numpy(label_db["step_id"].values.astype(np.int64)),
        "step_id_prob": torch.from_numpy(label_db["step_id_prob"].values.astype(np.float32)),
        "step_id_plateau": torch.from_numpy(label_db["step_id_plateau"].values.astype(np.int64)),
        "step_id_plateau_prob": torch.from_numpy(label_db["step_id_plateau_prob"].values.astype(np.float32)),
        "step_id_vector": torch.from_numpy(step_id_db.values.astype(np.float32)),
        "step_id_plateau_vector": torch.from_numpy(step_id_plateau_db.values.astype(np.float32)),
    }

    debug_frames = {
        "task_id (asymmetric_peak)": task_id_db.rename(columns=TASK_NAMES),
        "task_id (plateau)": task_id_plateau_db.rename(columns=TASK_NAMES),
        "mistake": mistake_plateau_db.rename(columns={0: "mistake"}),
        "task_progress": task_progress_db.rename(columns=TASK_NAMES),
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
                           label_config: LabelConfiguration, exclude_steps: list | None,
                           label_key: str = "step_id",
                           columns: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(asymmetric_peak, plateau) score matrices. *label_key* names the entry
    field that picks a column -- "step_id" for the raw annotation tiers,
    "task_id" for the collapsed model classes -- and *columns* is the full
    column set to force, so the matrix width never depends on which labels a
    given take happens to contain."""
    return (
        _build_label_matrix(label_entries, frame_size, frame_index, label_config,
                             smooth_type="asymmetric_peak", exclude_steps=exclude_steps,
                             label_key=label_key, columns=columns),
        _build_label_matrix(label_entries, frame_size, frame_index, label_config,
                             smooth_type="plateau", exclude_steps=exclude_steps,
                             label_key=label_key, columns=columns),
    )


def _build_label_matrix(label_entries: list[dict], frame_size: int, frame_index: pd.RangeIndex,
                         label_config: LabelConfiguration, smooth_type: str,
                         exclude_steps: list | None, label_key: str = "step_id",
                         columns: list | None = None) -> pd.DataFrame:
    label_db = pd.DataFrame(index=frame_index)
    for label_info in label_entries:
        column = label_info.get(label_key)
        start_frame = label_info.get("start_frame")
        end_frame = label_info.get("end_frame")
        if column is None or start_frame is None or end_frame is None:
            continue
        if exclude_steps is not None and label_info.get("step_id") in exclude_steps:
            continue
        label_db = _accumulate_label_series(
            label_db, column,
            _define_step_label_entry(frame_size, start_frame, end_frame,
                                      buffer=label_config.buffer, smooth_type=smooth_type),
            frame_index, max_value=1.0,
        )
    return _ensure_db_columns(label_db, frame_index, columns)


def _build_progress_matrix(label_entries: list[dict], frame_size: int, frame_index: pd.RangeIndex,
                            exclude_steps: list | None, label_key: str = "step_id",
                            columns: list | None = None) -> pd.DataFrame:
    progress_db = pd.DataFrame(index=frame_index)
    grouped_entries: dict[tuple[int, int | None], list[dict]] = {}
    for label_info in label_entries:
        column = label_info.get(label_key)
        piece_id = label_info.get("piece_id")
        start_frame = label_info.get("start_frame")
        end_frame = label_info.get("end_frame")
        if column is None or start_frame is None or end_frame is None:
            continue
        if exclude_steps is not None and label_info.get("step_id") in exclude_steps:
            continue
        group_key = (int(column), None if piece_id is None else int(piece_id))
        grouped_entries.setdefault(group_key, []).append(label_info)

    for (column, _piece_id), group_entries in grouped_entries.items():
        # Groups sharing a column land on the same curve. Their spans are
        # disjoint apart from a handful of annotations whose end frame is the
        # next one's start frame, so clip rather than let those single frames
        # sum past a full 100%.
        progress_db = _accumulate_label_series(
            progress_db, column,
            _build_progress_series_for_piece_group(group_entries, frame_size, frame_index),
            frame_index, max_value=100.0,
        )
    return _ensure_db_columns(progress_db, frame_index, columns)


def _build_progress_series_for_piece_group(label_entries: list[dict], frame_size: int,
                                            frame_index: pd.RangeIndex) -> pd.Series:
    """One progress series for all entries sharing a (step_id, piece_id).

    Each entry ramps start_progress -> end_progress across its own frames and
    the series is 0 outside every entry, so with proc_annotations.py's default
    --progress-mode per-label -- where every annotation carries a full 0..100 --
    each labelled span rises 0% -> 100% on its own and resets between spans.

    The carry-forward below only engages when an entry stops short of 100,
    which is what --progress-mode per-group produces: repeats of one
    (step_id, piece_id) split a single 0..100 range, and the progress reached
    so far must hold through the gaps between them instead of dropping to 0.

    Mistake entries are applied last, OVERWRITING their own frames rather than
    queueing after the entries already placed. A mistake is usually annotated
    INSIDE the span of the task it spoils -- "M - Lift" 26.5-43.6s sits within
    "Lift" 9.7-66.8s -- and the sequential pass treats each entry as following
    the previous one, so a nested entry's effective start is pushed past its own
    end and it is dropped. Overwriting is also the right semantics: during the
    mistake the task is being undone, whatever the enclosing span said.
    """
    progress_series = pd.Series(0.0, index=frame_index, dtype=float)
    previous_end_frame: int | None = None
    previous_end_progress = 0.0

    def sort_key(entry):
        return (float(entry.get("start_frame", 0.0)),
                float(entry.get("end_frame", 0.0)),
                float(entry.get("timestamp", 0.0)))

    ordered_entries = sorted(
        (e for e in label_entries if not e.get("is_mistake")), key=sort_key)
    mistake_entries = sorted(
        (e for e in label_entries if e.get("is_mistake")), key=sort_key)

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

    for label_info in mistake_entries:
        start_frame = int(round(float(label_info["start_frame"])))
        end_frame = int(round(float(label_info["end_frame"])))
        if start_frame > end_frame:
            continue
        segment_series = _cal_status_progress(
            frame_size, start_frame, end_frame,
            float(label_info.get("start_progress", 100.0)),
            float(label_info.get("end_progress", 0.0)))
        progress_series.loc[start_frame:end_frame] = segment_series.loc[start_frame:end_frame]

    return progress_series


def _ensure_db_columns(label_db: pd.DataFrame, frame_index: pd.RangeIndex,
                        columns: list | None = None) -> pd.DataFrame:
    """Forces a stable set of label columns so a take's vectors always have the
    same width, whichever labels it happens to contain."""
    columns = ACTIVE_LABEL_IDS if columns is None else columns
    return label_db.reindex(index=frame_index, columns=columns, fill_value=0.0)


def _highest_value_label_per_frame(label_db: pd.DataFrame, frame_index: pd.RangeIndex) -> pd.Series:
    if label_db.empty:
        return pd.Series(-1, index=frame_index, dtype="int64")
    return label_db.idxmax(axis=1).astype(int)


def _value_at_selected_label(value_db: pd.DataFrame, selected_labels: pd.Series,
                              frame_index: pd.RangeIndex) -> pd.Series:
    """Per frame, the value_db entry for the label named in *selected_labels*.

    Used to read one step's progress out of the per-step progress matrix, so
    the scalar follows the step the frame is classified as rather than whichever
    step happens to be furthest along.
    """
    if value_db.empty:
        return pd.Series(0.0, index=frame_index, dtype="float64")
    column_position = {label: position for position, label in enumerate(value_db.columns)}
    positions = selected_labels.map(column_position).to_numpy()
    values = value_db.to_numpy()
    picked = np.where(
        pd.isna(positions), 0.0,
        values[np.arange(len(values)), np.nan_to_num(positions, nan=0).astype(int)])
    return pd.Series(picked, index=frame_index, dtype="float64")


def _highest_value_value_per_frame(label_db: pd.DataFrame, frame_index: pd.RangeIndex) -> pd.Series:
    if label_db.empty:
        return pd.Series(0.0, index=frame_index, dtype="float64")
    return label_db.max(axis=1)
