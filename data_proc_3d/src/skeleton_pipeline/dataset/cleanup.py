"""Make a take's features safe to normalize and to feed an LSTM: drop the
empty edges, then hold whatever NaN is left at its last known value.

Both steps exist for the same reason -- the pose pipeline emits NaN rows
wherever it found no skeleton (see feature_io / h36m_features), and a NaN is
not a neutral value anywhere downstream of here:

  - trim_empty_edges() deletes the leading/trailing span with no skeleton at
    all, which a take can open or close with for minutes at a time,
  - fill_nan_last_known() forward-fills whatever NaN survives that (short
    interior gaps, degenerate columns) with a zero-order hold.

This lives in src/ rather than inside one app/ script because both app/
build_training_pairs.py (the original takes) and app/augment_dataset.py (the
mirrored/rotated/noised copies of those same takes) have to clean their
features IDENTICALLY. An augmented .pt built without these steps puts back
into the training set exactly the NaNs the originals were cleaned of, and one
such file is enough to make a dataset-wide normalization NaN.
"""
import numpy as np


def trim_empty_edges(take_id, metadata, features, label_tensors, logger,
                      min_valid_fraction=0.0):
    """Drop the leading and trailing frames that have no skeleton at all.

    WHY. A take can open or close with a long span the detector never found
    the subject in -- the subject has not walked on yet, or the camera kept
    rolling after they left. The annotations, written against the video
    timeline, still cover that span. The result is feature rows that are
    entirely NaN paired with real task/mistake/progress labels. Measured over
    this dataset: 2.7% of all frames have no skeleton, and almost all of it is
    at the edges (only 55 interior frames in ~1.06M), but it is concentrated
    brutally -- cam-07_uid-10_take-01 is 50.9% empty (trailing) and
    cam-07_uid-06_take-02 is 47.1% empty (the first 195.8 s), of which 4881
    empty frames sit inside a labelled task.

    Those rows are worse than useless. A NaN anywhere in an LSTM input
    propagates through every later timestep of that sequence, so a training
    window overlapping the empty span does not merely contribute nothing --
    it destroys its own gradient, while the label says a task was underway.

    ONLY THE EDGES are trimmed, deliberately. Interior gaps are left as NaN
    rather than deleted: removing an interior frame would splice two
    non-adjacent moments together, and the velocity/acceleration features are
    computed from consecutive frames, so the splice would manufacture a huge
    fake motion at the joint. Leaving them NaN keeps time contiguous and lets
    h36m_features' own per-run handling (see _valid_runs) deal with them; what
    is still NaN after this trim is then held at its last known value by
    fill_nan_last_known, which runs next.

    min_valid_fraction: raise if the surviving span is still mostly empty --
    such a take is better fixed (re-annotated, re-tracked, or dropped) than
    quietly trained on.

    Returns (metadata, features, label_tensors), trimmed in place of the
    originals. metadata records source_total_frames/trim_start/trim_stop so
    the .pt still says which part of the original video it came from.
    """
    panels = [v.numpy() for v in features.values()]
    stacked = np.concatenate(panels, axis=1)          # (T, sum(n_cols))
    empty = np.isnan(stacked).all(axis=1)
    total = len(empty)

    valid_idx = np.flatnonzero(~empty)
    if valid_idx.size == 0:
        raise ValueError(f"{take_id}: no frame has a skeleton at all -- nothing to train on.")

    start, stop = int(valid_idx[0]), int(valid_idx[-1]) + 1
    kept = stop - start
    if start == 0 and stop == total:
        logger.info("  No empty edges to trim (%s frames).", total)
    else:
        fps = metadata["fps"]
        logger.warning(
            "  Trimmed empty edges: %s -> %s frames (dropped %s leading = %.1fs, "
            "%s trailing = %.1fs).", total, kept, start, start / fps,
            total - stop, (total - stop) / fps)

    interior_empty = int(np.count_nonzero(empty[start:stop]))
    if interior_empty:
        logger.warning("  %s interior frame(s) still have no skeleton -- left as NaN here "
                       "on purpose (see trim_empty_edges), then held at the last known "
                       "value by fill_nan_last_known.", interior_empty)

    valid_fraction = (kept - interior_empty) / kept
    if valid_fraction < min_valid_fraction:
        raise ValueError(
            f"{take_id}: only {valid_fraction:.1%} of the trimmed span has a skeleton "
            f"(below --min-valid-fraction={min_valid_fraction:.0%}). Refusing to build it; "
            "re-check the tracking (review_tracks.py / --target-tracks) for this take.")

    if start != 0 or stop != total:
        features = {key: tensor[start:stop] for key, tensor in features.items()}
        # Every label tensor is per-frame on dim 0 (task_id, *_vector, progress,
        # mistake, step_*), so one slice keeps them aligned with the features.
        # Asserted rather than assumed -- a future non-per-frame label would
        # otherwise be silently corrupted here.
        for key, tensor in label_tensors.items():
            assert tensor.shape[0] == total, (
                f"{take_id}: label {key!r} has {tensor.shape[0]} rows, expected {total} -- "
                "trim_empty_edges assumes every label tensor is per-frame.")
        label_tensors = {key: tensor[start:stop] for key, tensor in label_tensors.items()}

    metadata["source_total_frames"] = total
    metadata["trim_start"] = start
    metadata["trim_stop"] = stop
    metadata["total_frames"] = kept
    metadata["interior_empty_frames"] = interior_empty
    return metadata, features, label_tensors


def fill_nan_last_known(take_id, metadata, features, logger):
    """Forward-fill every remaining NaN in the features with the last known
    value of its own column (a zero-order hold).

    WHY. trim_empty_edges only removes the empty EDGES; interior gaps -- a
    frame the detector lost the subject in, or a valid run too short for
    h36m_features' Savitzky-Golay window (see _valid_runs) -- survive as NaN,
    and so do individual degenerate columns (a bone ratio whose shoulder width
    measured ~0, an angle between two near-zero vectors). Those NaNs poison
    everything downstream: a per-column mean/std computed across the dataset
    for normalization comes out NaN for that whole column, and (x - mean) / std
    then makes EVERY frame of EVERY take NaN -- one lost frame in one take is
    enough to destroy the entire normalized dataset. Holding the last value
    keeps the row finite and, over a short gap, states what the subject was
    doing better than a zero (which reads as "standing perfectly still at the
    origin") or a column mean (which invents motion that never happened).

    Leading NaNs have no earlier value to hold, so they are back-filled with
    the column's FIRST valid value instead -- the mirror of the same hold. A
    column that is NaN for the whole take has nothing to hold at all: it is
    filled with 0.0 and logged as a warning, because that is a feature bug
    worth looking at, not something to paper over quietly.

    Deliberately NOT gap-length limited: trim_empty_edges has already removed
    the long empty spans, so what reaches here is short (measured over this
    dataset: 55 interior frames in ~1.06M). metadata records how much was
    filled, so a .pt still says which part of it is held rather than measured.
    """
    import torch  # local: keeps this module importable without torch installed

    filled_values = 0
    filled_frames = 0
    dead_columns = []

    filled_features = {}
    for key, tensor in features.items():
        array = tensor.numpy().copy()
        nan_mask = np.isnan(array)
        if not nan_mask.any():
            filled_features[key] = tensor
            continue

        total_frames, n_cols = array.shape
        columns = np.arange(n_cols)

        # Forward fill: per column, the row index of the most recent non-NaN
        # value at or before this row. It stays 0 while no valid value has been
        # seen yet, so a leading NaN run still reads NaN here -- the back fill
        # below is what resolves those.
        source_row = np.where(~nan_mask, np.arange(total_frames)[:, None], 0)
        np.maximum.accumulate(source_row, axis=0, out=source_row)
        array = array[source_row, columns[None, :]]

        # Back fill each column's leading NaN run with its first valid value.
        for col in columns[np.isnan(array).any(axis=0)]:
            valid_rows = np.flatnonzero(~nan_mask[:, col])
            if valid_rows.size == 0:
                dead_columns.append((key, int(col)))
                array[:, col] = 0.0
            else:
                array[:valid_rows[0], col] = array[valid_rows[0], col]

        filled_values += int(np.count_nonzero(nan_mask))
        filled_frames += int(np.count_nonzero(nan_mask.any(axis=1)))
        filled_features[key] = torch.from_numpy(array.astype(np.float32))

    for key, col in dead_columns:
        logger.warning("  %s: feature %r (panel %s) is NaN for the WHOLE take -- filled "
                       "with 0.0. Nothing to hold; check h36m_features for this column.",
                       take_id, metadata["panel_columns"][key][col], key)

    if filled_values:
        logger.warning("  Filled %s NaN value(s) over %s frame(s) with the last known value "
                       "(see fill_nan_last_known).", filled_values, filled_frames)
    else:
        logger.info("  No NaN left to fill.")

    metadata["nan_filled_values"] = filled_values
    metadata["nan_filled_frames"] = filled_frames
    metadata["nan_filled_dead_columns"] = [metadata["panel_columns"][key][col]
                                           for key, col in dead_columns]
    return metadata, filled_features
