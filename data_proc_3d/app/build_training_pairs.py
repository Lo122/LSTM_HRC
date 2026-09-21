"""Build LSTM training-data .pt files (features + labels) for cam-04 takes
from this project's 3D pipeline, mirroring the schema of data_proc_2d's
existing .pt files (e.g. ".../Videos/dataset/original/
features__cam-04_uid-01_take-01.pt": a dict with "metadata" / "features" /
"labels" keys, torch.save'd) -- see data_proc_2d/app/build_training_pairs.py,
which this file is the 3D counterpart of, following the same app/ (thin
orchestrator) vs src/ (logic) split data_proc_2d/data_proc_3d both use.

Pipeline this reads from (must already have been run -- see
generate_lstm_training_data.py):
    video (raw/cam-04/*.mp4)
      -> YOLO 2D pose -> MotionBERT 2D->3D lift -> bone-length stabilize
      -> data_proc_3d/results/video__cam-04_uid-XX_take-XX.npz
         (keypoints_3d: (T, 17, 3) root-relative H36M skeleton, meters)

This script then, per take:
  1. loads its .npz and computes kinematic features
     (skeleton_pipeline.dataset.feature_io.extract_features, itself a thin
     wrapper over skeleton_pipeline.features.h36m_features.compute_all_features),
  2. loads the matching label__uid-XX_take-YY.json annotation (written by
     app/proc_annotations.py from the ELAN exports) and turns it into smooth
     per-frame label tensors (skeleton_pipeline.dataset.labels.extract_labels),
  3. trims the leading/trailing frames that have NO skeleton at all (see
     trim_empty_edges -- a take can start or end with minutes of video the
     detector never found the subject in, while the annotations still cover
     that span, which pairs empty features with real labels),
  4. torch.save's one {"metadata", "features", "labels"} dict per take
     (skeleton_pipeline.dataset.io_utils.save_torch),
  5. writes two plots per take into PLOT_DIR: label_plot__* (the label curves
     alone) and overview__* (features and labels on one shared time axis).

See skeleton_pipeline/dataset/labels.py's module docstring for the
reimplementation note on why its label-smoothing math isn't a port of
data_proc_2d's (lost) labelling_utils.py.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skeleton_pipeline.dataset import io_utils, labels, feature_io
from skeleton_pipeline.plotting.label_plots import plot_label_debug
from skeleton_pipeline.plotting.feature_plots import plot_overview_with_labels


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOOGLE_DRIVE_ROOT = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
NPZ_ROOT = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation" / "raw_npz"
ANNOTATIONS_DIR = GOOGLE_DRIVE_ROOT / "annotations" / "ceiling_installation"
OUT_PT_DIR =  GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation_03" / "original"
PLOT_DIR = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation_03" / "label_plots"
LOG_PATH = PROJECT_ROOT / "logs" / "build_training_pairs_3d.log"

# The trailing "-N" is a take split across two annotation files
# (video__cam-06_uid-02_take-03-1.npz); without it those 8 takes are silently
# skipped rather than built.
NPZ_FILE_PATTERN = re.compile(r"^video__(cam-\d{2}_uid-\d{2}_take-\d{2}(?:-\d+)?)\.npz$")

SAVE_PLOTS = True
SHOW_PLOTS = False





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
    h36m_features' own per-run handling (see _valid_runs) deal with them.

    min_valid_fraction: raise if the surviving span is still mostly empty --
    such a take is better fixed (re-annotated, re-tracked, or dropped) than
    quietly trained on.

    Returns (metadata, features, label_tensors), trimmed in place of the
    originals. metadata records source_total_frames/trim_start/trim_stop so
    the .pt still says which part of the original video it came from.
    """
    import torch  # local: keeps this module importable without torch for --help

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
        logger.warning("  %s interior frame(s) still have no skeleton -- left as NaN on "
                       "purpose, see trim_empty_edges.", interior_empty)

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


def plot_feature_label_overview(take_id, metadata, features, label_tensors):
    """Write PLOT_DIR/overview__<take_id>.png: the stacked feature rows with the
    three model targets -- task ribbon, mistake flag, task_progress -- sharing
    their time axis. Shows the collapsed LABEL_MAP_DICT tasks, not the 15 raw
    annotation tiers, so what is plotted is what will be trained on."""
    feature_dict, panel_groups = feature_io.to_feature_dict(
        features, metadata["panel_columns"])
    fps = metadata["fps"]
    # Offset by the trim so the x axis stays REAL video time -- otherwise a
    # trimmed take's plot restarts at 0 s and no longer lines up with the
    # source video or the ELAN annotations when cross-checking.
    trim_start = metadata.get("trim_start", 0)
    timestamps = (trim_start + np.arange(metadata["total_frames"])) / fps

    return plot_overview_with_labels(
        feature_dict, panel_groups, timestamps, PLOT_DIR, take_id,
        # The per-task vector, not the argmax scalar: tasks overlap in time and
        # one lane each is the only way to see that.
        label_vector=label_tensors["task_id_plateau_vector"].numpy(),
        label_names=[labels.TASK_NAMES[task_id] for task_id in labels.TASK_IDS],
        task_progress=label_tensors["task_progress"].numpy(),
        mistake=label_tensors["mistake"].numpy(),
        fps=fps,
        output_name=f"overview__{take_id}.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=str(OUT_PT_DIR),
                        help="Where the features__*.pt land. Point it somewhere "
                             "local to try a run without writing to the shared Drive.")
    parser.add_argument("--plot-dir", type=str, default=str(PLOT_DIR))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only build the first N takes -- for a quick check.")
    parser.add_argument("--no-causal-features", dest="causal_features", action="store_false",
                         help="Compute features with CENTERED Savitzky-Golay windows instead of "
                              "trailing ones. Not recommended: a centered window at feature "
                              "window_length=9 uses 4 frames of lookahead (400 ms at the 10 Hz "
                              "the live loop runs at), which skeleton3d_pipeline.py's streaming "
                              "extractor cannot reproduce -- the model would be trained on an "
                              "estimator deployment does not have. See skeleton_pipeline/features/"
                              "h36m_features.py's _savgol_derivative.")
    parser.set_defaults(causal_features=True)
    parser.add_argument("--no-trim-empty", dest="trim_empty", action="store_false",
                        help="Keep leading/trailing frames that have no skeleton. Off by "
                             "default because those frames are all-NaN yet still carry task "
                             "labels, and a NaN propagates through an LSTM sequence -- see "
                             "trim_empty_edges.")
    parser.set_defaults(trim_empty=True)
    parser.add_argument("--min-valid-fraction", type=float, default=0.0,
                        help="Refuse to build a take if, after trimming, less than this "
                             "fraction of its frames has a skeleton (e.g. 0.8). Default 0 = "
                             "build everything and just report. Use it to fail loudly on "
                             "takes whose tracking needs fixing.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip the per-take plots (they dominate the runtime).")
    return parser.parse_args()


def main():
    global PLOT_DIR
    args = parse_args()
    out_pt_dir = Path(args.output_dir)
    PLOT_DIR = Path(args.plot_dir)
    save_plots = SAVE_PLOTS and not args.no_plots

    logger = io_utils.setup_logger("build_training_pairs_3d")
    io_utils.save_log_to_file(logger, str(LOG_PATH))

    npz_files = sorted(p for p in NPZ_ROOT.glob("*.npz") if NPZ_FILE_PATTERN.match(p.name))
    if args.limit:
        npz_files = npz_files[:args.limit]
    logger.info("Found %s .npz file(s) under %s", len(npz_files), NPZ_ROOT)
    logger.info("Writing .pt -> %s", out_pt_dir)

    built = failed = 0
    trimmed_frames = 0
    for npz_file in npz_files:
        logger.info("Processing %s", npz_file.name)
        try:
            take_id = NPZ_FILE_PATTERN.match(npz_file.name).group(1)  # e.g. "cam-04_uid-01_take-01"
            annotation_data_id = "_".join(take_id.split("_")[1:])
            metadata, features = feature_io.extract_features(
                npz_file, logger, causal=args.causal_features)
            metadata["label_config"] = labels.DEFAULT_LABEL_CONFIG.__dict__

            label_tensors, debug_frames = labels.extract_labels(
                annotation_data_id, ANNOTATIONS_DIR, frame_size=metadata["total_frames"], logger=logger)

            # AFTER extract_labels (which needs the full frame count to line the
            # annotations up against the video timeline) and BEFORE the plots, so
            # what gets plotted is what gets saved.
            if args.trim_empty:
                metadata, features, label_tensors = trim_empty_edges(
                    take_id, metadata, features, label_tensors, logger,
                    min_valid_fraction=args.min_valid_fraction)
                trimmed_frames += metadata["source_total_frames"] - metadata["total_frames"]

            if save_plots or SHOW_PLOTS:
                plot_label_debug(take_id, debug_frames, logger, show=SHOW_PLOTS, save=save_plots,
                                  save_path=PLOT_DIR / f"label_plot__{take_id}.png")
            if save_plots:
                # Features and labels on one time axis -- the label curves alone
                # (plot_label_debug above) show what was annotated, this shows it
                # against the motion it was annotated from.
                overview_path = plot_feature_label_overview(take_id, metadata, features,
                                                            label_tensors)
                logger.info("  Saved feature+label overview -> %s", overview_path)

            metadata["task_names"] = labels.TASK_NAMES
            metadata["label_map"] = labels.LABEL_MAP_DICT
            output_data = {"metadata": metadata, "features": features, "labels": label_tensors}
            io_utils.save_torch(output_data, out_pt_dir / f"features__{take_id}.pt", logger=logger)
            built += 1

        except Exception:
            failed += 1
            logger.exception("Failed to process %s", npz_file)

    logger.info("Built %s take(s), %s failed -> %s", built, failed, out_pt_dir)
    if args.trim_empty:
        logger.info("Trimmed %s empty edge frame(s) across all takes.", trimmed_frames)


if __name__ == "__main__":
    main()
