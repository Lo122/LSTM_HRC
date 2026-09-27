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
     skeleton_pipeline.dataset.cleanup.trim_empty_edges -- a take can start or
     end with minutes of video the detector never found the subject in, while
     the annotations still cover that span, pairing empty features with real
     labels),
  4. fills the NaNs that survive that trim -- interior gaps, runs too short for
     the feature windows, degenerate columns -- with the last known value of
     their own column (cleanup.fill_nan_last_known; a single NaN would otherwise
     make that column's dataset-wide mean/std NaN and normalization would
     spread it over every frame of every take),
  5. torch.save's one {"metadata", "features", "labels"} dict per take
     (skeleton_pipeline.dataset.io_utils.save_torch),
  6. writes two plots per take into PLOT_DIR: label_plot__* (the label curves
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
from skeleton_pipeline.dataset import cleanup, io_utils, labels, feature_io
from skeleton_pipeline.plotting.label_plots import plot_label_debug
from skeleton_pipeline.plotting.feature_plots import plot_overview_with_labels


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOOGLE_DRIVE_ROOT = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
NPZ_ROOT = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation" / "raw_npz"
ANNOTATIONS_DIR = GOOGLE_DRIVE_ROOT / "annotations" / "ceiling_installation"
OUT_PT_DIR =  GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation_05" / "original"
PLOT_DIR = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation_05" / "label_plots"
LOG_PATH = PROJECT_ROOT / "logs" / "build_training_pairs_3d.log"

# The trailing "-N" is a take split across two annotation files
# (video__cam-06_uid-02_take-03-1.npz); without it those 8 takes are silently
# skipped rather than built.
NPZ_FILE_PATTERN = re.compile(r"^video__(cam-\d{2}_uid-\d{2}_take-\d{2}(?:-\d+)?)\.npz$")

SAVE_PLOTS = True
SHOW_PLOTS = False





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
    parser.add_argument("--no-fill-nan", dest="fill_nan", action="store_false",
                        help="Leave the NaNs that survive trimming (interior gaps, short "
                             "runs, degenerate columns) in the saved features. Off by "
                             "default because a single NaN makes that column's dataset-wide "
                             "mean/std NaN, which normalization then spreads over every "
                             "frame of every take -- see fill_nan_last_known.")
    parser.set_defaults(fill_nan=True)
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
    trimmed_frames = filled_frames = 0
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
                metadata, features, label_tensors = cleanup.trim_empty_edges(
                    take_id, metadata, features, label_tensors, logger,
                    min_valid_fraction=args.min_valid_fraction)
                trimmed_frames += metadata["source_total_frames"] - metadata["total_frames"]

            # AFTER the trim (so the long empty edges are gone rather than held
            # for minutes) and still BEFORE the plots, so the overview shows the
            # filled values that actually get saved.
            if args.fill_nan:
                metadata, features = cleanup.fill_nan_last_known(
                    take_id, metadata, features, logger)
                filled_frames += metadata["nan_filled_frames"]

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
    if args.fill_nan:
        logger.info("Filled NaNs on %s frame(s) across all takes with the last known value.",
                    filled_frames)


if __name__ == "__main__":
    main()
