"""Per-panel feature plots for already-built .pt takes -- the same
skeleton_pipeline.plotting.feature_plots.plot_panels() that
generate_lstm_training_data.py/debug_visualize.py use straight off the
.npz (raw compute_all_features() output), but reading from a saved
{"metadata", "features", "labels"} .pt instead. That matters for anything
downstream of build_training_pairs.py -- augment_dataset.py's rotated/noised
copies have DIFFERENT feature values than the source .npz's own plots
would show (that's the point of the augmentation), and segment_data.py's
chunks aren't full clips at all -- so re-deriving plots from the .npz
wouldn't show what's actually in a given .pt. This script plots exactly
what training will see.

Usage:
    uv run python plot_features.py                                            # plots dataset/original
    uv run python plot_features.py --in-dir ../results/dataset/augmented --out-dir ../results/feature_plots/augmented
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skeleton_pipeline.dataset import io_utils, feature_io
from skeleton_pipeline.plotting.feature_plots import plot_panels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "dataset" / "original"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "feature_plots" / "original"
LOG_PATH = PROJECT_ROOT / "logs" / "plot_features_3d.log"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR),
                         help="Folder of features__*.pt files to plot (dataset/original, "
                              "dataset/augmented, dataset/segment, ...).")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-lines-per-plot", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    logger = io_utils.setup_logger("plot_features_3d")
    io_utils.save_log_to_file(logger, str(LOG_PATH))

    pt_files = list(io_utils.iter_files(in_dir, extension=".pt"))
    logger.info("Found %s .pt file(s) under %s", len(pt_files), in_dir)

    for pt_file in pt_files:
        take_id = pt_file.stem.replace("features__", "", 1)
        logger.info("Plotting %s", pt_file.name)
        try:
            data = io_utils.load_torch(pt_file, logger=logger)
            metadata = data["metadata"]
            fps = metadata["fps"]
            feature_dict, panel_groups = feature_io.to_feature_dict(data["features"], metadata["panel_columns"])

            # metadata["total_frames"] is the SOURCE clip's length -- for a
            # segment_data.py chunk that's NOT this file's own frame count,
            # so derive T from the actual feature tensors instead (works
            # for original/augmented/segmented .pt's alike).
            n_frames = len(next(iter(feature_dict.values())))
            frame_start = metadata.get("frame_start", 0)  # segment_data.py sets this; original/augmented takes don't
            timestamps = frame_start / fps + np.arange(n_frames) / fps

            # Flat out_dir/<take_id>_<panel>.png, same convention
            # generate_lstm_training_data.py/debug_visualize.py use -- no
            # extra out_dir/take_id/ subfolder (that duplicated take_id in
            # both the folder and the filename, needlessly lengthening
            # paths towards Windows' MAX_PATH limit for long panel names).
            plot_paths = plot_panels(feature_dict, panel_groups, timestamps, out_dir, take_id,
                                      max_lines_per_plot=args.max_lines_per_plot)
            logger.info("  %s panel plot(s) -> %s", len(plot_paths), out_dir)
        except Exception:
            logger.exception("Failed to plot %s", pt_file)


if __name__ == "__main__":
    main()
