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
  2. loads the matching label__cam-04_uid-XX_take-YY.json / label_detail__
     ...json annotation pair (the same files data_proc_2d/app/
     proc_annotations.py produces from ELAN exports, still present on the
     annotations Drive folder) and turns them into smooth per-frame label
     tensors (skeleton_pipeline.dataset.labels.extract_labels),
  3. torch.save's one {"metadata", "features", "labels"} dict per take
     (skeleton_pipeline.dataset.io_utils.save_torch).

See skeleton_pipeline/dataset/labels.py's module docstring for the
reimplementation note on why its label-smoothing math isn't a port of
data_proc_2d's (lost) labelling_utils.py.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skeleton_pipeline.dataset import io_utils, labels, feature_io
from skeleton_pipeline.plotting.label_plots import plot_label_debug


# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ_ROOT = PROJECT_ROOT / "data_proc_3d" / "results" / "raw"           # generate_lstm_training_data.py output
GOOGLE_DRIVE_ROOT = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
ANNOTATIONS_DIR = GOOGLE_DRIVE_ROOT / "annotations" / "cam-04"
OUT_PT_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "dataset" / "original"
PLOT_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "label_plots"
LOG_PATH = PROJECT_ROOT / "logs" / "build_training_pairs_3d.log"

NPZ_FILE_PATTERN = re.compile(r"^video__(cam-\d{2}_uid-\d{2}_take-\d{2})\.npz$")

SAVE_PLOTS = True
SHOW_PLOTS = False


def main():
    logger = io_utils.setup_logger("build_training_pairs_3d")
    io_utils.save_log_to_file(logger, str(LOG_PATH))

    npz_files = sorted(p for p in NPZ_ROOT.glob("*.npz") if NPZ_FILE_PATTERN.match(p.name))
    logger.info("Found %s .npz file(s) under %s", len(npz_files), NPZ_ROOT)

    for npz_file in npz_files:
        logger.info("Processing %s", npz_file.name)
        try:
            take_id = NPZ_FILE_PATTERN.match(npz_file.name).group(1)  # e.g. "cam-04_uid-01_take-01"

            metadata, features = feature_io.extract_features(npz_file, logger)
            metadata["label_config"] = labels.DEFAULT_LABEL_CONFIG.__dict__

            label_tensors, debug_frames = labels.extract_labels(
                take_id, ANNOTATIONS_DIR, frame_size=metadata["total_frames"], logger=logger)

            if SAVE_PLOTS or SHOW_PLOTS:
                plot_label_debug(take_id, debug_frames, logger, show=SHOW_PLOTS, save=SAVE_PLOTS,
                                  save_path=PLOT_DIR / f"label_plot__{take_id}.png")

            output_data = {"metadata": metadata, "features": features, "labels": label_tensors}
            print(f"Saving {take_id} to {OUT_PT_DIR / f'features__{take_id}.pt'}")  
            # io_utils.save_torch(output_data, OUT_PT_DIR / f"features__{take_id}.pt", logger=logger)

        except Exception:
            logger.exception("Failed to process %s", npz_file)


if __name__ == "__main__":
    main()
