"""Generate rotation + joint-noise augmented copies of the cam-04 LSTM
training-data takes -- see skeleton_pipeline.dataset.augment's module
docstring for what each transform does and why it must run on RAW (T, 17,
3) positions rather than on already-derived features.

For each source video__cam-04_uid-XX_take-XX.npz (same source
build_training_pairs.py reads), this writes NUM_AUGMENTATIONS_PER_TAKE
augmented copies -- features__cam-04_uid-XX_take-XX_aug-01.pt,
..._aug-02.pt, etc. -- to OUT_PT_DIR, each with a fresh random rotation +
noise draw. Labels are unaffected by a spatial transform of the skeleton
(they're purely a function of frame index/annotation), so they're reused
as-is from skeleton_pipeline.dataset.labels.extract_labels rather than
recomputed per augmentation.

Run build_training_pairs.py first (or alongside -- they're independent) to
get the un-augmented "original" take for every video too; a typical
training set uses both.

AUGMENT THE TRAINING SPLIT ONLY, and split by SUBJECT: an augmented copy of a
held-out take is a direct leak, and so is putting cam-05 of a take in train
while cam-07 of the same take is in test -- the three camera angles are one
performance with identical labels.

Usage:
    uv run python augment_dataset.py
    uv run python augment_dataset.py --mirror half       # half the copies mirrored
    uv run python augment_dataset.py --mirror all --no-rotate --no-noise
    uv run python augment_dataset.py --noise-sigma-m 0.02 --rotation-axis z
    uv run python augment_dataset.py --no-noise --rotation-axis z --rotation-range-deg -20 20
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skeleton_pipeline.dataset import io_utils, labels, feature_io
from skeleton_pipeline.dataset.augment import (
    AugmentationConfig, augment_positions, mirror_bone_length_targets)
from skeleton_pipeline.plotting.label_plots import plot_label_debug

# Same sources build_training_pairs.py reads -- keep the two in step.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOOGLE_DRIVE_ROOT = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
DATASET_ROOT = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation"
NPZ_ROOT = DATASET_ROOT / "raw_npz"                    # generate_lstm_training_data.py output
ANNOTATIONS_DIR = GOOGLE_DRIVE_ROOT / "annotations" / "ceiling_installation"
OUT_PT_DIR = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation_02" / "augmented_rotations"
PLOT_DIR = GOOGLE_DRIVE_ROOT / "dataset" / "skeleton_3d" / "ceiling_panel_installation_02" / "label_plots"
LOG_PATH = PROJECT_ROOT / "logs" / "augment_dataset_3d.log"

# The trailing "-N" is a take split across two annotation files.
NPZ_FILE_PATTERN = re.compile(r"^video__(cam-\d{2}_uid-\d{2}_take-\d{2}(?:-\d+)?)\.npz$")

NUM_AUGMENTATIONS_PER_TAKE = 3
SAVE_PLOTS = True  # labels are identical across augmentations of the same take -- skip by default, redundant
SHOW_PLOTS = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-augmentations", type=int, default=NUM_AUGMENTATIONS_PER_TAKE)
    parser.add_argument("--rotation-axis", type=str, default="z", choices=["x", "y", "z", "xyz"],
                         help="'z' (yaw only, default) rotates about the vertical axis only -- see "
                              "skeleton_pipeline/dataset/augment.py's module docstring for why that's "
                              "the default over a full random 3D rotation.")
    parser.add_argument("--rotation-range-deg", type=float, nargs=2, default=(-20.0, 20.0))
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--noise-sigma-m", type=float, default=0.01,
                         help="Per-joint per-frame Gaussian position jitter, in meters (default 1 cm).")
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--mirror", choices=("off", "all", "half"), default="off",
                        help="Sagittal mirroring (see skeleton_pipeline/dataset/augment.py's "
                             "mirror_positions on when it is valid). 'half' mirrors every "
                             "other augmentation of a take, so one run yields both handedness "
                             "variants; 'all' mirrors every copy.")
    parser.add_argument("--no-causal-features", dest="causal_features", action="store_false",
                         help="Compute features with CENTERED Savitzky-Golay windows instead of "
                              "trailing ones. Not recommended: a centered window at "
                              "feature window_length=9 uses 4 frames of lookahead (400 ms at the "
                              "10 Hz the live loop runs at), which the streaming extractor in "
                              "skeleton3d_pipeline.py cannot reproduce -- the model would be "
                              "trained on an estimator deployment does not have. See "
                              "skeleton_pipeline/features/h36m_features.py's _savgol_derivative.")
    parser.set_defaults(causal_features=True)
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed -- augmentation k of take N uses seed + N*1000 + k for "
                              "reproducibility without every take/augmentation sharing one draw.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = io_utils.setup_logger("augment_dataset_3d")
    io_utils.save_log_to_file(logger, str(LOG_PATH))

    config = AugmentationConfig(
        mirror=False,                      # decided per augmentation below, from --mirror
        rotate=not args.no_rotate,
        rotation_axis=args.rotation_axis,
        rotation_range_deg=tuple(args.rotation_range_deg),
        noise=not args.no_noise,
        noise_sigma_m=args.noise_sigma_m,
    )

    logger.info("Feature windows: %s",
                "CAUSAL (trailing -- matches the live streaming extractor)"
                if args.causal_features else
                "CENTERED (uses lookahead -- live CANNOT reproduce this)")

    npz_files = sorted(p for p in NPZ_ROOT.glob("*.npz") if NPZ_FILE_PATTERN.match(p.name))
    logger.info("Found %s .npz file(s) under %s -- generating %s augmentation(s) each",
                len(npz_files), NPZ_ROOT, args.num_augmentations)

    for take_index, npz_file in enumerate(npz_files):
        take_id = NPZ_FILE_PATTERN.match(npz_file.name).group(1)  # e.g. "cam-04_uid-01_take-01"
        annotation_data_id = "_".join(take_id.split("_")[1:])
        logger.info("Processing %s", npz_file.name)
        try:
            data = np.load(npz_file, allow_pickle=True)
            positions = data["keypoints_3d"]
            fps = float(data["fps"])
            total_frames = positions.shape[0]
            
            label_tensors, debug_frames = labels.extract_labels(
                annotation_data_id, ANNOTATIONS_DIR, frame_size=total_frames, logger=logger)

            for aug_index in range(1, args.num_augmentations + 1):
                seed = args.seed + take_index * 1000 + aug_index
                mirror = args.mirror == "all" or (args.mirror == "half" and aug_index % 2 == 0)
                run_config = AugmentationConfig(
                    **{**config.__dict__, "seed": seed, "mirror": mirror})

                augmented_positions, applied_params = augment_positions(positions, run_config)
                features, panel_columns = feature_io.features_from_positions(
                    augmented_positions, fps, causal=args.causal_features)

                # Mirroring swaps the subject's left and right limbs, so the
                # per-edge targets have to follow the positions.
                bone_length_targets = data["bone_length_targets"]
                if mirror:
                    bone_length_targets = mirror_bone_length_targets(
                        bone_length_targets, data["bone_length_edges"])

                metadata = {
                    "source_npz": str(npz_file),
                    "total_frames": total_frames,
                    "fps": fps,
                    "bone_length_edges": data["bone_length_edges"].tolist(),
                    "bone_length_targets": np.asarray(bone_length_targets).tolist(),
                    "body_scale_m": float(data["body_scale_m"]),
                    "gravity_aligned": bool(data["gravity_aligned"]),
                    "panel_columns": panel_columns,
                    "causal_features": bool(args.causal_features),
                    "label_config": labels.DEFAULT_LABEL_CONFIG.__dict__,
                    "augmentation": applied_params,
                }

                take_aug_id = f"{take_id}_aug-{aug_index:02}"
                output_data = {"metadata": metadata, "features": features, "labels": label_tensors}
                io_utils.save_torch(output_data, OUT_PT_DIR / f"features__{take_aug_id}.pt", logger=logger)
                logger.info("  aug-%02d: mirror=%s rotation=%s noise=%s -> %s frames",
                            aug_index, applied_params["mirror"], applied_params["rotation"],
                            applied_params["noise"], total_frames)

                if SAVE_PLOTS or SHOW_PLOTS:
                    plot_label_debug(take_aug_id, debug_frames, logger, show=SHOW_PLOTS, save=SAVE_PLOTS,
                                      save_path=PLOT_DIR / f"label_plot__{take_aug_id}.png")

        except Exception:
            logger.exception("Failed to process %s", npz_file)


if __name__ == "__main__":
    main()
