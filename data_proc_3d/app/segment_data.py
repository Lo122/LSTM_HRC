"""Split every {"metadata", "features", "labels"} .pt take under
IN_PT_DIR into fixed-length (SEGMENT_SIZE-frame) segments -- the 3D
counterpart of data_proc_2d/app/segment_data.py, built on top of
skeleton_pipeline.dataset.segment (that module's slicing logic is
dtype-agnostic, so it works unmodified on both build_training_pairs.py's
"original" takes and augment_dataset.py's "augmented" ones -- point
IN_PT_DIR at either).

Usage:
    uv run python segment_data.py
    uv run python segment_data.py --in-dir ../results/dataset/augmented --out-dir ../results/dataset/augmented_segment
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skeleton_pipeline.dataset import io_utils, segment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "dataset" / "original"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "dataset" / "segment"
LOG_PATH = PROJECT_ROOT / "logs" / "segment_data_3d.log"

SEGMENT_SIZE = 1000  # frames per segment, adjust as needed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    logger = io_utils.setup_logger("segment_data_3d")
    io_utils.save_log_to_file(logger, str(LOG_PATH))

    pt_files = list(io_utils.iter_files(in_dir, extension=".pt"))
    logger.info("Found %s .pt file(s) under %s", len(pt_files), in_dir)

    for pt_file in pt_files:
        logger.info("Segmenting %s", pt_file.name)
        try:
            data = io_utils.load_torch(pt_file, logger=logger)
            for segment_index, segment_data in segment.iter_segments(data, segment_size=args.segment_size):
                out_name = segment.format_segment_file_name(pt_file.name, segment_index)
                io_utils.save_torch(segment_data, out_dir / out_name, logger=logger)
        except Exception:
            logger.exception("Failed to segment %s", pt_file)


if __name__ == "__main__":
    main()
