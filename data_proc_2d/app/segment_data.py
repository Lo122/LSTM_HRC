
import sys
from pathlib import Path
import re

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

if str(PROJECT_SRC_ROOT / "data_proc_2d") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT / "data_proc_2d"))
from utilities import log_utils
from src.file_io_utils import load_torch, save_torch, iter_files


SEGMENT_SIZE = 1000  # number of frames per segment, adjust as needed


def main():
    # setup logger and file paths
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("segment_data")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/segment_data.log"))
    
    # I/O paths
    google_drive_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
    pt_root = google_drive_path / "dataset" / "original"
    pt_files = list(iter_files(pt_root, extension=".pt"))
    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)
    
    out_pt_dir = google_drive_path / "dataset" / "segment"
    
    # main loop to process each .pt file and extract features
    for pt_file in pt_files:
        logger.info("Processing %s", pt_file.name)
        try:
            segment_data(pt_file, out_pt_dir, segment_size=SEGMENT_SIZE, logger=logger)

        except Exception as error:
            logger.exception("Failed to process %s: %s", pt_file, error)


def segment_data(pt_file: Path, out_pt_dir: Path, segment_size: int, logger):
    data = load_torch(str(pt_file), logger=logger)

    metadata = data.get("metadata", {})
    features = data.get("features", {})
    labels = data.get("labels", {})
    
    # segment data by segment size and save it to output directory
    for segment_index, start in enumerate(range(0, metadata.get("total_frames", 0), segment_size)):
        end = start + segment_size
        metadata['segment_index'] = segment_index
        metadata['segment_size'] = segment_size
        metadata['frame_start'] = start
        metadata['frame_end'] = end
        output_data = {
            "metadata": metadata,
            "features": {key: _materialize_segment_slice(value, start, end) for key, value in features.items()},
            "labels": {key: _materialize_segment_slice(value, start, end) for key, value in labels.items()},
        }

        file_name =  _format_output_file_name(pt_file.name, segment_index=segment_index)
        out_pt_path = out_pt_dir / file_name
        save_torch(output_data, str(out_pt_path), logger=logger)


def _materialize_segment_slice(value, start: int, end: int):
    segment = value[start:end]

    if hasattr(segment, "clone"):
        return segment.clone()

    if hasattr(segment, "copy"):
        try:
            return segment.copy()
        except TypeError:
            pass

    return segment
    

def _format_output_file_name(input_file_name: str, segment_index: int) -> str:
    pattern = r"^features__(cam-\d{2}_uid-\d{2}_take-\d{2})\.pt$"
    replacement = rf"features__\1_seg-{segment_index:02}.pt"
    return re.sub(pattern, replacement, input_file_name)



def debug_segment_data():
    
    # setup logger and file paths
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("segment_data")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/segment_data.log"))
    
    # I/O paths
    google_drive_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
    pt_root = google_drive_path / "dataset" / "segment"
    pt_files = list(iter_files(pt_root, extension=".pt"))
    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)
    
    for pt_file in pt_files:
        logger.info("Processing %s", pt_file.name)
        data = load_torch(str(pt_file), logger=logger)

        metadata = data.get("metadata", {})
        features = data.get("features", {})
        labels = data.get("labels", {})
        
        logger.info("Metadata: %s", metadata)
        for key, value in features.items():
            logger.info("Feature %s: shape=%s dtype=%s", key, value.shape, value.dtype)
            print(value)
        for key, value in labels.items():
            logger.info("Label %s: shape=%s dtype=%s", key, value.shape, value.dtype)
            print(value)
        
        
        

if __name__ == "__main__":
    main()
    # debug_segment_data()