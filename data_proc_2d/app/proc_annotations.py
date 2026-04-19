import logging
from pathlib import Path
import re
from shutil import copy2
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utilities import file_io, log_utils
from src import file_io_utils
from src.annotation_config import ANNOTATION_CONFIG


RENAME_ORIGINAL_FILES = False
VIDEO_SOURCE_ROOT_PATH = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos\annotations")
GOOGLE_DRIVE_ROOT_PATH = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
LOG_ROOT_PATH = Path(__file__).resolve().parents[2] / "logs"


regex_pattern = (
    r"^(?P<prefix>[A-Za-z0-9]+)"
    r"__cam-(?P<camera_id>\d+)"
    r"_uid-(?P<user_id>\d+)"
    r"_take-(?P<take_id>\d+)$"
)

def main():
    logger = setup_logging()

    # file path setup
    save_annotations_root_path = GOOGLE_DRIVE_ROOT_PATH / "annotations"
    save_json_root_path = GOOGLE_DRIVE_ROOT_PATH / "dataset" / "annotations" 
    
    for csv_file in sorted(VIDEO_SOURCE_ROOT_PATH.glob(f"*.csv")):
        output_dict, video_file_name = file_io_utils.load_elan_label_data(csv_file, config=ANNOTATION_CONFIG)
        if video_file_name is None:
            logger.warning("SKIP %s: failed to load ELAN label data", csv_file)
            continue
        
        match = re.fullmatch(regex_pattern, Path(video_file_name).stem)
        if not match:
            logger.warning("SKIP %s: filename does not match expected pattern", csv_file)
            continue
        
        camera_id = file_io.pad_id(match.group("camera_id"))
        base_file_name = f"label__cam-{camera_id}_uid-{match.group('user_id')}_take-{match.group('take_id')}"
        save_json_file_path = save_json_root_path / f"cam-{camera_id}" / \
            f"{base_file_name}.json"
        save_annotations_file_path = save_annotations_root_path / f"cam-{camera_id}" / \
            f"{base_file_name}.csv"
        
        upload_file(csv_file, save_annotations_file_path, rename=RENAME_ORIGINAL_FILES, logger=logger)
        file_io.save_json(output_dict, str(save_json_file_path), logger=logger)
        
        src_file_path = csv_file.with_suffix(".eaf")
        if src_file_path.exists():
            upload_file(src_file_path, save_annotations_file_path.with_suffix(".eaf"), rename=RENAME_ORIGINAL_FILES, logger=logger)


# copy file and optionally rename it before saving into the target folder
def upload_file(
    src_file_path: Path,
    target_file_path: Path,
    rename: bool = True,
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or logging.getLogger(Path(__file__).stem)
    destination_path = target_file_path 

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        logger.warning("SKIP %s: target already exists -> %s", src_file_path, destination_path.name)
        return

    logger.info("COPY %s -> %s", src_file_path, destination_path)
    copy2(src_file_path, destination_path)

    if rename:
        rename_file_path = src_file_path.parent / target_file_path.name
        if rename_file_path.exists():
            logger.warning("SKIP %s: renamed source already exists -> %s", src_file_path, rename_file_path.name)
            return
        src_file_path.rename(rename_file_path)
        logger.info("RENAME %s -> %s", src_file_path, rename_file_path)
    

def setup_logging() -> logging.Logger:
    logger_name = Path(__file__).stem
    logger = logging.getLogger(logger_name)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger = log_utils.setup_logger(logger_name)
    logger.propagate = False
    LOG_ROOT_PATH.mkdir(parents=True, exist_ok=True)

    log_file_path = LOG_ROOT_PATH / f"{Path(__file__).stem}.log"
    log_utils.save_log_to_file(logger, str(log_file_path))
    logger.info("Saving logs to %s", log_file_path)
    return logger




if __name__ == "__main__":
    main()