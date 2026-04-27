import logging
from pathlib import Path
import re
from shutil import copy2
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utilities import file_io, log_utils
from src import file_io_utils
from src.annotation_config import ANNOTATION_CONFIG


RENAME_ORIGINAL_FILES = False
VIDEO_SOURCE_ROOT_PATH = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos\annotations\15.04.2026")
GOOGLE_DRIVE_ROOT_PATH = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
LOG_ROOT_PATH = Path(__file__).resolve().parents[2] / "logs"


regex_pattern = (
    r"^(?P<prefix>[A-Za-z0-9]+)"
    r"__cam-(?P<camera_id>\d+)"
    r"_uid-(?P<user_id>\d+)"
    r"_take-(?P<take_id>\d+)$"
)

def main():
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("proc_annotations")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/video_processing.log"))
    
    # file path setup
    save_annotations_root_path = GOOGLE_DRIVE_ROOT_PATH / "annotations"
    
    for csv_file in sorted(VIDEO_SOURCE_ROOT_PATH.glob(f"*.csv")):
        try:
            labels, video_file_name, meta_data = file_io_utils.load_elan_label_data(csv_file, config=ANNOTATION_CONFIG, logger=logger)
        except ValueError as e:
            logger.error("Failed to load ELAN label data from %s: %s", csv_file, e)
            continue
        
        if video_file_name is None:
            logger.warning("SKIP %s: failed to load ELAN label data", csv_file)
            continue

        search_path = GOOGLE_DRIVE_ROOT_PATH / "raw"
        video_file = find_source_video(VIDEO_SOURCE_ROOT_PATH, search_path, video_file_name)
        meta_data["file_url"] = str(video_file) if video_file else None
        meta_data["video_file_path"] = meta_data.pop("file_url")
        
        output_dict = {
            "meta_data": meta_data, 
            "labels": labels,
        }


        match = re.fullmatch(regex_pattern, Path(video_file).stem)
        if not match:
            logger.warning("SKIP %s: filename does not match expected pattern", csv_file)
            continue
        
        camera_id = file_io.pad_id(match.group("camera_id"))
        if "detail" in csv_file.stem.lower():
            base_file_name = f"label_detail__cam-{camera_id}_uid-{match.group('user_id')}_take-{match.group('take_id')}"
        else:
            base_file_name = f"label__cam-{camera_id}_uid-{match.group('user_id')}_take-{match.group('take_id')}"
        
        save_json_file_path = save_annotations_root_path / f"cam-{camera_id}" / \
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
    

def find_source_video(ref_video_path: Path, search_path: Path, video_file_name: str) -> Path | None:

    def get_video_info(path: Path) -> dict | None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        info = {
            "file_size": path.stat().st_size,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return info

    video_file_path = None
    for video_file in ref_video_path.glob("*.mp4"):
        if video_file.name == video_file_name:
            video_file_path = video_file.resolve()
            break
    
    if video_file_path is None:
        return search_path / video_file_name

    ref_info = get_video_info(video_file_path)
    if ref_info is None:
        return None

    for video_file in search_path.glob("**/*.mp4"):
        info = get_video_info(video_file)
        if info == ref_info:
            return video_file

    return None


if __name__ == "__main__":
    main()