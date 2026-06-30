import json
from shutil import copy2
import subprocess
from datetime import datetime
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parent))
from utilities import log_utils



GOOGLE_DRIVE_ROOT_PATH = Path(r"C:\Users\Owner\Downloads\test")

USER_MAP ={
    "Benan": "01",
    "Yang": "02",
    "Masaki": "03",
    "Gonzalo": "04",
    "Simon": "05",
    "Ainsleigh": "06",
    "Aleks": "07",
    "Anishwar": "08",
    "Ferhat": "09"
}


def rename_video_files(video_path: Path, suffixes: str, logger: logging.Logger) -> Path | None:
    
    for camera_folder in video_path.iterdir():
        if not camera_folder.is_dir():
            continue
        camera_name = camera_folder.name
        if not camera_name.lower().startswith("cam"):
            logger.warning("SKIP %s: folder name '%s' does not start with 'cam'", camera_folder, camera_name)
            continue
        
        for video_folder in camera_folder.iterdir():
            if not video_folder.is_dir():
                continue
            user_name = video_folder.name
            if user_name.capitalize() not in USER_MAP.keys():
                logger.warning("SKIP %s: parent folder name '%s' not found in legacy user map", video_folder, user_name)
                continue
            
            user_id = USER_MAP[user_name.capitalize()]
            
            video_files = sorted(
                [path for path in video_folder.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}],
                key=get_taken_datetime,
            )
            for counter, video_file in enumerate(video_files, start=1):
                
                file_name = f"video__{camera_name}_uid-{user_id}_take-{counter:02}"
                
                target_file_path = GOOGLE_DRIVE_ROOT_PATH / camera_name / file_name
                target_file_path = target_file_path.with_suffix(".mp4")
                
                upload_file(video_file, target_file_path, rename=True, logger=logger)
                logger.info("Renamed %s to %s", video_file.name, target_file_path.name)



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
    


def get_taken_datetime(video_file: Path) -> datetime:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format_tags=creation_time:stream_tags=creation_time",
                "-of", "json",
                str(video_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return datetime.fromtimestamp(video_file.stat().st_mtime)

    if result.returncode == 0 and result.stdout.strip():
        data = json.loads(result.stdout)

        format_tags = data.get("format", {}).get("tags", {})
        creation_time = format_tags.get("creation_time")
        if creation_time:
            return datetime.fromisoformat(creation_time.replace("Z", "+00:00"))

        for stream in data.get("streams", []):
            stream_tags = stream.get("tags", {})
            creation_time = stream_tags.get("creation_time")
            if creation_time:
                return datetime.fromisoformat(creation_time.replace("Z", "+00:00"))

    return datetime.fromtimestamp(video_file.stat().st_mtime)



if __name__ == "__main__":
    
    root_folder_path = GOOGLE_DRIVE_ROOT_PATH
    logger = log_utils.setup_logger(Path(__file__).stem)
    
    rename_video_files(Path(root_folder_path), suffixes="mp4", logger=logger)