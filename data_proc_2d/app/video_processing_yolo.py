
import sys
from pathlib import Path

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from src.human_pose_extractor import VideoPoseExtractor
from src import file_io_utils
from utilities import log_utils


INPUT_MODE = "video"  # "video" or "pt"
FILE_NAME_FILTER = "keypoints_test_01__cam-02_uid-01_take-02"
WRITE_OUTPUT_VIDEO = True


def main():
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("pose_processing")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/video_processing.log"))
    
    # file path setup
    video_root_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
    # input_video_path = video_root_path / "cropped" / "cam-02"
    input_video_path = video_root_path / "raw" / "cam-04"
    input_pt_path = video_root_path / "dataset" / "keypoints"
    # output_root_path = Path(__file__).resolve().parents[1] / "dataset" / "train" / "raw"
    output_root_path = video_root_path / "dataset" / "keypoints"
    output_video_root_path = video_root_path / "archive" / "skeleton_videos"
    output_root_path.mkdir(parents=True, exist_ok=True)
    output_video_root_path.mkdir(parents=True, exist_ok=True)

    if INPUT_MODE == "video":
        source_root = input_video_path
        source_files = sorted(file_io_utils.iter_video_files(source_root))
    elif INPUT_MODE == "pt":
        source_root = input_pt_path
        source_files = sorted(file_io_utils.iter_files(source_root, extension=".pt"))
    else:
        raise ValueError(f"Unsupported INPUT_MODE: {INPUT_MODE}")

    logger.info("Found %s %s file(s) under %s", len(source_files), INPUT_MODE, source_root)

    for index, source_file in enumerate(source_files, start=1):
        # path setup for output files, preserving relative structure and changing extension
        rel_path = source_file.relative_to(source_root)
        modified_rel_path = file_io_utils.modify_file_name(rel_path, prefix=f"skeleton_video")
        output_mp4 = (output_video_root_path / modified_rel_path).with_suffix(".mp4") if WRITE_OUTPUT_VIDEO else None
        modified_rel_path = file_io_utils.modify_file_name(rel_path, prefix=f"keypoints")
        output_pt = (output_root_path / modified_rel_path).with_suffix(".pt")
        if output_mp4 is not None:
            output_mp4.parent.mkdir(parents=True, exist_ok=True)
        output_pt.parent.mkdir(parents=True, exist_ok=True)

        logger.info("[%s/%s] Processing %s", index, len(source_files), source_file)
        
        try:
            extractor_kwargs = {
                "output_pt": str(output_pt),
                "output_video": str(output_mp4) if output_mp4 is not None else None,
                "model_path": "model_data/yolo26n-pose.pt",
                "show_every_n_frames": 0,
                "use_visibility": False,
                "logger": logger,
                "use_tracking": True,
            }
            if INPUT_MODE == "video":
                extractor = VideoPoseExtractor(
                    video_path=str(source_file),
                    **extractor_kwargs,
                )
            else:
                extractor = VideoPoseExtractor(
                    input_pt=str(source_file),
                    **extractor_kwargs,
                )
            extractor.run_pose_extraction()
        except Exception as error:
            logger.exception("Failed processing %s: %s", source_file, error)



if __name__ == "__main__":
    main()