
import sys
from pathlib import Path

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from utilities.human_pose_extractor import VideoPoseExtractor
from utilities import log_utils

OUTPUT_FOLDER_NAME = "skeleton_videos"


def main():
    logger = log_utils.setup_logger("pose_processing")

    # file path setup
    video_root_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos")
    input_video_path = video_root_path / "originals"
    output_root_path = video_root_path / OUTPUT_FOLDER_NAME
    output_root_path.mkdir(parents=True, exist_ok=True)

    

    video_files = sorted(iter_video_files(input_video_path))
    logger.info("Found %s videos under %s", len(video_files), input_video_path)

    for index, video_file in enumerate(video_files, start=1):
        rel_path = video_file.relative_to(input_video_path)
        modified_rel_path = modify_file_name(rel_path)
        output_mp4 = (output_root_path / modified_rel_path).with_suffix(".mp4")
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        output_pt = (output_root_path / modified_rel_path).with_suffix(".pt")
        output_pt.parent.mkdir(parents=True, exist_ok=True)


        logger.info("[%s/%s] Processing %s", index, len(video_files), video_file)
        try:
            extractor = VideoPoseExtractor(
                video_path=str(video_file),
                output_pt=str(output_pt),
                output_video=str(output_mp4),
                model_path="model_data/yolo26n-pose.pt",
                show_every_n_frames=0,
                use_visibility=False,
                logger=logger,
                use_tracking=True,
                smoothing="one_euro",
                
            )
            extractor.run_pose_extraction()
        except Exception as error:
            logger.exception("Failed processing %s: %s", video_file, error)



def iter_video_files(root: Path):
    suffixes = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in suffixes:
            if "human_skeletons" in file_path.parts:
                continue
            yield file_path



def modify_file_name(rel_path: Path) -> Path:
    # Example modification: change "cam1_spacer_M2.mp4" to "cam1_spacer_M2_pose.mp4"
    if rel_path.suffix.lower() == ".mp4":
        return rel_path.with_name(rel_path.stem + "_pose" + rel_path.suffix)
    return rel_path

if __name__ == "__main__":
    main()
