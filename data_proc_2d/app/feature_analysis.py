
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

# from src.utilities import log_utils
from src.file_io_utils import load_torch, save_json, load_step_ids_from_json, load_elan_label_data
from data_proc_2d.src import pose_analysis_yolo
from data_proc_2d.src import plot_utils


def _get_optional_tensor(data: dict, *names: str):
    for name in names:
        value = data.get(name)
        if value is not None:
            return value, name
    return None, None


def main():
    # setup logger and file paths
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("pose_processing")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/video_analysis.log"))

    db_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos")
    pt_root = db_path / ".pt"
    pt_files = list(iter_pt_files(pt_root))

    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)
    
    # label files: map video file name to list of step labels
    label_root = db_path / ".json"
    label_files = list(iter_files(label_root, extension=".json"))
    step_label_list = {}
    for label_file in label_files:
        logger.info("Found label file: %s", label_file)
        label_info, video_file_name = load_step_ids_from_json(label_file, logger=logger)
        step_label_list[video_file_name] = {"labels": label_info}
    
    
    elan_label_root = Path(r"G:\My Drive\University of Stuttgart\ITECH_Thesis\ELAN")
    
    label_files = list(iter_files(elan_label_root, extension=".csv"))
    elan_step_label_list = {}
    for label_file in label_files:
        logger.info("Found ELAN label file: %s", label_file)
        label_info, video_file_name = load_elan_label_data(label_file, logger=logger)
        elan_step_label_list[video_file_name] = {"labels": label_info}


    plots_dir = folder_root / "plots" / "video_analysis_yolo"
    csv_dir   = folder_root / "data_output" / "video_analysis_yolo"
    output_json = folder_root / "data_output" / "video_analysis_yolo" / "feature_results.json"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)


    # main loop to process each .pt file and extract features
    results: dict = {}
    for pt_file in pt_files:
        logger.info("Processing %s", pt_file.name)
        try:
            results = video_analysis_yolo(pt_file, plots_dir, csv_dir, logger, results, step_label_list, elan_step_label_list)

        except Exception as error:
            logger.exception("Failed to process %s: %s", pt_file, error)

    save_json(results, str(output_json), logger=logger)
    logger.info("Feature extraction complete. Results saved to %s", output_json)


def video_analysis_yolo(pt_file: Path, plots_dir: Path, csv_dir: Path, logger, results: dict, step_label_list: dict, elan_step_label_list: dict) -> dict:

    data      = load_torch(str(pt_file), logger=logger)
    landmarks = data["landmarks"]         # (T, 17, 2)
    speed, speed_key = _get_optional_tensor(data, "speed", "feat_speed")
    acceleration, acceleration_key = _get_optional_tensor(data, "acceleration", "feat_acc")
    metadata  = data.get("metadata", {})

    if speed_key is None:
        logger.info("  No stored speed tensor found for %s; deriving speed from landmarks", pt_file.name)
    if acceleration_key is None:
        logger.info(
            "  No stored acceleration tensor found for %s; deriving acceleration from available data",
            pt_file.name,
        )

    features             = pose_analysis_yolo.extract_features(
        landmarks,
        speed=speed,
        acceleration=acceleration,
    )
    features["metadata"] = metadata

    results[pt_file.stem] = features

    logger.info(
        "  frames=%d  body_speed_mean=%.4f  body_speed_max=%.4f",
        features["num_frames"],
        features["body_speed_mean"],
        features["body_speed_max"],
    )

    stem = pt_file.stem

    # ---- compute DataFrames ----
    speed_df         = pose_analysis_yolo.compute_joint_speeds_df(landmarks, speed=speed)
    accel_df         = pose_analysis_yolo.compute_joint_acceleration_df(
        landmarks,
        acceleration=acceleration,
        speed=speed,
    )
    angle_df         = pose_analysis_yolo.compute_joint_angles_df(landmarks)
    ratio_df         = pose_analysis_yolo.compute_distance_ratios_df(landmarks)
    dist_hip_df      = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="hip")
    dist_shoulder_df = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="shoulder")
    dist_half_df     = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="half_body")
    
    
    video_path = metadata.get("video")
    video_file_name = os.path.basename(video_path) if video_path else "unknown"
    step_labels = step_label_list.get(video_file_name, {}).get("labels", [])
    elan_step_labels = elan_step_label_list.get(video_file_name, {}).get("labels", [])
    if step_labels:
        logger.info("  Found %d step labels for video %s", len(step_labels), video_file_name)
        features["step_labels"] = step_labels
    if elan_step_labels:
        logger.info("  Found %d ELAN step labels for video %s", len(elan_step_labels), video_file_name)
        features["elan_step_labels"] = elan_step_labels
    
    # ---- save CSVs ----
    merged_df = pose_analysis_yolo.merge_pose_dfs(
        speed_df,
        accel_df,
        dist_hip_df,
        dist_shoulder_df,
        dist_half_df,
        angle_df=angle_df,
        ratio_df=ratio_df,
    )
    merged_df.to_csv(str(csv_dir / f"{stem}_merged.csv"), index=False)
    logger.info("  Saved CSVs for %s", stem)

    # ---- plot (all panels in one image) ----
    plot_utils.plot_pose_analysis(
        speed_df,
        accel_df,
        angle_df,
        dist_half_df,
        dist_hip_df,
        dist_shoulder_df,
        ratio_df=ratio_df,
        save_path=str(plots_dir / f"{stem}_pose_analysis.png"),
        suptitle=stem,
        step_labels=step_labels,
        elan_step_labels=elan_step_labels,
    )
    logger.info("  Saved plots for %s", stem)
    
    return results


def iter_pt_files(root: Path):
    """Yield all .pt files recursively under *root*."""
    for pt_file in sorted(root.rglob("*.pt")):
        yield pt_file


def iter_files(root: Path, extension: str = ".json"):
    """Yield all files with the given extension recursively under *root*."""
    for file in sorted(root.rglob(f"*{extension}")):
        yield file







if __name__ == "__main__":
    file_path = Path(r"C:\Users\Owner\OneDrive - Universität Stuttgart\2025_26_Thesis\codes\LSTM_HRC\data_proc_2d\app\dataset\train\raw\video__cam_04_uid-07_take-1_pose.pt")
    
    data      = load_torch(str(file_path))
    print(data.keys())
    main()