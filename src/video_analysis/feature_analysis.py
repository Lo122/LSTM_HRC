
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from utilities import log_utils
from utilities.file_io import load_torch, save_json
from utilities import pose_analysis_yolo
from utilities import plot_utils


def main():
    # setup logger and file paths
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("pose_processing")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/video_analysis.log"))

    db_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos")
    pt_root = db_path / ".pt"
    pt_files = list(iter_pt_files(pt_root))

    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)

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
            results = video_analysis_yolo(pt_file, plots_dir, csv_dir, logger, results)

        except Exception as error:
            logger.exception("Failed to process %s: %s", pt_file, error)

    save_json(results, str(output_json), logger=logger)
    logger.info("Feature extraction complete. Results saved to %s", output_json)


def video_analysis_yolo(pt_file: Path, plots_dir: Path, csv_dir: Path, logger, results: dict) -> dict:

    data      = load_torch(str(pt_file), logger=logger)
    landmarks = data["landmarks"]        # (T, 17, 2)
    metadata  = data.get("metadata", {})

    features             = pose_analysis_yolo.extract_features(landmarks)
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
    speed_df         = pose_analysis_yolo.compute_joint_speeds_df(landmarks)
    accel_df         = pose_analysis_yolo.compute_joint_acceleration_df(landmarks)
    dist_hip_df      = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="hip")
    dist_shoulder_df = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="shoulder")
    dist_half_df     = pose_analysis_yolo.compute_joint_distances_df(landmarks, center="half_body")
    
    # ---- save CSVs ----
    merged_df = pose_analysis_yolo.merge_pose_dfs(
        speed_df, accel_df, dist_hip_df, dist_shoulder_df, dist_half_df
    )
    merged_df.to_csv(str(csv_dir / f"{stem}_merged.csv"), index=False)
    logger.info("  Saved CSVs for %s", stem)

    # ---- plot (all panels in one image) ----
    plot_utils.plot_pose_analysis(
        speed_df,
        accel_df,
        dist_half_df,
        dist_hip_df,
        dist_shoulder_df,
        save_path=str(plots_dir / f"{stem}_pose_analysis.png"),
        suptitle=stem,
    )
    logger.info("  Saved plots for %s", stem)
    
    return results


def iter_pt_files(root: Path):
    """Yield all .pt files recursively under *root*."""
    for pt_file in sorted(root.rglob("*.pt")):
        yield pt_file



if __name__ == "__main__":
    main()