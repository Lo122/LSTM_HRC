import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

if str(PROJECT_SRC_ROOT / "data_proc_2d") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT / "data_proc_2d"))

from utilities import log_utils, file_io
from src.file_io_utils import load_torch, load_step_ids_from_json, load_elan_label_data, iter_files
from src import pose_analysis_yolo
from src import plot_utils
from src.annotation_config import ANNOTATION_CONFIG
from data_proc_2d.src import feature_extraction
from src.yolo_pose_config import (
    JOINT_ANGLE_TRIPLETS,
    JOINT_ANGLE_TRIPLETS_CAL,
    RATIO_BETWEEN_DISTS,
)


IS_SOURCE_PT = True  # Set to True if the .pt files are in the Google Drive path; False if they are in the local data_proc_2d directory

FEATURE_PANEL_CONFIG : list[tuple[str, str, str, str]] = [
    ("velocity_scale_db", "Speed (norm. units / frame)", "Joint Speed", "log"),
    (
        "acceleration_scale_db",
        "Acceleration (norm. units / frame²)",
        "Joint Acceleration",
        "log",
    ),
    (
        "velocity_x_db",
        "Velocity component X (norm. units / frame)",
        "Joint Velocity X",
        "log",
    ),
    (
        "velocity_y_db",
        "Velocity component Y (norm. units / frame)",
        "Joint Velocity Y",
        "log",
    ),
    (
        "acceleration_x_db",
        "Acceleration component X (norm. units / frame²)",
        "Joint Acceleration X",
        "log",
    ),
    (
        "acceleration_y_db",
        "Acceleration component Y (norm. units / frame²)",
        "Joint Acceleration Y",
        "log",
    ),
    ("pol_vectors_x_db", "Polar vector component X", "Polar Vectors X", "linear"),
    ("pol_vectors_y_db", "Polar vector component Y", "Polar Vectors Y", "linear"),
    ("pol_angles_db", "Polar angle (deg)", "Polar Angles", "linear"),
    ("joint_angles_db", "Angle (deg)", "Joint Angles", "linear"),
    ("ratios_db", "Ratio", "Distance Ratios", "log"),
    (
        "dist_ratios_db",
        "Distance from center ratio",
        "Distance from Center",
        "linear",
    ),
]


def main():
    # setup logger and file path
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("pose_processing")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/video_analysis.log"))

    # I/O paths
    io_root = folder_root / "data_proc_2d" / "results" / "video_analysis_yolo_smoothed"
    db_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
    pt_root =  folder_root / r"data_proc_2d\dataset\train\raw" if IS_SOURCE_PT else db_path / "dataset" / "original"
    label_root = db_path / "annotations"/ "cam-04"
    anno_root = db_path / "dataset" / "annotations"
    
    plot_dir = io_root / "plots"
    csv_dir   = io_root / "source"
    output_json = io_root / "source"  / "feature_results.json"
    
    
    # load pt files
    pt_files = list(iter_files(pt_root, extension=".pt"))
    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)
    
    # label files: map video file name to list of step labels
    label_files = list(iter_files(label_root, extension=".json"))
    step_label_list = {}
    for label_file in label_files:
        logger.info("Found label file: %s", label_file)
        label_info, video_file_name = load_step_ids_from_json(label_file, logger=logger)
        step_label_list[video_file_name] = {"labels": label_info}

    # ELAN label files: map video file name to list of step labels 
    label_files = list(iter_files(anno_root, extension=".json"))
    elan_step_label_list = {}
    for label_file in label_files:
        logger.info("Found ELAN label file: %s", label_file)
        video_file_name = f"video__{label_file.stem.split('__')[1]}.mp4"  # Extract video file name from ELAN label file name
        label_info = file_io.load_json(str(label_file), logger=logger)
        elan_step_label_list[video_file_name] = {"labels": label_info}

    
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # main loop to process each .pt file and extract features
    results: dict = {}
    feature_dataframes_by_video: dict[str, dict[str, pd.DataFrame]] = {}
    annotations_by_video: dict[str, dict[str, list[dict]]] = {}
    for pt_file in pt_files:
        if pt_file.stem not in ["keypoints_test_02__cam-02_uid-01_take-02"]:
            continue

        logger.info("Processing %s", pt_file.name)
        if IS_SOURCE_PT:
            pt_data = extract_features(pt_file, logger)
        else:
            pt_data = load_torch(str(pt_file), logger=logger)
            
        metadata  = pt_data.get("metadata", {})
        features = {k: v for k, v in pt_data.items() if k != "metadata"}
        annotations = find_annotations(metadata, step_label_list, elan_step_label_list, logger)
        video_name = Path(metadata.get("video", pt_file.stem)).stem
        video_name = video_name.split("__", maxsplit=1)[1] if "__" in video_name else video_name
        feature_dataframes_by_video[video_name] = pose_analysis_yolo.build_feature_dataframes(features)
        annotations_by_video[video_name] = annotations
        
        results[pt_file.stem] = plot_features(features, annotations, plot_dir, csv_dir, metadata, logger)


    # results["_all_videos_per_feature"] = plot_all_users_per_feature(
    #     feature_dataframes_by_video,
    #     annotations_by_video,
    #     plot_dir,
    #     csv_dir,
    #     logger,
    # )

    file_io.save_json(results, str(output_json), logger=logger)
    logger.info("Feature extraction complete. Results saved to %s", output_json)



def extract_features(pt_file: Path, logger) -> dict:

    data = load_torch(str(pt_file), logger=logger)
    
    metadata = data.get("metadata", {})
    try:
        landmarks = data["smoothed_landmarks"]         # (T, 17, 2)
    except KeyError:
        logger.warning("No smoothed_landmarks found in %s, falling back to proc_landmarks", pt_file)
        landmarks = data["proc_landmarks"]         # (T, 17, 2)
        
    features = {}
    features["metadata"] = metadata
    
    # clean up NaN, inf, zero or any malicious values for tranning models in landmarks
    landmarks = feature_extraction._log_malicious_tensor(landmarks, "landmarks")
    landmarks = feature_extraction._fill_zero_frames_with_previous(landmarks, "landmarks")
    
    # velocity and acceleration (magnitude)
    velocity_scale, acceleration_scale, velocity_xy, acceleration_xy = feature_extraction.veclocity_acceleration_magnitude(landmarks)
    features["velocity_scale"] = velocity_scale
    features["acceleration_scale"] = acceleration_scale
    features["velocity_x"] = velocity_xy[:, :, 0]
    features["velocity_y"] = velocity_xy[:, :, 1]

    features["acceleration_x"] = acceleration_xy[:, :, 0]
    features["acceleration_y"] = acceleration_xy[:, :, 1]
    
    
    vectors, angles = feature_extraction.polar_coordinate_features(landmarks)
    features["pol_vectors_x"] = vectors[:, :, 0]
    features["pol_vectors_y"] = vectors[:, :, 1]
    features["pol_angles"] = angles
    
        
    # angles at joints
    angles = []
    for _, a, vertex, c in JOINT_ANGLE_TRIPLETS:
        ang = feature_extraction.angle_at_joint(landmarks, a, vertex, c)  # (T,)
        angles.append(ang)
    
    for _, a, b, c in JOINT_ANGLE_TRIPLETS_CAL:
        middle_point = (landmarks[:, a, :] + landmarks[:, b, :]) / 2
        angle_values = feature_extraction.angle_at_joint(landmarks, a, middle_point, c)
        angles.append(angle_values)
        
    features["joint_angles"] = torch.stack(angles, dim=1)  # (T, num_angles)
    
    # ratios between distances of keypoint pairs
    distance_ratios = []
    for _, first_dist_feature, second_dist_feature in RATIO_BETWEEN_DISTS:
        ratio = feature_extraction.distance_ratio(landmarks, first_dist_feature, second_dist_feature)
        distance_ratios.append(ratio)
    features["ratios"] = torch.stack(distance_ratios, dim=1)  # (T, num_ratios)
    
    features['dist_ratios'] \
        = feature_extraction.distance_from_center(landmarks,
                                                ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
                                                ("left_hip", "right_hip"))
    
    return features



def plot_features(features: dict, annotations: dict, plot_dir: Path, csv_dir: Path, metadata: dict, logger)  -> dict:

    feature_dfs = pose_analysis_yolo.build_feature_dataframes(features)
    features_db = pd.concat(feature_dfs.values(), axis=1) if feature_dfs else pd.DataFrame()
    panel_data = pose_analysis_yolo.build_panel_data(feature_dfs, FEATURE_PANEL_CONFIG)
    file_name = Path(metadata.get("video", "unknown_video")).stem
    
    plot_utils.plot_features(
        panel_data=panel_data,
        annotations=annotations,
        save_path= str(plot_dir / f"{file_name}_features.png"),
        suptitle=f"Video: {file_name}",
    )
    
    features_db.to_csv(str(csv_dir / f"{file_name}_features.csv"), index=False)
    
    return {
        "feature_dataframes": {
            dataframe_name: feature_df.columns.tolist()
            for dataframe_name, feature_df in feature_dfs.items()
        },
        "panel_titles": [panel.title for panel in panel_data],
        "feature_columns": features_db.columns.tolist(),
        "num_frames": int(features_db.shape[0]),
        "num_features": int(features_db.shape[1]),
        "annotations": annotations,
    }
    


def plot_all_users_per_feature(
    feature_dataframes_by_video: dict[str, dict[str, pd.DataFrame]],
    annotations_by_video: dict[str, dict[str, list[dict]]],
    plot_dir: Path,
    csv_dir: Path,
    logger,
) -> dict[str, dict[str, object]]:
    """Create one plot per feature group with alternating feature and annotation panels per video."""
    summary: dict[str, dict[str, object]] = {}

    for dataframe_name, ylabel, panel_title, yscale in FEATURE_PANEL_CONFIG:
        prefixed_frames: list[pd.DataFrame] = []
        included_videos: list[str] = []
        video_panels: list[plot_utils.VideoFeaturePanels] = []

        for video_name, feature_dataframes in feature_dataframes_by_video.items():
            feature_df = feature_dataframes.get(dataframe_name)
            if feature_df is None or feature_df.empty:
                continue
            
            prefixed_frames.append(feature_df.add_prefix(f"{video_name}__"))
            included_videos.append(video_name)
            video_panels.append(
                plot_utils.VideoFeaturePanels(
                    video_name=video_name,
                    feature_panel=plot_utils.PanelData(
                        df=feature_df,
                        names=list(feature_df.columns),
                        cols=list(feature_df.columns),
                        ylabel=ylabel,
                        title=f"{panel_title}: {video_name}",
                        yscale=yscale,
                    ),
                    annotations=annotations_by_video.get(video_name, {}),
                )
            )

        if not prefixed_frames or not video_panels:
            continue

        combined_df = pd.concat(prefixed_frames, axis=1)
        combined_columns = list(combined_df.columns)

        plot_path = plot_dir / f"{dataframe_name}_all_videos.png"
        csv_path = csv_dir / f"{dataframe_name}_all_videos.csv"
        plot_utils.plot_features_by_video(
            video_panels=video_panels,
            save_path=str(plot_path),
            suptitle=f"{panel_title} Across All Videos",
        )
        combined_df.to_csv(csv_path, index=True, index_label="frame")

        logger.info(
            "Saved cross-video feature plot %s with %d video(s) and %d trace(s)",
            plot_path.name,
            len(included_videos),
            len(combined_columns),
        )
        summary[dataframe_name] = {
            "plot_path": str(plot_path),
            "csv_path": str(csv_path),
            "num_videos": len(included_videos),
            "num_traces": len(combined_columns),
            "videos": included_videos,
            "columns": combined_columns,
        }

    return summary
    
    
    

    
def find_annotations(metadata: dict, step_label_list: dict,
                     elan_step_label_list: dict, logger) -> dict:
    
    video_path = metadata.get("video")
    video_file_name = os.path.basename(video_path) if video_path else "unknown"
    step_labels = step_label_list.get(video_file_name, {}).get("labels", [])
    elan_step_labels = elan_step_label_list.get(video_file_name, {}).get("labels", [])

    annotations = {}
    if step_labels:
        logger.info("  Found %d step labels for video %s", len(step_labels), video_file_name)
        annotations["step_labels"] = step_labels
    if elan_step_labels:
        logger.info("  Found %d ELAN step labels for video %s", len(elan_step_labels), video_file_name)
        annotations["elan_step_labels"] = elan_step_labels
        
    return annotations
    



if __name__ == "__main__":
    # file_path = Path(r"C:\Users\Owner\OneDrive - Universität Stuttgart\2025_26_Thesis\codes\LSTM_HRC\data_proc_2d\app\dataset\train\raw\video__cam_04_uid-07_take-1_pose.pt")
    # data      = load_torch(str(file_path))
    # print(data.keys())
    main()