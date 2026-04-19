
import os
import sys
from pathlib import Path
import re

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

if str(PROJECT_SRC_ROOT / "data_proc_2d") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT / "data_proc_2d"))
from utilities import log_utils, file_io
from src.file_io_utils import load_torch, save_torch
from src import feature_analysis
from src.yolo_pose_config import JOINT_ANGLE_TRIPLETS, JOINT_ANGLE_TRIPLETS_CAL, RATIO_BETWEEN_DISTS



def main():
    # setup logger and file paths
    folder_root = Path(__file__).parents[2]
    logger = log_utils.setup_logger("pose_processing")
    log_utils.save_log_to_file(logger, str(folder_root / "logs/video_analysis.log"))

    google_drive_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
    pt_root = folder_root / r"data_proc_2d\dataset\train\raw"
    pt_files = list(iter_pt_files(pt_root))
    logger.info("Found %s .pt files under %s", len(pt_files), pt_root)
    
    out_pt_dir   = google_drive_path / "dataset" / "train" / "raw"
   
    # main loop to process each .pt file and extract features
    for pt_file in pt_files:
        logger.info("Processing %s", pt_file.name)
        try:
            features = video_analysis_yolo(pt_file, logger)
            save_features(features, out_pt_dir / pt_file.name, logger)

        except Exception as error:
            logger.exception("Failed to process %s: %s", pt_file, error)



def video_analysis_yolo(pt_file: Path, logger) -> dict:

    data      = load_torch(str(pt_file), logger=logger)
    
    metadata  = data.get("metadata", {})
    landmarks = data["proc_landmarks"]         # (T, 17, 2)
    
    features = {}
    features["metadata"] = metadata
    
    # velocity and acceleration (magnitude)
    velocity_scale, acceleration_scale, velocity_xy, acceleration_xy = feature_analysis.veclocity_acceleration_magnitude(landmarks)
    features["velocity_scale"] = velocity_scale
    features["acceleration_scale"] = acceleration_scale
    features["velocity_xy"] = flatten_features(velocity_xy)
    features["acceleration_xy"] = flatten_features(acceleration_xy)
    
    
    vectors, angles = feature_analysis.polar_coordinate_features(landmarks)
    features["pol_vectors"] = flatten_features(vectors)
    features["pol_angles"] = angles
    
        
    # angles at joints
    angles = []
    for name, a, vertex, c in JOINT_ANGLE_TRIPLETS:
        ang = feature_analysis.angle_at_joint(landmarks, a, vertex, c)  # (T,)
        angles.append(ang)
    
    for name, a, b, c in JOINT_ANGLE_TRIPLETS_CAL:
        middle_point = (landmarks[:, a, :] + landmarks[:, b, :]) / 2
        angle_values = feature_analysis.angle_at_joint(landmarks, a, middle_point, c)
        angles.append(angle_values)
        
    features["joint_angles"] = torch.stack(angles, dim=1)  # (T, num_angles)
    
    # ratios between distances of keypoint pairs
    distance_ratios = []
    for name, first_dist_feature, second_dist_feature in RATIO_BETWEEN_DISTS:
        ratio = feature_analysis.distance_ratio(landmarks, first_dist_feature, second_dist_feature)
        distance_ratios.append(ratio)
    features["ratios"] = torch.stack(distance_ratios, dim=1)  # (T, num_ratios)
    
    features['dist_ratios'] \
        = feature_analysis.distance_from_center(landmarks,
                                                ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
                                                ("left_hip", "right_hip"))
    
    
    return features
    
    
def save_features(features: dict, file_path: Path, logger) -> None:
    
    # regex 
    pattern = r"^video__(cam-\d{2}_uid-\d{2}_take-\d{2})_pose\.pt$"
    replacement = r"features__\1.pt"
    
    # save features to .pt file
    file_name = file_path.name
    new_file_name = re.sub(pattern, replacement, file_name)
    
    output_pt = file_path.parent / new_file_name
    num_frames = features.get("num_frames", 0)
    save_torch(features, str(output_pt), logger=logger)
    logger.info("Saved %s frames to %s", num_frames, output_pt)



def flatten_features(feature: torch.Tensor) -> torch.Tensor:
    """Flatten the last two dimensions of *feature* into one."""
    T = feature.shape[0]
    return feature.view(T, -1)

def iter_pt_files(root: Path):
    """Yield all .pt files recursively under *root*."""
    for pt_file in sorted(root.rglob("*.pt")):
        yield pt_file



if __name__ == "__main__":
    # main()
    file_path = Path(r"G:\My Drive\University of Stuttgart\ITECH_Thesis\Videos\dataset\train\raw\features__cam-01_uid-01_take-01.pt")
    data = load_torch(str(file_path))
    print(data.keys())