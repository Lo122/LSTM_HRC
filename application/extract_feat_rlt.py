
from collections import deque
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "data_proc_2d" / "src"))
from filter_utils import RealTimeSGFilter, PerJointKinematicTracker
import feature_extraction
from yolo_pose_config import JOINT_ANGLE_TRIPLETS, JOINT_ANGLE_TRIPLETS_CAL, RATIO_BETWEEN_DISTS


# filtering and smoothing configuration
HISTORY_SIZE = 3
EXCLUDE_NODES = [13, 14, 15, 16]

FILTER_CONFIG = {
    "window_size": 7,  # Must be an odd number (e.g., 5, 7, 11)
    "poly_order": 3,   # Typically less than window_size
    "max_jump": 0.2,     # Max allowed jump in normalized coordinates (e.g., 0.5 means 50% of the frame)
    "max_hold_frames": 60, # Max frames to hold a lost joint before resetting
    "global_confidence_threshold": 0.5,  # Threshold for overall detection confidence
    "joint_confidence_threshold": 0.3,    # Threshold for individual joint confidence
}

# Keep a short rolling history for temporal features for velocity and acceleration.
hist_smoothed_kpts = deque(maxlen=HISTORY_SIZE)


def setup_filtering() -> PerJointKinematicTracker:

    _filter = RealTimeSGFilter(
            window_size=FILTER_CONFIG["window_size"],
            poly_order=FILTER_CONFIG["poly_order"]
            )
    
    kinematic_tracker = PerJointKinematicTracker(
            sg_filter=_filter,
            num_joints=17,
            dims_per_joint=2,
            max_jump=FILTER_CONFIG["max_jump"],  # Max allowed jump in normalized coordinates (e.g., 0.5 means 50% of the frame)
            max_hold_frames=FILTER_CONFIG["max_hold_frames"],  # Max frames to hold a lost joint before resetting
        )
    
    _filter.reset()
    kinematic_tracker.reset()

    return kinematic_tracker


def smooth_kpt(raw_kpts: torch.Tensor, kinematic_tracker: PerJointKinematicTracker) -> torch.Tensor:

    try:
        raw_kpts = _parse_filter_pose_data(raw_kpts, kinematic_tracker)
    
    except ValueError as e:
        print("Error processing raw keypoints: %s. Using zeros for this frame." % e)
        zeros = torch.zeros((kinematic_tracker.num_joints, kinematic_tracker.dims_per_joint), dtype=torch.float32)
        return zeros

    smoothed_kpts = kinematic_tracker.update(raw_kpts, joint_confidence=None, detection_confidence=None)  # Assuming no confidence scores are provided
    if smoothed_kpts is None:
        smoothed_kpts = torch.zeros_like(raw_kpts)

    return smoothed_kpts


def _update_landmark_history(smoothed_kpt: torch.Tensor) -> torch.Tensor:
    """Append latest frame and return a fixed-size temporal window."""
    hist_smoothed_kpts.append(smoothed_kpt.clone())

    if len(hist_smoothed_kpts) < HISTORY_SIZE:
        first = hist_smoothed_kpts[0]
        pad_count = HISTORY_SIZE - len(hist_smoothed_kpts)
        window = [first.clone() for _ in range(pad_count)] + list(hist_smoothed_kpts)
        return torch.stack(window, dim=0)

    return torch.stack(list(hist_smoothed_kpts), dim=0)


def extract_features(smoothed_kpt: torch.Tensor, selected_feats: list[str] | None = None) -> dict[str, torch.Tensor]:
    landmarks = _update_landmark_history(smoothed_kpt)  # (HISTORY_SIZE, num_joints, 2)
    return _extract_features_from_landmarks(landmarks, selected_feats, EXCLUDE_NODES)


def _extract_features_from_landmarks(
    landmarks: torch.Tensor,
    selected_feats: list[str] | None,
    exclude_nodes: list[int],
) -> dict[str, torch.Tensor]:
    _features = {}

    # clean up NaN, inf, zero or any malicious values for tranning models in landmarks
    landmarks = feature_extraction._log_malicious_tensor(landmarks, "smoothed_kpt")
    landmarks = feature_extraction._fill_zero_frames_with_previous(landmarks, "smoothed_kpt")
    
    # velocity and acceleration (magnitude)
    include_node_list = [i for i in range(landmarks.shape[1]) if i not in exclude_nodes]
    velocity_scale, acceleration_scale, velocity_xy, acceleration_xy = feature_extraction.veclocity_acceleration_magnitude(landmarks)
    _features["velocity_scale"] = velocity_scale[:, include_node_list]
    _features["acceleration_scale"] = acceleration_scale[:, include_node_list]
    _features["velocity_xy"] = _flatten_features(velocity_xy[:, include_node_list, :])
    _features["acceleration_xy"] = _flatten_features(acceleration_xy[:, include_node_list, :])
    
    
    vectors, angluer_velocity, distance_velocity, angles, distance = feature_extraction.polar_coordinate_features(landmarks[:, include_node_list, :])
    _features["pol_vectors"] = _flatten_features(vectors)
    _features["pol_distance"] = distance
    _features["pol_angles"] = angles
    _features["pol_distance_velocity"] = distance_velocity
    _features["pol_angluer_velocity"] = angluer_velocity
        
    # angles at joints
    angles = []
    for _, a, vertex, c in JOINT_ANGLE_TRIPLETS:
        ang = feature_extraction.angle_at_joint(landmarks, a, vertex, c)  # (T,)
        angles.append(ang)
    
    for _, a, b, c in JOINT_ANGLE_TRIPLETS_CAL:
        middle_point = (landmarks[:, a, :] + landmarks[:, b, :]) / 2
        angle_values = feature_extraction.angle_at_joint(landmarks, a, middle_point, c)
        angles.append(angle_values)
        
    _features["joint_angles"] = torch.stack(angles, dim=1)  # (T, num_angles)
    
    # ratios between distances of keypoint pairs
    distance_ratios = []
    for _, first_dist_feature, second_dist_feature in RATIO_BETWEEN_DISTS:
        ratio = feature_extraction.distance_ratio(landmarks, first_dist_feature, second_dist_feature)
        distance_ratios.append(ratio)
    _features["ratios"] = torch.stack(distance_ratios, dim=1)  # (T, num_ratios)
    
    _features['dist_ratios'] \
        = feature_extraction.distance_from_center(landmarks[:, include_node_list, :],
                                                ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
                                                ("left_hip", "right_hip"))
    
    features = {}
    for feat_name, feat_value in _features.items():
        _feature = feature_extraction._log_malicious_tensor(feat_value, feat_name)
        features[feat_name] = _feature[-1, :, :] if _feature.ndim == 3 else _feature[-1, :]

    if selected_feats:
        return {k: v for k, v in features.items() if k in selected_feats}
    

    return features


def _parse_filter_pose_data(tensor: torch.Tensor, kinematic_tracker: PerJointKinematicTracker) -> torch.Tensor:
        
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.detach().clone()
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        
        tensor = tensor.reshape(kinematic_tracker.num_joints,
                                kinematic_tracker.dims_per_joint)
        
        if torch.all(tensor == 0):
            raise ValueError("All elements in the tensor are zero")
        
        return tensor


def _flatten_features(feature: torch.Tensor) -> torch.Tensor:
    """Flatten the last two dimensions of *feature* into one."""
    T = feature.shape[0]
    return feature.view(T, -1)







def unit_test():
    # Test the feature extraction pipeline with dummy data
    kinematic_tracker = setup_filtering()
    
    # Create dummy raw keypoints (T=10 frames, 17 joints, 2D coordinates)
    T = 10
    for i in range(T):
        raw_kpts = torch.rand((1, 17, 2))
        smoothed_kpts = smooth_kpt(raw_kpts, kinematic_tracker)
        print(f"Frame {i}: Smoothed Keypoints:\n{smoothed_kpts}\n")
        
        # Extract features
        selected_feats = ["velocity_scale", "acceleration_scale", "pol_vectors", "pol_distance", "pol_angles", "joint_angles"]
        features = extract_features(smoothed_kpts, selected_feats)
        
        # Print the shapes of the extracted features
        for feat_name, feat_value in features.items():
            print(f"{feat_name}: shape={feat_value.shape}")


if __name__ == "__main__":
    unit_test()