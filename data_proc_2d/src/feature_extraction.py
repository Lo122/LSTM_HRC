
import logging
import sys
from pathlib import Path

import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parent
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))
    
from yolo_pose_config import KEYPOINT_NAMES

_KEYPOINT_INDEX_BY_NAME = {name: index for index, name in enumerate(KEYPOINT_NAMES)}
_LOGGER = logging.getLogger("pose_processing")



def veclocity_acceleration_magnitude(kpts: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # velocity and acceleration
    frame_count = kpts.shape[0]
    velocity = torch.diff(kpts, dim=0)
    acceleration = torch.diff(velocity, dim=0)
    
    velocity_scale = torch.norm(velocity, dim=2)
    acceleration_scale = torch.norm(acceleration, dim=2)

    velocity_scale = _log_malicious_tensor(
        _pad_to_frame_count(velocity_scale, frame_count),
        "veclocity_acceleration_magnitude.velocity_scale",
    )
    acceleration_scale = _log_malicious_tensor(
        _pad_to_frame_count(acceleration_scale, frame_count),
        "veclocity_acceleration_magnitude.acceleration_scale",
    )

    velocity_xy = _log_malicious_tensor(
        _pad_to_frame_count(velocity, frame_count),
        "veclocity_acceleration_magnitude.velocity_xy",
    )
    acceleration_xy = _log_malicious_tensor(
        _pad_to_frame_count(acceleration, frame_count), 
        "veclocity_acceleration_magnitude.acceleration_xy",
    )
    
    return velocity_scale, acceleration_scale, velocity_xy, acceleration_xy


def _pad_to_frame_count(values: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Pad the leading frames by repeating the first detected value.

    If no value has been detected yet, fall back to zeros for the missing
    frames so the output still matches the input frame count.
    """
    if values.shape[0] >= frame_count:
        return values

    pad_count = frame_count - values.shape[0]
    if values.shape[0] == 0:
        prefix = torch.zeros(
            (pad_count, *values.shape[1:]),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        prefix = values[:1].expand(pad_count, *values.shape[1:])

    return torch.cat([prefix, values], dim=0)


def polar_coordinate_features(kpts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # center point for polar coordinate (between mid-hip and mid-shoulder)
    mid_hip = (kpts[:, _KEYPOINT_INDEX_BY_NAME["left_hip"], :] + kpts[:, _KEYPOINT_INDEX_BY_NAME["right_hip"], :]) / 2
    mid_shoulder = (kpts[:, _KEYPOINT_INDEX_BY_NAME["left_shoulder"], :] + kpts[:, _KEYPOINT_INDEX_BY_NAME["right_shoulder"], :]) / 2
    center = (mid_hip + mid_shoulder) / 2
    
    vectors = _log_malicious_tensor(
        kpts - center[:, None, :],
        "polar_coordinate_features.vectors",
    )  # (T, 17, 2)
    angluer_velocity = torch.diff(torch.atan2(vectors[..., 1], vectors[..., 0]), dim=0).rad2deg()  # (T-1, 17)
    angluer_velocity = _log_malicious_tensor(
        _pad_to_frame_count(angluer_velocity, kpts.shape[0]),
        "polar_coordinate_features.angluer_velocity",
    )
    distance_velocity = torch.diff(vectors.norm(dim=2), dim=0)  # (T-1, 17)
    distance_velocity = _log_malicious_tensor(
        _pad_to_frame_count(distance_velocity, kpts.shape[0]),
        "polar_coordinate_features.distance_velocity",
    )
    angles = _log_malicious_tensor(
        torch.atan2(vectors[..., 1], vectors[..., 0]).rad2deg(),
        "polar_coordinate_features.angles",
    )  # (T, 17)
    distance = _log_malicious_tensor(
        vectors.norm(dim=2),
        "polar_coordinate_features.distance",
    )  # (T, 17)
    
    return vectors, angluer_velocity, distance_velocity, angles, distance
    

def angle_at_joint(
    kpts: torch.Tensor,
    a: int,
    vertex: int | torch.Tensor,
    c: int,
) -> torch.Tensor:
    """Compute the angle (degrees) at *vertex* across T frames.

    Args:
        kpts:   (T, 17, 2) normalised keypoints.
        a:      index of the first arm keypoint.
        vertex: index of the joint at which the angle is measured, or a
                per-frame tensor with shape ``(T, 2)``.
        c:      index of the second arm keypoint.

    Returns:
        (T,) tensor of angles in degrees.
    """
    if isinstance(vertex, int):
        vertex_points = kpts[:, vertex, :]
    elif torch.is_tensor(vertex):
        if vertex.ndim != 2 or vertex.shape != (kpts.shape[0], 2):
            raise ValueError(
                "Expected vertex tensor to have shape "
                f"({kpts.shape[0]}, 2), got {tuple(vertex.shape)}."
            )
        vertex_points = vertex.to(dtype=kpts.dtype, device=kpts.device)
    else:
        raise TypeError(
            "vertex must be either an integer keypoint index or a tensor "
            f"with shape ({kpts.shape[0]}, 2), got {type(vertex).__name__}."
        )

    va = kpts[:, a, :] - vertex_points   # (T, 2)
    vc = kpts[:, c, :] - vertex_points   # (T, 2)
    dot  = (va * vc).sum(dim=1)                    # (T,)
    norm = va.norm(dim=1) * vc.norm(dim=1) + 1e-8  # (T,)
    cos_angle = (dot / norm).clamp(-1.0, 1.0)
    return _log_malicious_tensor(
        torch.acos(cos_angle).rad2deg(),
        "angle_at_joint",
    )          # (T,)


def distance_between_keypoints(
    kpts: torch.Tensor,
    point_a_name: str,
    point_b_name: str,
) -> torch.Tensor:
    """Compute per-frame Euclidean distance between two named keypoints."""
    try:
        point_a_idx = _KEYPOINT_INDEX_BY_NAME[point_a_name]
        point_b_idx = _KEYPOINT_INDEX_BY_NAME[point_b_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown keypoint name in ratio config: {error.args[0]!r}."
        ) from error

    delta = kpts[:, point_a_idx, :] - kpts[:, point_b_idx, :]
    return _log_malicious_tensor(
        delta.norm(dim=1),
        "distance_between_keypoints",
    )


def distance_ratio(
    kpts: torch.Tensor,
    first_dist_feature: tuple[str, str],
    second_dist_feature: tuple[str, str],
) -> torch.Tensor:
    """Compute per-frame ratio between two configured segment distances."""
    first_distance = distance_between_keypoints(kpts, *first_dist_feature)
    second_distance = distance_between_keypoints(kpts, *second_dist_feature)
    return _log_malicious_tensor(
        first_distance / second_distance.clamp_min(1e-8),
        "distance_ratio",
    )


def distance_from_center(kpts: torch.Tensor, center_point_names: tuple[str, ...], base_point_names: tuple[str, ...]) -> torch.Tensor:
    """Compute per-frame distance of each keypoint from a center point."""
    center_points = torch.stack(
        [kpts[:, _KEYPOINT_INDEX_BY_NAME[name], :] for name in center_point_names],
        dim=0,
    ).mean(dim=0)  # (T, 2)
    base_points = torch.stack(
        [kpts[:, _KEYPOINT_INDEX_BY_NAME[name], :] for name in base_point_names],
        dim=0,
    ).mean(dim=0)  # (T, 2)
    
    base_dist = (base_points - center_points).norm(dim=1, keepdim=True)  # (T, 1)
    delta = kpts - center_points[:, None, :]  # (T, 17, 2)  
    dist = delta.norm(dim=2)  # (T, 17)
    
    dist = dist / base_dist
    return _log_malicious_tensor(dist, "distance_from_center")  # (T, 17)



def _log_malicious_tensor(values: torch.Tensor, feature_name: str) -> torch.Tensor:
    """Log a warning when a tensor output contains NaN, inf, or zero values."""
    if not (values.is_floating_point() or values.is_complex()):
        return values

    nan_mask = torch.isnan(values)
    if nan_mask.any().item():
        first_nan_indices = torch.nonzero(nan_mask, as_tuple=False)[:20].tolist()
        _LOGGER.warning(
            "%s contains %d NaN value(s); shape=%s; first_indices=%s",
            feature_name,
            int(nan_mask.sum().item()),
            tuple(values.shape),
            first_nan_indices,
        )
        
    if torch.isinf(values).any().item():
        inf_mask = torch.isinf(values)
        first_inf_indices = torch.nonzero(inf_mask, as_tuple=False)[:20].tolist()
        _LOGGER.warning(
            "%s contains %d infinite value(s); shape=%s; first_indices=%s",
            feature_name,
            int(inf_mask.sum().item()),
            tuple(values.shape),
            first_inf_indices,
        )
        
    if values.eq(0).all().item():
        _LOGGER.warning(
            "%s contains all zero values; shape=%s",
            feature_name,
            tuple(values.shape),
        )

    return values



def _fill_zero_frames_with_previous(
    values: torch.Tensor,
    feature_name: str,
) -> torch.Tensor:
    """Forward-fill all-zero frames with the previous valid frame value."""
    if values.ndim == 0 or values.shape[0] == 0:
        return values

    filled = values.clone()
    flattened = filled.reshape(filled.shape[0], -1)
    zero_frame_mask = flattened.eq(0).all(dim=1)

    replaced_count = 0
    last_valid_index: int | None = None
    for frame_index in range(filled.shape[0]):
        if zero_frame_mask[frame_index].item():
            if last_valid_index is not None:
                filled[frame_index] = filled[last_valid_index]
                replaced_count += 1
        else:
            last_valid_index = frame_index

    if replaced_count > 0:
        _LOGGER.warning(
            "%s had %d all-zero frame(s); filled from previous frame",
            feature_name,
            replaced_count,
        )

    return filled


def _flatten_features(feature: torch.Tensor) -> torch.Tensor:
    """Flatten the last two dimensions of *feature* into one."""
    T = feature.shape[0]
    return feature.view(T, -1)
