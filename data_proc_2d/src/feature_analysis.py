
import sys
from pathlib import Path

import torch

PROJECT_SRC_ROOT = Path(__file__).resolve().parent
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))
    
from yolo_pose_config import KEYPOINT_NAMES


_KEYPOINT_INDEX_BY_NAME = {name: index for index, name in enumerate(KEYPOINT_NAMES)}



def veclocity_acceleration_magnitude(kpts: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # velocity and acceleration
    frame_count = kpts.shape[0]
    velocity = torch.diff(kpts, dim=0)
    acceleration = torch.diff(velocity, dim=0)
    
    velocity_scale = torch.norm(velocity, dim=2)
    acceleration_scale = torch.norm(acceleration, dim=2)

    velocity_scale = _pad_to_frame_count(velocity_scale, frame_count)
    acceleration_scale = _pad_to_frame_count(acceleration_scale, frame_count)

    velocity_xy = _pad_to_frame_count(velocity, frame_count)
    acceleration_xy = _pad_to_frame_count(acceleration, frame_count)
    
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


def polar_coordinate_features(kpts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    # center point for polar coordinate (between mid-hip and mid-shoulder)
    mid_hip = (kpts[:, _KEYPOINT_INDEX_BY_NAME["left_hip"], :] + kpts[:, _KEYPOINT_INDEX_BY_NAME["right_hip"], :]) / 2
    mid_shoulder = (kpts[:, _KEYPOINT_INDEX_BY_NAME["left_shoulder"], :] + kpts[:, _KEYPOINT_INDEX_BY_NAME["right_shoulder"], :]) / 2
    center = (mid_hip + mid_shoulder) / 2
    
    vectors = kpts - center[:, None, :]  # (T, 17, 2)
    angles = torch.atan2(vectors[..., 1], vectors[..., 0]).rad2deg()  # (T, 17)
    
    return vectors, angles
    

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
    return torch.acos(cos_angle).rad2deg()          # (T,)


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
    return delta.norm(dim=1)


def distance_ratio(
    kpts: torch.Tensor,
    first_dist_feature: tuple[str, str],
    second_dist_feature: tuple[str, str],
) -> torch.Tensor:
    """Compute per-frame ratio between two configured segment distances."""
    first_distance = distance_between_keypoints(kpts, *first_dist_feature)
    second_distance = distance_between_keypoints(kpts, *second_dist_feature)
    return first_distance / second_distance.clamp_min(1e-8)



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
    
    dist /= base_dist
    return dist  # (T, 17)