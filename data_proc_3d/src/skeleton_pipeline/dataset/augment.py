"""Spatial data augmentation for root-relative H36M (T, 17, 3) skeleton
sequences, applied to RAW positions -- i.e. before
skeleton_pipeline.features.h36m_features.compute_all_features -- not to the
already-derived features, and not to the (frame-index-only) labels, which
neither depend on nor need re-deriving from either transform.

Two independent transforms, both operating "based on the origin" (pelvis,
joint 0, which is exactly (0, 0, 0) at every frame by this project's
root-relative convention -- see motionbert_lifter.py):

  - rotate_positions(): ONE random rigid rotation about the origin, shared
    across every frame and every joint of a clip. This must be a SINGLE
    rotation per clip, not a fresh random angle per frame -- rotating each
    frame independently would inject fake inter-frame motion (a spurious
    angular velocity) that corrupts the velocity/acceleration features
    computed downstream. One shared rotation, by contrast, is a rigid
    relabeling of the coordinate frame: it changes each joint's
    x/y/z-component features (position/velocity/acceleration/azimuth) but
    leaves every rotation-invariant feature (speed/accel magnitude, joint
    angles, distance ratios) exactly unchanged -- a "the camera/subject was
    facing a different way" augmentation, not a fake-motion one. Defaults
    to yaw-only (rotate about the vertical Z axis) since that is the
    natural nuisance variation for this dataset (subject's facing
    direction relative to the camera is arbitrary; which way is "up" is
    not, and a full random 3D rotation would tilt "up" into a direction
    gravity-dependent features like elevation/z-position aren't meant to
    see in training).

  - add_joint_noise(): independent small per-frame, per-joint Gaussian
    jitter, simulating 2D-detector / MotionBERT-lifter position noise the
    model should be robust to. Pelvis (joint 0) is left untouched by
    default -- it is the origin/root by construction, not an independent
    measurement, so "noise" on it isn't meaningful the way it is for every
    other (relative-to-root) joint.

augment_positions() composes both (rotate first, then add noise -- for
isotropic Gaussian noise the order doesn't change the resulting
distribution, but rotating clean data first keeps the noise sigma
interpretable directly in meters regardless of the random rotation drawn).
"""
from dataclasses import dataclass, field

import numpy as np

PELVIS = 0


@dataclass
class AugmentationConfig:
    rotate: bool = True
    rotation_axis: str = "z"                     # "z" (yaw/vertical, default), "x", "y", or "xyz" (full random 3D)
    rotation_range_deg: tuple[float, float] = (0.0, 360.0)
    noise: bool = True
    noise_sigma_m: float = 0.01                   # per-joint per-frame position jitter, meters (1 cm default)
    noise_exclude_joints: tuple[int, ...] = (PELVIS,)
    seed: int | None = None


def _rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(f"Unknown rotation axis: {axis!r} (expected 'x', 'y', or 'z')")


def rotate_positions(positions: np.ndarray, axis: str = "z",
                      angle_range_deg: tuple[float, float] = (0.0, 360.0),
                      rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """positions: (T, 17, 3). Draws ONE random rotation (see module
    docstring for why it must be one shared rotation, not per-frame) and
    applies it to every joint of every frame, pivoting on the origin
    (pelvis). axis="xyz" draws one independent random angle per axis and
    composes them (a full random 3D rotation) instead of a single-axis one.
    Returns (rotated_positions, params) where params records the drawn
    angle(s) for reproducibility/metadata."""
    rng = rng or np.random.default_rng()
    axes = list(axis) if axis == "xyz" else [axis]

    R = np.eye(3)
    angles = {}
    for a in axes:
        angle_deg = rng.uniform(*angle_range_deg)
        angles[a] = float(angle_deg)
        R = _rotation_matrix(a, angle_deg) @ R

    rotated = positions @ R.T  # (T, 17, 3) @ (3, 3).T, broadcasts over T and the 17 joints
    return rotated, {"axis": axis, "angle_deg": angles}


def add_joint_noise(positions: np.ndarray, sigma_m: float = 0.01,
                     exclude_joints: tuple[int, ...] = (PELVIS,),
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """positions: (T, 17, 3). Adds i.i.d. N(0, sigma_m^2) noise to each
    x/y/z component of every frame/joint independently, except
    *exclude_joints* (default: just the pelvis/root, see module
    docstring). NaN rows (missing detections) stay NaN -- noise on a
    missing frame isn't meaningful and would turn a clean "no detection"
    signal into a bogus small skeleton."""
    noisy = positions.copy()
    joint_mask = np.ones(positions.shape[1], dtype=bool)
    for j in exclude_joints:
        joint_mask[j] = False

    noise = rng.normal(0.0, sigma_m, size=positions.shape) if rng is not None \
        else np.random.default_rng().normal(0.0, sigma_m, size=positions.shape)
    noise[:, ~joint_mask, :] = 0.0
    noisy = noisy + noise
    return noisy


def augment_positions(positions: np.ndarray, config: AugmentationConfig) -> tuple[np.ndarray, dict]:
    """Applies config.rotate then config.noise (see module docstring for
    ordering rationale) to *positions* (T, 17, 3). Returns
    (augmented_positions, applied_params) -- applied_params is meant to be
    stashed in the output .pt's metadata for reproducibility/traceability."""
    rng = np.random.default_rng(config.seed)
    out = np.asarray(positions, dtype=np.float64).copy()
    applied = {"seed": config.seed}

    if config.rotate:
        out, rotation_params = rotate_positions(out, axis=config.rotation_axis,
                                                 angle_range_deg=config.rotation_range_deg, rng=rng)
        applied["rotation"] = rotation_params
    else:
        applied["rotation"] = None

    if config.noise:
        out = add_joint_noise(out, sigma_m=config.noise_sigma_m,
                               exclude_joints=config.noise_exclude_joints, rng=rng)
        applied["noise"] = {"sigma_m": config.noise_sigma_m, "exclude_joints": list(config.noise_exclude_joints)}
    else:
        applied["noise"] = None

    return out, applied
