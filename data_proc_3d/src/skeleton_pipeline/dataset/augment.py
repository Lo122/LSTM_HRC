"""Spatial data augmentation for root-relative H36M (T, 17, 3) skeleton
sequences, applied to RAW positions -- i.e. before
skeleton_pipeline.features.h36m_features.compute_all_features -- not to the
already-derived features, and not to the (frame-index-only) labels, which
neither depend on nor need re-deriving from either transform.

Three independent transforms, all operating "based on the origin" (pelvis,
joint 0, which is exactly (0, 0, 0) at every frame by this project's
root-relative convention -- see motionbert_lifter.py):

  - mirror_positions(): reflection in the sagittal plane -- negate x AND swap
    the left/right joint pairs. A reflection is not in the rotation group, so
    this is the one transform rotate_positions() cannot produce no matter what
    angle it draws, which is why it is the cheapest first augmentation to add.
    See mirror_positions' own docstring for why the index swap is not optional.

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

augment_positions() composes all three (mirror, then rotate, then add noise --
for isotropic Gaussian noise the order doesn't change the resulting
distribution, but transforming clean data first keeps the noise sigma
interpretable directly in meters regardless of the random rotation drawn).
"""
from dataclasses import dataclass, field

import numpy as np

PELVIS = 0

# Left/right joint pairs in this project's H36M order -- keep in sync with
# skeleton_pipeline/features/h36m_features.py's H36M_JOINT_NAMES:
#   0 pelvis, 1 r_hip, 2 r_knee, 3 r_ankle, 4 l_hip, 5 l_knee, 6 l_ankle,
#   7 spine, 8 thorax, 9 neck, 10 head,
#   11 l_shoulder, 12 l_elbow, 13 l_wrist, 14 r_shoulder, 15 r_elbow, 16 r_wrist
# The unpaired joints (pelvis, spine, thorax, neck, head) lie on the midline
# and only need their x negated.
MIRROR_JOINT_PAIRS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))


@dataclass
class AugmentationConfig:
    mirror: bool = False                          # opt-in: see mirror_positions on when it is valid
    rotate: bool = True
    rotation_axis: str = "z"                     # "z" (yaw/vertical, default), "x", "y", or "xyz" (full random 3D)
    rotation_range_deg: tuple[float, float] = (0.0, 360.0)
    noise: bool = True
    noise_sigma_m: float = 0.01                   # per-joint per-frame position jitter, meters (1 cm default)
    noise_exclude_joints: tuple[int, ...] = (PELVIS,)
    seed: int | None = None


def mirror_positions(positions: np.ndarray,
                      joint_pairs: tuple[tuple[int, int], ...] = MIRROR_JOINT_PAIRS) -> np.ndarray:
    """positions: (T, 17, 3) -> the same motion performed left-right reflected.

    TWO steps, and the second is not optional: negate x, AND swap the left and
    right joint indices. Negating x alone reflects the skeleton in space but
    leaves each joint in its original slot, so what was the right elbow now sits
    at a left-elbow position while still being READ as the right elbow. Every
    left/right feature then reports inverted, and the bone tree connects
    l_shoulder to what is geometrically a right arm -- an anatomically
    impossible body that trains the model on poses no person can adopt.
    Swapping the pairs restores a valid skeleton that happens to be mirrored.

    Applies to raw positions only, like every transform here: the derived
    features must be recomputed afterwards, not mirrored themselves.

    A reflection, unlike rotate_positions', is orientation-REVERSING, so it is
    genuinely new data rather than something a yaw angle could have produced.
    It leaves the labels alone (a task id says nothing about handedness) and
    preserves every bone length and joint angle, so no bone-length constraint is
    violated.

    WHEN IT IS VALID: only where handedness is a nuisance rather than part of
    the task. If every subject is right-handed and deployment is right-handed
    too, mirroring manufactures left-handed executions that never occur at test
    time -- usually still a useful regulariser, but the reason this is opt-in
    (AugmentationConfig.mirror defaults to False) and worth measuring against a
    no-augmentation baseline rather than assuming.

    Deterministic: there is exactly one reflection, so unlike the rotation and
    noise draws there is nothing random to seed. Callers wanting a mixed set
    should mirror some copies and not others.
    """
    mirrored = np.asarray(positions, dtype=np.float64).copy()
    mirrored[..., 0] *= -1.0

    left = [a for a, _ in joint_pairs]
    right = [b for _, b in joint_pairs]
    mirrored[:, left + right, :] = mirrored[:, right + left, :]
    return mirrored


def mirror_bone_length_targets(targets, edges,
                                joint_pairs: tuple[tuple[int, int], ...] = MIRROR_JOINT_PAIRS):
    """Reorder per-edge bone lengths to match mirror_positions()' output.

    Mirroring swaps the subject's left and right limbs, so the edge
    (l_shoulder, l_elbow) now measures what was the RIGHT upper arm. The
    recorded targets must follow, or the metadata describes a body the
    positions no longer have -- and these subjects are not symmetric: the
    lifter gives one of them an 11% (up to 0.15 m) left/right difference.

    targets: (n_edges,) lengths. edges: (n_edges, 2) joint index pairs, in the
    same order. Returns a new array of the same shape.
    """
    swap = {a: b for a, b in joint_pairs}
    swap.update({b: a for a, b in joint_pairs})

    edges = np.asarray(edges)
    targets = np.asarray(targets, dtype=float)
    lookup = {frozenset((int(u), int(v))): value for (u, v), value in zip(edges, targets)}

    mirrored = targets.copy()
    for index, (u, v) in enumerate(edges):
        counterpart = frozenset((swap.get(int(u), int(u)), swap.get(int(v), int(v))))
        if counterpart in lookup:
            mirrored[index] = lookup[counterpart]
    return mirrored


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

    # Mirror first: it is deterministic, so drawing it before the random
    # transforms keeps a given seed's rotation/noise identical whether or not
    # mirroring is on, which makes the two variants directly comparable.
    if config.mirror:
        out = mirror_positions(out)
    applied["mirror"] = bool(config.mirror)

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
