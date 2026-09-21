"""Anchor a MotionBERT root-relative lifted skeleton (see motionbert_lifter.py)
to an ABSOLUTE, metric position in the camera frame using real depth-sensor
data (LiDAR/TrueDepth, from a processed .r3d capture -- see
app/iphone_lidar_test.py and app/r3d_skeleton_pipeline.py).

This is an alternative to metric_depth_estimator.py's anthropometric
(assumed body height) depth estimate, for when actual per-pixel depth is
available -- a plain monocular RGB video has neither.

Why split it this way (MotionBERT for SHAPE, depth for SCALE+POSITION)
instead of just using the depth-measured points directly as the skeleton:
depth data is sparse (only currently-visible, in-range, non-reflective
surface points have a reading) and noisy/holey (NaN at occlusion
boundaries, LiDAR max range, dark/reflective clothing), so it alone can't
give a clean full 17-joint skeleton every frame. MotionBERT's output is the
opposite: a clean, occlusion-robust full skeleton, but only root-relative
and "screen scaled" -- its coordinates are in ``crop_scale``'s normalized
image-space units, not meters (see motionbert_lifter.py's docstring: "only
the root-relative SHAPE is ever kept"). Combining them: fit_scale_translation
finds the single scalar scale + 3D translation that best maps MotionBERT's
relative skeleton onto however many depth-measured joint positions are
available this frame (least squares, NOT a full rotation-fitting Procrustes
-- MotionBERT's own output is already expressed in this same camera's local
axes, see motionbert_lifter.py's axis-remap docstring, so fitting a rotation
here would let a handful of noisy depth samples override a posture
MotionBERT already got right).

MetricSkeletonAnchor.update() falls back to a temporally-smoothed scale
(1-Euro filtered, same idea as metric_depth_estimator.py's
_PersonDepthState.last_depth "hold last good value") when too few joints
have usable depth to solve for scale this particular frame -- e.g. a frame
where only the torso is depth-visible still gets correctly placed by
translation, using whatever scale was last well-determined.
"""
import numpy as np

from skeleton_pipeline.metric_depth_estimator import OneEuroFilter


def sample_depth(depth_frame, u, v, color_size, conf_frame=None, min_confidence=1,
                  patch_radius=2):
    """Robust depth (meters) at color-image pixel (u, v), or None if no
    usable reading nearby.

    depth_frame may be a different resolution than the color image the
    pixel coordinates came from (e.g. LiDAR depth is often much lower-res
    than the color camera -- see iphone_lidar_test.py's --align-depth-to-rgb,
    which this function makes unnecessary by just rescaling the query point
    instead). color_size: (width, height) of the image (u, v) is in.

    Takes the median over a (2*patch_radius+1)^2 pixel window around the
    mapped point, ignoring NaN (invalid depth) and, if conf_frame is given,
    pixels below min_confidence (Record3D's 0/1/2 LiDAR confidence scale --
    see iphone_lidar_test.py's module docstring) -- a single depth pixel is
    noisy and the joint's true surface point rarely lands exactly on the
    pixel center anyway.
    """
    if u is None or v is None or not (np.isfinite(u) and np.isfinite(v)):
        return None
    dh, dw = depth_frame.shape[:2]
    w, h = color_size
    du = u * (dw / w)
    dv = v * (dh / h)
    if not (0 <= du < dw and 0 <= dv < dh):
        return None
    ci, cj = int(round(dv)), int(round(du))
    i0, i1 = max(ci - patch_radius, 0), min(ci + patch_radius + 1, dh)
    j0, j1 = max(cj - patch_radius, 0), min(cj + patch_radius + 1, dw)

    patch = depth_frame[i0:i1, j0:j1]
    if conf_frame is not None and conf_frame.size:
        conf_patch = conf_frame[i0:i1, j0:j1]
        patch = np.where(conf_patch >= min_confidence, patch, np.nan)

    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def back_project(u, v, depth_m, K):
    """Pixel (u, v) + depth (meters) -> camera-frame (X, Y, Z) meters,
    OpenCV convention (+X right, +Y down, +Z forward) -- same convention
    camera_utils/transforms.py and the rest of this project's camera code
    uses. K: 3x3 pinhole intrinsic matrix (calibration_io.load_intrinsics)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    return np.array([X, Y, depth_m], dtype=np.float64)


def camera_to_worldlike(xyz_camera):
    """Re-express an OpenCV-convention camera-frame point (+X right, +Y
    down, +Z forward) in the SAME "world-like" axes MotionBERT's output
    uses (+X right, +Y forward/depth, +Z up) -- the exact axis permutation
    motionbert_lifter.py's _postprocess_axes applies to its own raw output,
    duplicated here (not imported, since that name is private to that
    module) so depth-measured points and MotionBERT's skeleton live in the
    same axes before fit_scale_translation compares them."""
    xyz_camera = np.asarray(xyz_camera, dtype=np.float64)
    x, y, z = xyz_camera[..., 0], xyz_camera[..., 1], xyz_camera[..., 2]
    return np.stack([x, z, -y], axis=-1)


def fit_scale_translation(relative_pts, measured_pts):
    """Least-squares scalar scale `s` + translation `t` (3,) minimizing
    sum |measured_pts_i - (s * relative_pts_i + t)|^2 -- the closed-form
    1-D least-squares solution (centered dot products), NOT a full
    Umeyama/Procrustes fit: deliberately no rotation term, see module
    docstring. relative_pts/measured_pts: (N, 3), N >= 2, same point
    correspondence/order (same joint each row)."""
    relative_pts = np.asarray(relative_pts, dtype=np.float64)
    measured_pts = np.asarray(measured_pts, dtype=np.float64)
    p_bar = relative_pts.mean(axis=0)
    q_bar = measured_pts.mean(axis=0)
    pc = relative_pts - p_bar
    qc = measured_pts - q_bar
    denom = float(np.sum(pc * pc))
    s = float(np.sum(pc * qc)) / denom if denom > 1e-12 else 1.0
    t = q_bar - s * p_bar
    return s, t


class MetricSkeletonAnchor:
    """Per-track state: turns one frame's (MotionBERT root-relative
    skeleton, 2D H36M pixel positions, depth frame) into an absolute,
    metric, camera-frame skeleton -- see module docstring for the method.
    """

    def __init__(self, min_points=3, min_confidence=1, patch_radius=2,
                 scale_min_cutoff=0.5, scale_beta=0.01,
                 translation_min_cutoff=1.0, translation_beta=0.02):
        """
        min_points: minimum number of joints with a usable depth reading
            needed to (re-)solve for scale this frame -- below this, the
            last successfully solved scale is reused (held), since fitting
            a scale from e.g. 1-2 noisy points is unreliable. Translation
            is still updated from however many points ARE available (even
            just 1), since a translation-only fit degrades gracefully.
        min_confidence, patch_radius: passed to sample_depth per joint.
        scale_*/translation_*: OneEuroFilter params (see that class) for
            temporally smoothing the fitted scale (scalar) and translation
            (filtered independently per X/Y/Z component).
        """
        self.min_points = min_points
        self.min_confidence = min_confidence
        self.patch_radius = patch_radius
        self._scale_filter = OneEuroFilter(scale_min_cutoff, scale_beta)
        self._translation_filters = [
            OneEuroFilter(translation_min_cutoff, translation_beta) for _ in range(3)]
        self._last_scale = None

    def reset(self):
        self._scale_filter.reset()
        for f in self._translation_filters:
            f.reset()
        self._last_scale = None

    def update(self, skeleton_3d_relative, keypoints_h36m_xy, depth_frame, K,
               conf_frame=None, color_size=None, timestamp=None):
        """
        skeleton_3d_relative: (17, 3) H36M-order, root-relative, MotionBERT
            "world-like" axes (motionbert_lifter.MotionBERTStreamingLifter.lift's
            output -- optionally already run through BoneLengthConstraintFilter,
            which only changes shape stability, not the scale/translation
            fitting here).
        keypoints_h36m_xy: (17, 2) H36M-order PIXEL coords in the color
            image -- i.e. skeleton_pipeline.coco_h36m.coco_to_h36m_xy's
            output on the same frame's YOLO 2D detection (NOT the 3D
            skeleton's own x/y -- those are relative/unitless).
        depth_frame, conf_frame: this frame's (dh, dw) arrays from
            iphone_lidar_test.py's depth.npz (conf_frame may be None).
        K: 3x3 color-camera intrinsic matrix (calibration_io.load_intrinsics).
        color_size: (width, height) of the image keypoints_h36m_xy is in.
            Defaults to depth_frame's own size (i.e. assumes depth and color
            share a resolution -- pass explicitly whenever they don't, see
            sample_depth).

        Returns (skeleton_3d_absolute, scale, translation), or None if not
        even one joint had usable depth AND no scale/translation has ever
        been solved yet (nothing to hold).
        """
        skeleton_3d_relative = np.asarray(skeleton_3d_relative, dtype=np.float64)
        keypoints_h36m_xy = np.asarray(keypoints_h36m_xy, dtype=np.float64)
        if color_size is None:
            dh, dw = depth_frame.shape[:2]
            color_size = (dw, dh)

        relative_pts, measured_pts = [], []
        for j in range(skeleton_3d_relative.shape[0]):
            u, v = keypoints_h36m_xy[j]
            depth_m = sample_depth(depth_frame, u, v, color_size, conf_frame=conf_frame,
                                    min_confidence=self.min_confidence,
                                    patch_radius=self.patch_radius)
            if depth_m is None:
                continue
            cam_xyz = back_project(u, v, depth_m, K)
            measured_pts.append(camera_to_worldlike(cam_xyz))
            relative_pts.append(skeleton_3d_relative[j])

        if len(measured_pts) >= self.min_points:
            relative_pts = np.array(relative_pts)
            measured_pts = np.array(measured_pts)
            scale, translation = fit_scale_translation(relative_pts, measured_pts)
            scale = self._scale_filter(scale, timestamp)
            self._last_scale = scale
        elif self._last_scale is not None and measured_pts:
            # Not enough points to trust a fresh scale fit -- hold the last
            # good scale and solve translation only from what we do have.
            scale = self._last_scale
            relative_pts = np.array(relative_pts)
            measured_pts = np.array(measured_pts)
            translation = (measured_pts - scale * relative_pts).mean(axis=0)
        else:
            return None  # never solved a scale, and nothing usable this frame either

        translation = np.array(
            [f(translation[i], timestamp) for i, f in enumerate(self._translation_filters)])

        skeleton_3d_absolute = skeleton_3d_relative * scale + translation
        return skeleton_3d_absolute, scale, translation
