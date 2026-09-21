"""3D human skeleton from a processed Record3D (.r3d) capture: YOLO 2D pose
-> MotionBERT DSTformer 2D->3D lifting for POSTURE (root-relative shape --
see motionbert_lifter.py: MotionBERT is only trusted for shape, not scale)
-> depth-sensor back-projection for absolute POSITION/SCALE (see
skeleton_pipeline/depth_anchor.py).

Consumes the output folder app/iphone_lidar_test.py's process_r3d writes
(rgb.mp4 + depth.npz + intrinsics.json) -- run that script on your .r3d
file first:
    uv run python iphone_lidar_test.py "C:\\Users\\Owner\\Downloads\\test\\2026-08-14--14-52-15\\Shareable\\2026-08-14--14-52-15.r3d" `
      --output-dir results/r3d 
    uv run python r3d_skeleton_pipeline.py results/r3d/2026-08-14--14-52-15 --output-dir results/skeletons --device cuda:0

Otherwise a near-identical YOLO->MotionBERT->BoneLengthConstraintFilter
pipeline to generate_lstm_training_data.py -- see that script's docstring
for the shared reasoning (--clip-len=81 matching the live deployment
window, why BoneLengthConstraintFilter is reset per-capture, etc.). The
real difference is this script ALSO depth-anchors every frame's skeleton
(skeleton_pipeline.depth_anchor.MetricSkeletonAnchor) to get an absolute,
metric camera-frame skeleton -- something a plain RGB video (generate_lstm_
training_data.py's input) has no way to provide, since it has no depth
sensor to back-project against.

Saved <capture-stem>.npz keys:
  keypoints_2d          (T, 17, 2) COCO order, pixel coords.
  keypoints_3d_relative  (T, 17, 3) H36M order, root-relative, bone-length-
                         stabilized MotionBERT output (meters-SHAPED but not
                         metric-SCALED -- see module docstring).
  keypoints_3d_absolute  (T, 17, 3) H36M order, ABSOLUTE camera-frame
                         meters (NaN row where depth anchoring couldn't
                         place that frame at all -- see MetricSkeletonAnchor).
  anchor_scale           (T,) the fitted relative->metric scale used per
                         frame (see depth_anchor.fit_scale_translation).
  anchor_translation     (T, 3) the fitted/filtered translation (meters)
                         used per frame -- keypoints_3d_absolute[t] ==
                         keypoints_3d_relative[t] * anchor_scale[t] +
                         anchor_translation[t].
  timestamps, fps, K, image_size, bone_length_edges/targets, body_scale_m
                         (bone lengths are in keypoints_3d_relative's own
                         unitless scale, same as generate_lstm_training_
                         data.py -- use body_scale_m * median(anchor_scale)
                         for a metric height proxy if needed).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skeleton_pipeline.bone_length_filter import BoneLengthConstraintFilter
from skeleton_pipeline.calibration_io import load_intrinsics
from skeleton_pipeline.coco_h36m import coco_to_h36m_xy
from skeleton_pipeline.depth_anchor import MetricSkeletonAnchor, sample_depth
from skeleton_pipeline.keypoint_filter import KeypointOutlierHoldFilter
from skeleton_pipeline.motionbert_lifter import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, MotionBERTStreamingLifter,
)
from skeleton_pipeline.person_selection import select_tracked_person_keypoints
from skeleton_pipeline.render.skeleton_video import FastSkeleton3DRenderer, render_combined_frame

_HEIGHT_CHAIN_EDGES = [(0, 1), (1, 2), (2, 3), (0, 7), (7, 8), (8, 9), (9, 10)]


def run_yolo_2d(yolo, frame, previous_keypoints=None, previous_conf=None, imgsz=None):
    kwargs = {"imgsz": imgsz} if imgsz is not None else {}
    results = yolo(frame, verbose=False, **kwargs)
    if not results:
        return None, None
    frame_result = results[0]
    if frame_result.keypoints is None or frame_result.keypoints.xy.numel() == 0:
        return None, None

    all_keypoints_xy = frame_result.keypoints.xy.cpu().numpy()
    all_keypoints_conf = (frame_result.keypoints.conf.cpu().numpy()
                          if frame_result.keypoints.conf is not None else None)
    detection_scores = (frame_result.boxes.conf.cpu().numpy()
                        if frame_result.boxes is not None and frame_result.boxes.conf is not None else None)
    return select_tracked_person_keypoints(
        all_keypoints_xy,
        all_keypoints_conf,
        detection_scores=detection_scores,
        previous_keypoints=previous_keypoints,
        previous_conf=previous_conf,
    )


def load_r3d_capture(r3d_dir):
    """Load one app/iphone_lidar_test.py output folder's rgb.mp4 + depth.npz
    + intrinsics.json. Returns (cap, depth_npz, K, image_size, fps)."""
    r3d_dir = Path(r3d_dir)
    video_path = r3d_dir / "rgb.mp4"
    depth_path = r3d_dir / "depth.npz"
    intrinsics_path = r3d_dir / "intrinsics.json"
    for p in (video_path, depth_path, intrinsics_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run iphone_lidar_test.py on your .r3d file first "
                f"(expects rgb.mp4/depth.npz/intrinsics.json under {r3d_dir}).")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}.")
    depth_npz = np.load(depth_path)
    K, _dist, image_size = load_intrinsics(intrinsics_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or float(depth_npz.get("fps", 30.0))
    return cap, depth_npz, K, image_size, fps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("r3d_dir", type=str,
                         help="Output folder from iphone_lidar_test.py's process_r3d "
                              "(contains rgb.mp4, depth.npz, intrinsics.json).")
    parser.add_argument("--output-dir", type=str, default="results/skeletons",
                         help="One <r3d_dir-name>.npz written here.")
    parser.add_argument("--yolo-model", type=str, default="dataset/model/yolo26m-pose.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                         help="'cpu', 'cuda:0', etc. -- passed to both YOLO and MotionBERT.")
    parser.add_argument("--motionbert-config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--motionbert-checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--clip-len", type=int, default=81,
                         help="MotionBERT rolling-buffer window length -- see "
                              "generate_lstm_training_data.py's docstring on why 81, not 243.")
    parser.add_argument("--no-keypoint-filter", action="store_true")
    parser.add_argument("--no-bone-length-filter", action="store_true")
    parser.add_argument("--min-depth-points", type=int, default=3,
                         help="Minimum joints with a usable depth reading needed to (re-)solve "
                              "depth-anchoring scale this frame -- see MetricSkeletonAnchor.")
    parser.add_argument("--depth-min-confidence", type=int, default=1,
                         help="Record3D LiDAR confidence (0/1/2) required to trust a depth pixel. "
                              "Ignored for captures with no confidence data (e.g. TrueDepth).")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: cap frames processed.")
    parser.add_argument("--no-render-video", action="store_true",
                         help="Don't write a <stem>_render.mp4 (2D overlay | 3D 4-view, now in "
                              "true meters since the 3D panel is fed the depth-anchored absolute "
                              "skeleton) -- on by default.")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(640, 360))
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    r3d_dir = Path(__file__).resolve().parents[1] / args.r3d_dir
    if not r3d_dir.exists():
        raise FileNotFoundError(f"{r3d_dir} not found -- run iphone_lidar_test.py on your .r3d file first.")
    stem = r3d_dir.name
    output_dir = Path(__file__).resolve().parents[1] / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.npz"

    cap, depth_npz, K, image_size, fps = load_r3d_capture(r3d_dir)
    depth_all = depth_npz["depth"]
    conf_all = depth_npz["confidence"] if depth_npz["confidence"].size else None
    timestamps_all = depth_npz["timestamps"]
    n_depth_frames = depth_all.shape[0]
    print(f"{r3d_dir.name}: color {image_size}, {n_depth_frames} depth frames, {fps} fps, "
          f"confidence data: {conf_all is not None}.")

    print(f"Loading YOLO ({args.yolo_model}) on {args.device}...")
    yolo_model_path = Path(__file__).resolve().parents[1] / args.yolo_model
    yolo = YOLO(str(yolo_model_path))
    print(f"Loading MotionBERT (clip_len={args.clip_len}) on {args.device}...")
    lifter = MotionBERTStreamingLifter(
        config_path=args.motionbert_config, checkpoint_path=args.motionbert_checkpoint,
        clip_len=args.clip_len, device=args.device)

    kp_filter = None if args.no_keypoint_filter else KeypointOutlierHoldFilter()
    bone_filter = None if args.no_bone_length_filter else BoneLengthConstraintFilter()
    anchor = MetricSkeletonAnchor(min_points=args.min_depth_points,
                                   min_confidence=args.depth_min_confidence)

    renderer_3d = None if args.no_render_video else FastSkeleton3DRenderer(args.panel_size)
    writer = None
    if renderer_3d is not None:
        render_path = output_dir / f"{stem}_render.mp4"
        panel_w, panel_h = args.panel_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(render_path), fourcc, fps, (panel_w * 2, panel_h))

    all_2d, all_3d_rel, all_3d_abs = [], [], []
    all_scale, all_translation, all_timestamps = [], [], []
    frame_idx = 0
    detected_count = 0
    anchored_count = 0
    tracked_keypoints_2d = None
    tracked_keypoints_conf = None
    t0 = time.time()
    n_frames = n_depth_frames if args.max_frames is None else min(n_depth_frames, args.max_frames)

    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            print(f"  WARNING: rgb.mp4 ran out at frame {frame_idx} "
                  f"(depth.npz has {n_depth_frames}) -- stopping early.")
            break
        h, w = frame.shape[:2]

        keypoints_2d, keypoints_conf = run_yolo_2d(
            yolo,
            frame,
            previous_keypoints=tracked_keypoints_2d,
            previous_conf=tracked_keypoints_conf,
            imgsz=args.yolo_imgsz,
        )
        if kp_filter is not None:
            keypoints_2d, keypoints_conf, _status = kp_filter.filter(keypoints_2d, keypoints_conf)

        if keypoints_2d is not None and np.any(keypoints_2d):
            tracked_keypoints_2d = keypoints_2d
            tracked_keypoints_conf = keypoints_conf
        else:
            tracked_keypoints_2d = None
            tracked_keypoints_conf = None

        skeleton_3d_rel = None
        skeleton_3d_abs = np.full((17, 3), np.nan)
        scale_used, translation_used = np.nan, np.full(3, np.nan)
        if keypoints_2d is not None and np.any(keypoints_2d):
            detected_count += 1
            skeleton_3d_rel = lifter.lift(keypoints_2d, image_size=(w, h), keypoints_conf=keypoints_conf)
            if skeleton_3d_rel is not None and bone_filter is not None:
                skeleton_3d_rel = bone_filter.filter(skeleton_3d_rel)

            if skeleton_3d_rel is not None:
                keypoints_h36m_xy = coco_to_h36m_xy(keypoints_2d)
                depth_frame = depth_all[frame_idx]
                conf_frame = conf_all[frame_idx] if conf_all is not None else None
                result = anchor.update(
                    skeleton_3d_rel, keypoints_h36m_xy, depth_frame, K,
                    conf_frame=conf_frame, color_size=(w, h),
                    timestamp=float(timestamps_all[frame_idx]))
                if result is not None:
                    skeleton_3d_abs, scale_used, translation_used = result
                    anchored_count += 1

        all_2d.append(keypoints_2d if keypoints_2d is not None else np.zeros((17, 2)))
        all_3d_rel.append(skeleton_3d_rel if skeleton_3d_rel is not None else np.full((17, 3), np.nan))
        all_3d_abs.append(skeleton_3d_abs)
        all_scale.append(scale_used)
        all_translation.append(translation_used)
        all_timestamps.append(float(timestamps_all[frame_idx]))

        if writer is not None:
            render_skeleton = skeleton_3d_abs if np.isfinite(skeleton_3d_abs).all() else skeleton_3d_rel
            distances, pelvis_distance = None, None
            if keypoints_2d is not None:
                # Raw depth-sensor sample per COCO keypoint (independent of
                # whether MotionBERT/depth-anchoring succeeded this frame) --
                # purely a preview annotation, see draw_2d_skeleton.
                depth_frame = depth_all[frame_idx]
                conf_frame = conf_all[frame_idx] if conf_all is not None else None
                distances = np.array([
                    sample_depth(depth_frame, u, v, (w, h), conf_frame=conf_frame,
                                 min_confidence=args.depth_min_confidence) or np.nan
                    for u, v in keypoints_2d
                ])
                # Pelvis pixel <- COCO l_hip(11)/r_hip(12) midpoint, same
                # derivation as coco_to_h36m_xy's h36m[0] (H36M pelvis).
                pelvis_u, pelvis_v = (keypoints_2d[11] + keypoints_2d[12]) / 2.0
                pelvis_distance = sample_depth(
                    depth_frame, pelvis_u, pelvis_v, (w, h), conf_frame=conf_frame,
                    min_confidence=args.depth_min_confidence)
            combined = render_combined_frame(
                frame, keypoints_2d, keypoints_conf, render_skeleton, renderer_3d, args.panel_size,
                distances=distances, pelvis_distance=pelvis_distance)
            writer.write(combined)

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  frame {frame_idx}/{n_frames}  ({frame_idx / elapsed:.1f} fps)")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"  Render video: {render_path}")

    n_processed = len(all_2d)
    print(f"  {n_processed} frames: {detected_count / n_processed:.1%} had a person detected, "
          f"{anchored_count / n_processed:.1%} depth-anchored ({time.time() - t0:.1f}s).")

    bone_edges = np.array([], dtype=np.int64).reshape(0, 2)
    bone_targets = np.array([], dtype=np.float64)
    body_scale_m = float("nan")
    if bone_filter is not None:
        targets = bone_filter.target_lengths()
        bone_edges = np.array(list(targets.keys()), dtype=np.int64)
        bone_targets = np.array(list(targets.values()), dtype=np.float64)
        body_scale_m = sum(targets.get(edge, 0.0) for edge in _HEIGHT_CHAIN_EDGES)

    np.savez_compressed(
        out_path,
        source_r3d_dir=str(r3d_dir),
        keypoints_2d=np.array(all_2d),
        keypoints_3d_relative=np.array(all_3d_rel),
        keypoints_3d_absolute=np.array(all_3d_abs),
        anchor_scale=np.array(all_scale),
        anchor_translation=np.array(all_translation),
        timestamps=np.array(all_timestamps),
        fps=fps,
        K=K,
        image_size=np.array(image_size),
        bone_length_edges=bone_edges,
        bone_length_targets=bone_targets,
        body_scale_m=body_scale_m,
    )
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
