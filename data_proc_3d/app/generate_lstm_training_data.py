"""Batch-generate LSTM training data: YOLO 2D pose detection -> MotionBERT
DSTformer 2D->3D lifting -> per-subject bone-length stabilization, over
every video in a folder (e.g. G:\\...\\Videos\\raw\\cam-04).

Runs in its OWN minimal venv (data_proc_3d/app/pyproject.toml) -- no
mmcv/mmpose/mmdet/tensorflow/record3d, unlike the big multi-purpose
world_pose project (data_proc_3d/src/world_pose) this was split out from.
Utility code lives in data_proc_3d/src/skeleton_pipeline/ (this script
inserts that directory onto sys.path, same convention world_pose's own
demo scripts use -- no `pip install` of skeleton_pipeline itself).

--clip-len defaults to 81, matching this project's LIVE deployment window
(see conversation notes / README: training data should be generated with
the SAME MotionBERT window size actually used live, not the higher-quality
243, to avoid a train/inference distribution mismatch -- an LSTM trained on
243-window's smoother/less-jittery output would see a systematically
different input distribution at deployment, where only 81-window fits the
real-time budget).

Standardization / height handling (see this script's docstring further
down, `--gravity-align`, and BoneLengthConstraintFilter):
  - WITHIN one video/subject: BoneLengthConstraintFilter is reset per video
    and used to stabilize bone lengths over that clip -- this removes the
    frame-to-frame shrink/stretch noise from monocular depth ambiguity
    (bad: the same physical person doesn't actually change height
    mid-task), while letting the STABLE value it converges to differ
    freely between videos.
  - ACROSS videos/subjects: deliberately NO cross-video rescaling to a
    canonical/unit height. Each video's own calibrated bone lengths are
    saved as-is in its .npz (bone_length_edges/bone_length_targets, plus a
    convenience body_scale_m estimate) -- a clean, temporally-noise-free
    per-subject scale feature, decoupled from the frame-by-frame posture
    data, that an LSTM (or its own preprocessing) can use explicitly if
    between-subject height differences are meant to help predict task
    state/progress, without that signal being contaminated by the
    per-frame lifter noise BoneLengthConstraintFilter is fixing.
  Do NOT additionally apply a generic "normalize every frame to a unit
  skeleton" step on top of this (a common technique in action-recognition
  literature) -- that would erase the between-subject height signal this
  project wants to keep, on top of not being needed since the temporal
  noise is already handled by the stabilizer above.

Usage:
    uv run python data_proc_3d/app/generate_lstm_training_data.py `
        --video-dir "G:\\My Drive\\University of Stuttgart\\ITECH_Thesis\\Videos\\raw\\cam-04" `
        --output-dir "C:\\Users\\Owner\\OneDrive - Universität Stuttgart\\2025_26_Thesis\\codes\\LSTM_HRC\\data_proc_3d\\results" `
        --device cuda:0 `
        --motionbert-config "C:\\Users\\Owner\\OneDrive - Universität Stuttgart\\2025_26_Thesis\\codes\\MotionBERT\\configs\\pose3d\\MB_ft_h36m.yaml" `
        --motionbert-checkpoint "C:\\Users\\Owner\\OneDrive - Universität Stuttgart\\2025_26_Thesis\\codes\\MotionBERT\\checkpoint\\pose3d\\FT_MB_release_MB_ft_h36m\\best_epoch.bin" 
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skeleton_pipeline.bone_length_filter import BoneLengthConstraintFilter
from skeleton_pipeline.calibration_io import load_extrinsics, load_intrinsics
from skeleton_pipeline.keypoint_filter import KeypointOutlierHoldFilter
from skeleton_pipeline.motionbert_lifter import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, MotionBERTStreamingLifter,
)
from skeleton_pipeline.features.h36m_features import compute_all_features
from skeleton_pipeline.plotting.feature_plots import plot_panels
from skeleton_pipeline.render.skeleton_video import FastSkeleton3DRenderer, render_combined_frame

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# H36M bone tree, leg+spine chain used for the convenience body_scale_m
# estimate (ankle -> hip -> spine -> thorax -> head, one side): NOT a
# calibrated real height, just a stable, comparable-across-videos scale
# proxy derived from BoneLengthConstraintFilter's own converged lengths.
_HEIGHT_CHAIN_EDGES = [(0, 1), (1, 2), (2, 3), (0, 7), (7, 8), (8, 9), (9, 10)]


def run_yolo_2d(yolo, frame, imgsz=None):
    kwargs = {"imgsz": imgsz} if imgsz is not None else {}
    results = yolo(frame, verbose=False, **kwargs)
    if not results or results[0].keypoints is None or results[0].keypoints.xy.numel() == 0:
        return None, None
    kpts_xy = results[0].keypoints.xy[0].cpu().numpy()
    conf = (results[0].keypoints.conf[0].cpu().numpy()
            if results[0].keypoints.conf is not None
            else np.ones(kpts_xy.shape[0], dtype=np.float32))
    return kpts_xy, conf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", type=str, required=True,
                         help="Folder containing video files (recursed one level via glob "
                              f"{VIDEO_EXTENSIONS}).")
    parser.add_argument("--output-dir", type=str, default="results",
                         help="One <video-stem>.npz written here per input video.")
    parser.add_argument("--yolo-model", type=str, default="data_proc_3d/dataset/model/yolo26m-pose.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                         help="'cpu', 'cuda:0', etc. -- passed to both YOLO and MotionBERT.")
    parser.add_argument("--motionbert-config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--motionbert-checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="Default: FT_MB_release_MB_ft_h36m/best_epoch.bin (full-size, "
                              "rootrel:True) -- see skeleton_pipeline/motionbert_lifter.py's "
                              "docstring for why this was chosen over the 'lite' checkpoint.")
    parser.add_argument("--clip-len", type=int, default=81,
                         help="MotionBERT rolling-buffer window length. Default 81 (NOT 243) to "
                              "match this project's live/deployment window -- see module "
                              "docstring on why training data should match deployment, not use "
                              "the higher-quality-but-mismatched 243 window.")
    parser.add_argument("--no-keypoint-filter", action="store_true",
                         help="Disable KeypointOutlierHoldFilter on the 2D detection.")
    parser.add_argument("--no-bone-length-filter", action="store_true",
                         help="Disable BoneLengthConstraintFilter -- NOT recommended for training "
                              "data (see module docstring: this is what removes the within-subject "
                              "shrink/stretch noise before it reaches the LSTM).")
    parser.add_argument("--calib-dir", type=str, default=None,
                         help="Directory with intrinsics.json/extrinsics.json for this camera (see "
                              "--intrinsics-file/--extrinsics-file for the exact filenames). If "
                              "extrinsics are found, --gravity-align applies; if not, that step is "
                              "skipped and a warning is printed -- calibration is optional, not "
                              "required, to run this script.")
    parser.add_argument("--intrinsics-file", type=str, default="intrinsics.json")
    parser.add_argument("--extrinsics-file", type=str, default="extrinsics.json")
    parser.add_argument("--gravity-align", action="store_true",
                         help="Rotate the root-relative skeleton by --calib-dir's "
                              "T_world_from_camera rotation, so 'up' in the saved data is true "
                              "gravity-up instead of this camera's own (possibly tilted) axis -- "
                              "makes skeletons comparable across videos recorded with the camera "
                              "mounted at different angles. Only applies if --calib-dir has "
                              "extrinsics; a translation-free rotation only (this pipeline has no "
                              "absolute/world position to translate, see module docstring).")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Hard cap on frames processed per video (debugging).")
    parser.add_argument("--no-render-video", action="store_true",
                         help="Don't write a <stem>_render.mp4 (2D overlay | 3D 4-view skeleton) "
                              "per video -- see skeleton_pipeline/render/skeleton_video.py. On by "
                              "default; disable for a faster run once you trust the pipeline.")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(640, 360),
                         help="Width height of each of the two render panels.")
    parser.add_argument("--no-plots", action="store_true",
                         help="Don't write per-panel feature plots (see skeleton_pipeline/"
                              "features/h36m_features.py and plotting/feature_plots.py) -- one PNG "
                              "per feature group (Joint Speed, Joint Angles, etc.), each with one "
                              "line per joint/column. On by default.")
    parser.add_argument("--plots-dir", type=str, default=None,
                         help="Default: <output-dir>/plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"

    video_dir = Path(args.video_dir)
    video_paths = sorted(
        p for ext in VIDEO_EXTENSIONS
        for p in video_dir.glob(f"*{ext}")
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos found in {args.video_dir} (extensions: {VIDEO_EXTENSIONS}).")
    print(f"Found {len(video_paths)} video(s) in {args.video_dir}.")

    R_world_from_camera = None
    if args.calib_dir is not None:
        calib_dir = Path(args.calib_dir)
        extrinsics_path = calib_dir / args.extrinsics_file
        if extrinsics_path.exists():
            T_world_from_camera, _ground_z, _robot_base = load_extrinsics(extrinsics_path)
            R_world_from_camera = T_world_from_camera[:3, :3]
            print(f"Loaded extrinsics from {extrinsics_path}.")
        elif args.gravity_align:
            print(f"NOTE: --gravity-align requested but {extrinsics_path} not found -- skipping "
                  f"gravity alignment for all videos in this run.")
        intrinsics_path = calib_dir / args.intrinsics_file
        if intrinsics_path.exists():
            K, dist, image_size = load_intrinsics(intrinsics_path)
            print(f"Loaded intrinsics from {intrinsics_path}: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}")
        else:
            K = None
    else:
        K = None

    print(f"Loading YOLO ({args.yolo_model}) on {args.device}...")
    yolo = YOLO(args.yolo_model)

    print(f"Loading MotionBERT (config={args.motionbert_config}, "
          f"checkpoint={args.motionbert_checkpoint}, clip_len={args.clip_len}) on {args.device}...")
    lifter = MotionBERTStreamingLifter(
        config_path=args.motionbert_config, checkpoint_path=args.motionbert_checkpoint,
        clip_len=args.clip_len, device=args.device)

    renderer_3d = FastSkeleton3DRenderer(args.panel_size) if not args.no_render_video else None

    for video_path in video_paths:
        stem = video_path.stem
        out_path = output_dir / f"{stem}.npz"
        print(f"\n=== {video_path.name} -> {out_path} ===")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  WARNING: could not open {video_path}, skipping.")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        lifter.reset()
        kp_filter = None if args.no_keypoint_filter else KeypointOutlierHoldFilter()
        bone_filter = None if args.no_bone_length_filter else BoneLengthConstraintFilter()

        writer = None
        if renderer_3d is not None:
            render_path = output_dir / f"{stem}_render.mp4"
            panel_w, panel_h = args.panel_size
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(render_path), fourcc, fps, (panel_w * 2, panel_h))

        all_2d, all_3d, all_timestamps = [], [], []
        frame_idx = 0
        detected_count = 0
        t0 = time.time()
        while args.max_frames is None or frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            keypoints_2d, keypoints_conf = run_yolo_2d(yolo, frame, imgsz=args.yolo_imgsz)
            if kp_filter is not None:
                keypoints_2d, keypoints_conf, _status = kp_filter.filter(keypoints_2d, keypoints_conf)

            skeleton_3d = None
            if keypoints_2d is not None and np.any(keypoints_2d):
                detected_count += 1
                skeleton_3d = lifter.lift(keypoints_2d, image_size=(w, h), keypoints_conf=keypoints_conf)
                if skeleton_3d is not None and bone_filter is not None:
                    skeleton_3d = bone_filter.filter(skeleton_3d)
                if skeleton_3d is not None and R_world_from_camera is not None and args.gravity_align:
                    skeleton_3d = skeleton_3d @ R_world_from_camera.T

            all_2d.append(keypoints_2d if keypoints_2d is not None else np.zeros((17, 2)))
            all_3d.append(skeleton_3d if skeleton_3d is not None else np.full((17, 3), np.nan))
            all_timestamps.append(frame_idx / fps)

            if writer is not None:
                combined = render_combined_frame(
                    frame, keypoints_2d, keypoints_conf, skeleton_3d, renderer_3d, args.panel_size)
                writer.write(combined)

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f"  frame {frame_idx}  ({frame_idx / elapsed:.1f} fps)")
        cap.release()
        if writer is not None:
            writer.release()
            print(f"  Render video: {render_path}")

        n_frames = len(all_2d)
        detection_rate = detected_count / n_frames if n_frames else 0.0
        print(f"  {n_frames} frames, {detection_rate:.1%} had a person detected "
              f"({time.time() - t0:.1f}s).")

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
            source_video=str(video_path),
            keypoints_2d=np.array(all_2d),
            keypoints_3d=np.array(all_3d),
            timestamps=np.array(all_timestamps),
            fps=fps,
            bone_length_edges=bone_edges,
            bone_length_targets=bone_targets,
            body_scale_m=body_scale_m,
            gravity_aligned=bool(R_world_from_camera is not None and args.gravity_align),
        )
        print(f"  Saved: {out_path} (body_scale_m proxy={body_scale_m:.3f})")

        if not args.no_plots:
            positions = np.array(all_3d)
            timestamps = np.array(all_timestamps)
            feature_dict, panel_groups = compute_all_features(positions, fps)
            plot_paths = plot_panels(feature_dict, panel_groups, timestamps, plots_dir, stem)
            print(f"  Feature plots ({len(plot_paths)} panels): {plots_dir}")


if __name__ == "__main__":
    main()
