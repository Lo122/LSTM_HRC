"""Fast, standalone debug/inspection script for skeleton_pipeline's
render/feature/plot code -- NO YOLO/MotionBERT model loading, so it runs
in seconds instead of minutes. Use this to check that visualization/plots/
result format look right, instead of re-running the full (slow)
generate_lstm_training_data.py pipeline every time you tweak render/
feature/plot code.

Two modes:
  --npz <path>   Load an existing generate_lstm_training_data.py output
                 and re-render/re-plot/inspect ITS real data -- lets you
                 iterate on render/feature/plot code against real numbers
                 without re-running the slow YOLO+MotionBERT pass, and
                 doubles as a quick "what does our saved format actually
                 look like" inspector.
  (no --npz)     Generate synthetic swaying-skeleton motion instead --
                 zero setup, works even before you've run the real
                 pipeline once.

Pass --source-video (only meaningful with --npz, and only if you still
have the original clip) to overlay the 2D skeleton on real frames instead
of a blank canvas -- optional, since keypoints_3d/keypoints_2d alone are
enough to check shapes/plots/3D render.

Usage:
    python debug_visualize.py                                  # synthetic data, fastest check
    python debug_visualize.py --npz results/video__cam-04_uid-01_take-01.npz             # inspect a real prior run
    python debug_visualize.py --npz results/video__cam-04_uid-01_take-01.npz --show      # + live preview window
    uv run python debug_visualize.py `
      --npz "C:\\Users\\Owner\\OneDrive - Universität Stuttgart\\2025_26_Thesis\\codes\\LSTM_HRC\\data_proc_3d\\results\\raw\\video__cam-04_uid-01_take-01.npz" `
        --source-video "G:\\.shortcut-targets-by-id\\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\\ITECH_Thesis\\Videos\\raw\\cam-04\\video__cam-04_uid-01_take-01.mp4" `
        --show
    """
    
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from skeleton_pipeline.features.h36m_features import compute_all_features
from skeleton_pipeline.plotting.feature_plots import plot_panels
from skeleton_pipeline.render.skeleton_video import FastSkeleton3DRenderer, render_combined_frame


def make_synthetic_motion(n_frames=150, fps=30.0):
    """A simple periodic arm-swinging H36M skeleton (root-relative,
    meters) -- not a real person, just enough rhythmic structure that
    velocity/acceleration/joint-angle plots look sensible instead of pure
    jagged noise, for a quick visual sanity check."""
    t = np.arange(n_frames) / fps
    base = np.array([
        [0, 0, 0],                                              # pelvis
        [-0.09, 0, 0], [-0.09, 0, -0.45], [-0.09, 0, -0.9],      # r_hip, r_knee, r_ankle
        [0.09, 0, 0], [0.09, 0, -0.45], [0.09, 0, -0.9],         # l_hip, l_knee, l_ankle
        [0, 0, 0.25], [0, 0, 0.5], [0, 0, 0.6], [0, 0, 0.75],    # spine, thorax, neck, head
        [0.18, 0, 0.45], [0.18, 0, 0.15], [0.18, 0, -0.1],       # l_shoulder, l_elbow, l_wrist
        [-0.18, 0, 0.45], [-0.18, 0, 0.15], [-0.18, 0, -0.1],    # r_shoulder, r_elbow, r_wrist
    ])
    positions = np.tile(base[None], (n_frames, 1, 1)).astype(np.float64)
    swing = 0.3 * np.sin(2 * np.pi * 0.5 * t)
    positions[:, 12, 1] += swing           # l_elbow swings forward/back
    positions[:, 13, 1] += swing * 1.6      # l_wrist swings further (longer lever arm)
    positions[:, 15, 1] -= swing            # r_elbow: opposite phase
    positions[:, 16, 1] -= swing * 1.6
    positions[:, 0] = 0.0  # pelvis stays exactly at root
    return positions


def print_npz_summary(data):
    print("Keys and shapes:")
    for key in data.files:
        arr = data[key]
        shape = getattr(arr, "shape", None)
        print(f"  {key:24s} shape={shape} dtype={arr.dtype}")
    kp3d = data["keypoints_3d"]
    n_valid = int(np.sum(~np.isnan(kp3d).any(axis=(1, 2))))
    print(f"\nkeypoints_3d: {kp3d.shape[0]} frames, {n_valid} with a valid (non-NaN) skeleton "
          f"({n_valid / kp3d.shape[0]:.1%}).")
    print(f"fps: {float(data['fps'])}, body_scale_m: {float(data['body_scale_m']):.3f}, "
          f"gravity_aligned: {bool(data['gravity_aligned'])}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=str, default=None,
                         help="Load a real generate_lstm_training_data.py .npz output instead of "
                              "synthetic data.")
    parser.add_argument("--source-video", type=str, default=None,
                         help="Only used with --npz. Overlay the 2D skeleton on real frames from "
                              "this video (must be the SAME video the .npz was generated from, so "
                              "frame indices line up) instead of a blank canvas.")
    parser.add_argument("--max-frames", type=int, default=1000,
                         help="Cap how many frames to render/plot (speed). Synthetic data is "
                              "generated at exactly this length.")
    parser.add_argument("--frame-size", type=int, nargs=2, default=(960, 720),
                         help="Blank-canvas size for the 2D panel when --source-video isn't given.")
    parser.add_argument("--output-dir", type=str, default="debug_output")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(480, 480))
    parser.add_argument("--show", action="store_true",
                         help="Also pop up a live cv2.imshow preview (press Q/ESC to stop early) "
                              "instead of only writing files to --output-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.npz is not None:
        print(f"Loading {args.npz} ...")
        data = np.load(args.npz, allow_pickle=True)
        print_npz_summary(data)
        positions_full = data["keypoints_3d"]
        keypoints_2d_full = data["keypoints_2d"]
        fps = float(data["fps"])
        stem = Path(args.npz).stem
    else:
        print("No --npz given -- using synthetic swaying-skeleton data.")
        fps = 30.0
        positions_full = make_synthetic_motion(args.max_frames, fps)
        keypoints_2d_full = None
        stem = "synthetic"

    n_frames = min(args.max_frames, len(positions_full))
    positions = positions_full[:n_frames]
    keypoints_2d = keypoints_2d_full[:n_frames] if keypoints_2d_full is not None else None
    timestamps = np.arange(n_frames) / fps
    print(f"\nUsing {n_frames} frames.")

    print("\nComputing features...")
    feature_dict, panel_groups = compute_all_features(positions, fps)
    print(f"  {len(feature_dict)} feature columns across {len(panel_groups)} panels:")
    for panel, cols in panel_groups.items():
        sample_col = cols[0]
        sample_val = feature_dict[sample_col]
        finite = sample_val[np.isfinite(sample_val)]
        rng_str = f"[{finite.min():.3g}, {finite.max():.3g}]" if len(finite) else "all-NaN"
        print(f"    {panel:32s} ({len(cols):2d} cols) e.g. {sample_col}={rng_str}")

    print("\nWriting plots...")
    plot_paths = plot_panels(feature_dict, panel_groups, timestamps, output_dir, stem)
    print(f"  {len(plot_paths)} panel plots -> {output_dir}")

    print("\nRendering preview video...")
    renderer_3d = FastSkeleton3DRenderer(args.panel_size)
    panel_w, panel_h = args.panel_size
    render_path = output_dir / f"{stem}_debug_render.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(render_path), fourcc, fps, (panel_w * 2, panel_h))

    cap = cv2.VideoCapture(args.source_video) if args.source_video else None
    blank_w, blank_h = args.frame_size
    blank_frame = np.full((blank_h, blank_w, 3), 245, dtype=np.uint8)

    for t in range(n_frames):
        if cap is not None:
            ok, frame_bgr = cap.read()
            if not ok:
                print(f"  --source-video ran out of frames at t={t}, using blank canvas from here.")
                cap.release()
                cap = None
                frame_bgr = blank_frame
        else:
            frame_bgr = blank_frame
        kp2d = keypoints_2d[t] if keypoints_2d is not None else None
        kpconf = np.ones(17, dtype=np.float32) if kp2d is not None else None
        combined = render_combined_frame(
            frame_bgr, kp2d, kpconf, positions[t], renderer_3d, args.panel_size)
        writer.write(combined)
        if args.show:
            cv2.imshow("debug_visualize", combined)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key in (ord("q"), 27):
                print("  Stopped by user (Q/ESC).")
                break
    if cap is not None:
        cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"  Preview video -> {render_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
