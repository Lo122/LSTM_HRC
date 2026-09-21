"""Offline processing of an exported Record3D (.r3d) capture file into an
RGB video + per-frame metric depth + camera intrinsics/poses.

This is the FILE-based counterpart to camera_utils/iphone_connection.py's
live USB-streaming IPhoneCamera -- that one needs the Record3D app open and
the phone tethered right now; this only needs a .r3d file previously
exported from the app (in-app "Share"/AirDrop/Files/iTunes), so it works on
captures recorded anywhere, at any time, without a phone connected at all.

.r3d file format (Record3D ships no format spec -- these fields were
confirmed by inspecting an actual exported capture; run with
--dump-structure on a new/unfamiliar capture to sanity-check these
assumptions still hold before trusting the rest of this script, since the
app has changed its export layout across versions, e.g. some versions pack
color as a single rgbd.mp4 instead of per-frame jpgs):

  .r3d is a ZIP archive containing:
    metadata            JSON, no extension, at the archive root. Keys used
                         here: "w"/"h" (color frame size), "dw"/"dh" (depth
                         frame size -- may differ from w/h, e.g. on LiDAR
                         devices whose depth sensor is much lower-res than
                         the color camera), "fps", "cameraType" (0 =
                         TrueDepth, 1 = LiDAR -- matches IPhoneCamera.
                         _DEVICE_TYPE_LIDAR's encoding of the live SDK's
                         device-type enum), "K" (COLUMN-major
                         [fx,0,0, 0,fy,0, cx,cy,1] -- note this is the
                         TRANSPOSE of the usual row-major pinhole matrix),
                         "poses" (per-frame [qx,qy,qz,qw,tx,ty,tz], same
                         fields/order as record3d.CameraPose), "frameTimestamps"
                         (seconds, one per frame), "perFrameIntrinsicCoeffs"
                         (per-frame [fx,fy,cx,cy] -- like Record3D's live
                         per-frame reported K; ARKit occasionally refines
                         focus mid-capture, see iphone_intrinsic_
                         calibration.py's --use-reported-intrinsics).
    rgbd/<i>.jpg         color frame i, standard JPEG (i = 0..num_frames-1).
    rgbd/<i>.depth       depth frame i: raw (dh, dw) float32 METERS,
                         row-major, LZFSE-compressed (Apple's codec, NOT
                         zlib/gzip -- needs the 'pyliblzfse' package, see
                         _decompress_lzfse). NaN marks a pixel with no valid
                         depth reading.
    rgbd/<i>.conf        optional (LiDAR captures only): (dh, dw) uint8
                         confidence (0/1/2), same LZFSE compression.

Output, written to --output-dir/<r3d-stem>/:
  rgb.mp4            color video at the capture's native fps -- drop-in
                     input for generate_lstm_training_data.py --video-dir.
  depth.npz          depth: (n, dh, dw) float32 meters (NaN = invalid),
                     confidence: (n, dh, dw) uint8, or an empty array if the
                     capture had no .conf files, timestamps: (n,) seconds,
                     native_size: [dw, dh] (the resolution the array is in
                     -- see --align-depth-to-rgb).
  depth_colormap.mp4 depth visualized as a false-color video (see
                     depth_to_colormap) at depth's own native resolution --
                     NOT numeric data, just for eyeballing the capture (hole
                     patterns, range, noise). Invalid/NaN pixels are black.
                     Color scale is fixed for the whole capture (min/max
                     depth percentile by default, see --colormap-min/-max)
                     so brightness is comparable frame-to-frame. Skippable
                     with --no-depth-colormap.
  rgbd_preview.mp4   color | depth_colormap side-by-side (depth resized to
                     match color's resolution for display only -- depth.npz
                     itself is never touched by this). Skippable with
                     --no-side-by-side-preview.
  intrinsics.json    camera_utils.calibration_io.save_intrinsics format --
                     this capture's own K (frame 0's perFrameIntrinsicCoeffs;
                     rarely moves once ARKit's focus locks) and image_size.
  poses.npz          qx, qy, qz, qw, tx, ty, tz: (n,) arrays, plus
                     timestamps -- same fields as camera_utils.
                     iphone_connection's live CameraPose, so
                     camera_pose_rotation_matrix/roll_from_camera_pose/
                     camera_gravity_direction_cv all work unchanged on these
                     too (see build_camera_pose below). This is ARKit's own
                     session-start origin, NOT this project's calibrated
                     world frame (see iphone_connection.py's module
                     docstring) -- don't mix it with T_world_from_camera
                     from iphone_extrinsic_calibration.py without going
                     through that distinction.

Needs the 'pyliblzfse' package (pip install pyliblzfse) for depth/confidence
decompression -- not part of this app's pyproject.toml dependency set used
by the rest of the pipeline, since only THIS script touches .r3d files
directly. It ships only as a source distribution (compiles a small bundled
C extension) but has built cleanly with a plain `pip install pyliblzfse` on
this project's Windows venvs.

Usage:
    uv run python iphone_lidar_test.py path/to/capture.r3d --output-dir results/r3d
    python iphone_lidar_test.py path/to/capture.r3d --output-dir results/r3d --capture-rotate90 90
    python iphone_lidar_test.py path/to/capture.r3d --dump-structure   # inspect a new capture first
    
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camera_utils.calibration_io import save_intrinsics
# Reusing iphone_connection.py's own capture_rotate90 math (private helpers)
# rather than re-deriving it, so a .r3d file processed here and a live
# IPhoneCamera(capture_rotate90=...) stream stay bit-for-bit consistent in
# how K/pixels get rotated.
from camera_utils.iphone_connection import _ROTATE_FLAGS, _rotate_intrinsics_90


class CameraPose(NamedTuple):
    """Matches record3d.CameraPose's fields (qx, qy, qz, qw, tx, ty, tz) --
    lets camera_utils.iphone_connection's camera_pose_rotation_matrix/
    roll_from_camera_pose/camera_gravity_direction_cv work unchanged on
    poses loaded from a .r3d file's poses.npz, not just a live stream."""
    qx: float
    qy: float
    qz: float
    qw: float
    tx: float
    ty: float
    tz: float


def _decompress_lzfse(raw: bytes) -> bytes:
    try:
        import liblzfse
    except ImportError as e:
        raise ImportError(
            "Reading depth/confidence from a .r3d file needs the 'pyliblzfse' "
            "package (pip install pyliblzfse). Record3D compresses these with "
            "Apple's LZFSE codec, not zlib/gzip -- see this module's docstring."
        ) from e
    return liblzfse.decompress(raw)


def _build_K(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def depth_to_colormap(depth, vmin, vmax, colormap=cv2.COLORMAP_TURBO):
    """depth: (h, w) float32 meters (NaN = invalid). Returns an (h, w, 3)
    BGR uint8 false-color image -- near=one end of the colormap, far=the
    other, invalid/NaN pixels forced to black (0,0,0) so holes read as
    holes rather than an arbitrary color. vmin/vmax: the depth range (m)
    mapped to the colormap's low/high end -- fix these across a whole
    capture (see compute_depth_range) so brightness is comparable
    frame-to-frame, not auto-stretched per frame."""
    valid = np.isfinite(depth)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    span = max(vmax - vmin, 1e-6)
    clipped = np.clip(depth, vmin, vmax)
    scaled[valid] = ((clipped[valid] - vmin) / span * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, colormap)
    colored[~valid] = (0, 0, 0)
    return colored


def draw_colorbar_legend(colored, vmin, vmax, colormap, bar_width=22, bar_height_frac=0.5,
                          margin=12, n_ticks=5):
    """Overlay a vertical colorbar legend (gradient strip + tick labels in
    meters) onto the top-right corner of an already depth_to_colormap'd BGR
    image -- so a viewer can read an actual distance off depth_colormap.mp4/
    rgbd_preview.mp4, not just relative near/far. Drawn on a copy; doesn't
    touch the input array. Top of the bar = vmax (far), bottom = vmin
    (near), matching depth_to_colormap's own low->high = near->far mapping.
    """
    out = colored.copy()
    h, w = out.shape[:2]
    bar_height = max(int(h * bar_height_frac), n_ticks * 14)
    bar_top = margin
    bar_left = w - margin - bar_width

    # Translucent backing panel first, so the bar/text stay legible over
    # whatever busy depth colors happen to be underneath them.
    label_width = 56
    panel_tl = (bar_left - label_width - 6, bar_top - 8)
    panel_br = (w - 2, bar_top + bar_height + 8)
    overlay = out.copy()
    cv2.rectangle(overlay, panel_tl, panel_br, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    bar = cv2.applyColorMap(gradient, colormap)
    out[bar_top:bar_top + bar_height, bar_left:bar_left + bar_width] = bar
    cv2.rectangle(out, (bar_left, bar_top), (bar_left + bar_width - 1, bar_top + bar_height - 1),
                  (255, 255, 255), 1, cv2.LINE_AA)

    text_x = bar_left - label_width
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)  # 0 = top = vmax, 1 = bottom = vmin
        y = int(bar_top + frac * (bar_height - 1))
        value = vmax - frac * (vmax - vmin)
        cv2.line(out, (bar_left - 4, y), (bar_left, y), (255, 255, 255), 1, cv2.LINE_AA)
        text_y = int(np.clip(y + 4, bar_top + 10, bar_top + bar_height - 2))
        cv2.putText(out, f"{value:.2f}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "m", (text_x, bar_top + bar_height + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compute_depth_range(depth_frames, low_pct=1.0, high_pct=99.0):
    """Robust (vmin, vmax) meters spanning most of a capture's valid depth
    values -- percentile, not plain min/max, so a handful of noisy
    near-zero or far-outlier readings don't wash out the whole colormap."""
    stacked = np.concatenate([d[np.isfinite(d)].ravel() for d in depth_frames])
    if stacked.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(stacked, [low_pct, high_pct])
    return float(vmin), float(vmax)


def dump_structure(zf: zipfile.ZipFile, metadata: dict) -> None:
    """Print a capture's zip listing + metadata keys, to sanity-check this
    module's format assumptions before trusting the rest of it -- see module
    docstring."""
    names = zf.namelist()
    print(f"{len(names)} entries in archive.")
    root_files = sorted(n for n in names if "/" not in n)
    print(f"Root-level files: {root_files}")
    rgbd_names = [n for n in names if n.startswith("rgbd/")]
    exts = sorted({Path(n).suffix for n in rgbd_names})
    print(f"rgbd/ extensions present: {exts} ({len(rgbd_names)} files)")
    print(f"metadata keys: {sorted(metadata.keys())}")
    for k, v in metadata.items():
        if isinstance(v, list):
            print(f"  {k}: list, len={len(v)}, first={v[0] if v else None}")
        else:
            print(f"  {k}: {v}")


def build_camera_pose(poses_npz, index: int) -> CameraPose:
    """Reconstruct a single frame's CameraPose from a poses.npz written by
    process_r3d, for use with camera_utils.iphone_connection's pose helpers."""
    return CameraPose(
        qx=float(poses_npz["qx"][index]), qy=float(poses_npz["qy"][index]),
        qz=float(poses_npz["qz"][index]), qw=float(poses_npz["qw"][index]),
        tx=float(poses_npz["tx"][index]), ty=float(poses_npz["ty"][index]),
        tz=float(poses_npz["tz"][index]),
    )


def process_r3d(r3d_path, output_dir, capture_rotate90=0, align_depth_to_rgb=False,
                 max_frames=None, save_depth_colormap=True, save_side_by_side_preview=True,
                 colormap_min=None, colormap_max=None, colormap=cv2.COLORMAP_TURBO,
                 show_legend=True):
    r3d_path = Path(r3d_path)
    zf = zipfile.ZipFile(r3d_path)
    metadata = json.loads(zf.read("metadata"))

    w, h = int(metadata["w"]), int(metadata["h"])
    dw, dh = int(metadata["dw"]), int(metadata["dh"])
    fps = float(metadata.get("fps", 30.0))
    poses = metadata["poses"]
    timestamps = metadata.get("frameTimestamps", [i / fps for i in range(len(poses))])
    per_frame_K = metadata.get("perFrameIntrinsicCoeffs")
    if per_frame_K is None:
        # Fall back to metadata["K"] (static, column-major -- see module docstring).
        K_static = np.array(metadata["K"], dtype=np.float64).reshape(3, 3).T
        per_frame_K = [[K_static[0, 0], K_static[1, 1], K_static[0, 2], K_static[1, 2]]] * len(poses)

    n_frames = len(poses)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    print(f"{r3d_path.name}: {n_frames} frames, {w}x{h} color, {dw}x{dh} depth, {fps} fps, "
          f"cameraType={metadata.get('cameraType')} (0=TrueDepth, 1=LiDAR).")

    has_conf = "rgbd/0.conf" in zf.namelist()

    out_dir = Path(output_dir) / r3d_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Post-rotation color frame size for the video writer (mirrors
    # iphone_connection.py's capture_rotate90 -- see _rotate_intrinsics_90).
    _, out_size = _rotate_intrinsics_90(np.eye(3), (w, h), capture_rotate90)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = out_dir / "rgb.mp4"
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, out_size)

    depth_frames = []
    conf_frames = [] if has_conf else None
    Ks = []
    qxs, qys, qzs, qws, txs, tys, tzs = [], [], [], [], [], [], []

    for i in range(n_frames):
        # cv2.imdecode gives BGR directly (unlike iphone_connection.py's
        # live path, which gets literal RGB arrays from Record3DStream and
        # has to cv2.cvtColor them) -- write straight to the BGR VideoWriter.
        jpg_bytes = zf.read(f"rgbd/{i}.jpg")
        bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not decode rgbd/{i}.jpg in {r3d_path}.")

        depth_raw = _decompress_lzfse(zf.read(f"rgbd/{i}.depth"))
        depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(dh, dw).copy()

        conf = None
        if has_conf:
            conf_raw = _decompress_lzfse(zf.read(f"rgbd/{i}.conf"))
            conf = np.frombuffer(conf_raw, dtype=np.uint8).reshape(dh, dw).copy()

        fx, fy, cx, cy = per_frame_K[i]
        K = _build_K(fx, fy, cx, cy)

        if capture_rotate90:
            flag = _ROTATE_FLAGS[capture_rotate90]
            bgr = cv2.rotate(bgr, flag)
            depth = cv2.rotate(depth, flag)
            if conf is not None:
                conf = cv2.rotate(conf, flag)
            K, _ = _rotate_intrinsics_90(K, (w, h), capture_rotate90)

        if align_depth_to_rgb and depth.shape[:2] != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            if conf is not None:
                conf = cv2.resize(conf, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        writer.write(bgr)
        depth_frames.append(depth)
        if conf is not None:
            conf_frames.append(conf)
        Ks.append(K)

        qx, qy, qz, qw, tx, ty, tz = poses[i]
        qxs.append(qx); qys.append(qy); qzs.append(qz); qws.append(qw)
        txs.append(tx); tys.append(ty); tzs.append(tz)

        if (i + 1) % 50 == 0 or i + 1 == n_frames:
            print(f"  frame {i + 1}/{n_frames}")

    writer.release()
    print(f"Saved {video_path}")

    depth_path = out_dir / "depth.npz"
    np.savez_compressed(
        depth_path,
        depth=np.stack(depth_frames),
        confidence=np.stack(conf_frames) if conf_frames else np.array([]),
        timestamps=np.array(timestamps[:n_frames]),
        native_size=np.array([dw, dh]),
    )
    print(f"Saved {depth_path}")

    if save_depth_colormap or save_side_by_side_preview:
        vmin, vmax = colormap_min, colormap_max
        if vmin is None or vmax is None:
            auto_vmin, auto_vmax = compute_depth_range(depth_frames)
            vmin = auto_vmin if vmin is None else vmin
            vmax = auto_vmax if vmax is None else vmax
        print(f"Depth colormap range: {vmin:.2f}-{vmax:.2f} m "
              f"({'auto 1st-99th percentile' if colormap_min is None or colormap_max is None else 'fixed'}).")

        depth_h, depth_w = depth_frames[0].shape[:2]

        colormap_writer = None
        if save_depth_colormap:
            colormap_path = out_dir / "depth_colormap.mp4"
            colormap_writer = cv2.VideoWriter(
                str(colormap_path), fourcc, fps, (depth_w, depth_h))

        preview_writer = None
        if save_side_by_side_preview:
            preview_path = out_dir / "rgbd_preview.mp4"
            preview_writer = cv2.VideoWriter(
                str(preview_path), fourcc, fps, (out_size[0] * 2, out_size[1]))

        for i in range(n_frames):
            colored = depth_to_colormap(depth_frames[i], vmin, vmax, colormap)
            if show_legend:
                colored = draw_colorbar_legend(colored, vmin, vmax, colormap)
            if colormap_writer is not None:
                colormap_writer.write(colored)
            if preview_writer is not None:
                # Re-decode+rotate this frame's color image rather than
                # holding every bgr frame in memory alongside depth_frames
                # (which already doubles this capture's raw size -- see
                # depth.npz above) -- jpg decode is cheap relative to that.
                jpg_bytes = zf.read(f"rgbd/{i}.jpg")
                bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if capture_rotate90:
                    bgr = cv2.rotate(bgr, _ROTATE_FLAGS[capture_rotate90])
                colored_resized = cv2.resize(colored, out_size) if colored.shape[:2][::-1] != out_size else colored
                preview_writer.write(np.hstack([bgr, colored_resized]))

        if colormap_writer is not None:
            colormap_writer.release()
            print(f"Saved {colormap_path}")
        if preview_writer is not None:
            preview_writer.release()
            print(f"Saved {preview_path}")

    poses_path = out_dir / "poses.npz"
    np.savez_compressed(
        poses_path,
        qx=np.array(qxs), qy=np.array(qys), qz=np.array(qzs), qw=np.array(qws),
        tx=np.array(txs), ty=np.array(tys), tz=np.array(tzs),
        timestamps=np.array(timestamps[:n_frames]),
    )
    print(f"Saved {poses_path}")

    intrinsics_path = out_dir / "intrinsics.json"
    dist = np.zeros(5, dtype=np.float64)  # Record3D reports an already-undistorted pinhole model
    save_intrinsics(intrinsics_path, Ks[0], dist, out_size)
    print(f"Saved {intrinsics_path}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("r3d_path", type=str, help="Path to the exported .r3d capture file.")
    parser.add_argument("--output-dir", type=str, default="results/r3d",
                         help="A <r3d-stem>/ subfolder is created here.")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                         help="Rotate rgb+depth+K together at the source, same convention as "
                              "camera_utils.iphone_connection.IPhoneCamera's capture_rotate90 -- "
                              "use if the exported video is sideways (phone held landscape etc.).")
    parser.add_argument("--align-depth-to-rgb", action="store_true",
                         help="Nearest-neighbor resize depth (and confidence) to match the color "
                              "frame's resolution when they differ (common on LiDAR devices, whose "
                              "depth sensor is much lower-res than the color camera). Off by "
                              "default -- depth.npz's native_size records the native depth "
                              "resolution either way.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: cap frames processed.")
    parser.add_argument("--no-depth-colormap", dest="save_depth_colormap", action="store_false",
                         help="Don't write depth_colormap.mp4 (false-color depth video, see module "
                              "docstring) -- on by default.")
    parser.add_argument("--no-side-by-side-preview", dest="save_side_by_side_preview",
                         action="store_false",
                         help="Don't write rgbd_preview.mp4 (color | depth colormap side-by-side) "
                              "-- on by default.")
    parser.set_defaults(save_depth_colormap=True, save_side_by_side_preview=True)
    parser.add_argument("--colormap-min", type=float, default=None,
                         help="Depth (meters) mapped to the colormap's near end. Default: this "
                              "capture's own 1st-percentile valid depth (see compute_depth_range).")
    parser.add_argument("--colormap-max", type=float, default=None,
                         help="Depth (meters) mapped to the colormap's far end. Default: this "
                              "capture's own 99th-percentile valid depth.")
    parser.add_argument("--colormap", type=str, default="TURBO",
                         help="Any cv2.COLORMAP_* name (without the prefix), e.g. TURBO, JET, "
                              "VIRIDIS, INFERNO.")
    parser.add_argument("--no-legend", dest="show_legend", action="store_false",
                         help="Don't overlay a colorbar legend (distance range in meters) on "
                              "depth_colormap.mp4/rgbd_preview.mp4 -- on by default (see "
                              "draw_colorbar_legend).")
    parser.set_defaults(show_legend=True)
    parser.add_argument("--dump-structure", action="store_true",
                         help="Print the archive's file listing + metadata keys and exit -- use "
                              "this on a new/unfamiliar capture before trusting this script's "
                              "format assumptions (see module docstring).")
    args = parser.parse_args()

    if args.dump_structure:
        zf = zipfile.ZipFile(args.r3d_path)
        metadata = json.loads(zf.read("metadata"))
        dump_structure(zf, metadata)
        return

    colormap_attr = f"COLORMAP_{args.colormap.upper()}"
    if not hasattr(cv2, colormap_attr):
        raise ValueError(f"Unknown --colormap {args.colormap!r} (no cv2.{colormap_attr}).")
    colormap = getattr(cv2, colormap_attr)

    out_dir_root = Path(__file__).parents[1] / args.output_dir

    process_r3d(args.r3d_path, out_dir_root, capture_rotate90=args.capture_rotate90,
                align_depth_to_rgb=args.align_depth_to_rgb, max_frames=args.max_frames,
                save_depth_colormap=args.save_depth_colormap,
                save_side_by_side_preview=args.save_side_by_side_preview,
                colormap_min=args.colormap_min, colormap_max=args.colormap_max, colormap=colormap,
                show_legend=args.show_legend)


if __name__ == "__main__":
    main()
