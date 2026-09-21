"""Inspect a video's person tracks so you can hand-pick the target ids.

Companion to generate_lstm_training_data.py's --target-tracks override. The
automatic target selection (longest-lived track + fragment linking, see
skeleton_pipeline/person_tracking.py) resolves most clips, but some are
genuinely ambiguous -- on cam-07_uid-10_take-01 the subject stops working,
stands watching a second person perform the task, then leaves the frame, so
no geometric rule can say which tracklets are "the subject" without
re-identifying her across a 90s absence.

This is offline training-data generation, so a human can just look and say.
This script makes that cheap:

  1. runs the tracker once and CACHES the result (a YOLO pass over a 20-minute
     video costs ~15 minutes; you will want to iterate without paying it
     twice),
  2. prints a table of every substantial track -- time span, frame count,
     body scale, image position,
  3. writes a contact sheet with one tile per track: a representative frame
     with that track's skeleton drawn on it, so you can see WHO each id is,
  4. prints a ready-to-edit JSON line to paste into your --target-tracks file.

Usage:
    cd data_proc_3d/app
    uv run python review_tracks.py `
        --video "G://...//raw//cam-07//video__cam-07_uid-10_take-01.mp4" `
        --out-dir "..//results//track_review" `
        --device cuda:0

Then edit the printed JSON down to just the subject's ids and save it as e.g.
target_tracks.json, and pass that to generate_lstm_training_data.py:

    uv run python generate_lstm_training_data.py ... --target-tracks target_tracks.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skeleton_pipeline.person_selection import _tracking_center
from skeleton_pipeline.person_tracking import (
    _robust_scale, _usable_frames_sorted, collect_person_tracks,
    link_track_fragments, select_dominant_track_id,
)

DEFAULT_TRACKER = Path(__file__).resolve().parent / "trackers" / "botsort_static_cam.yaml"

# COCO-17 skeleton, same edge list the render module uses for the 2D overlay.
SKELETON_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
                  (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/track_review")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Where the pickled tracks live (default: <out-dir>/cache). "
                             "Delete the .pkl to force a re-run of the tracker.")
    parser.add_argument("--yolo-model", type=str,
                        default="data_proc_3d/dataset/model/yolo26m-pose.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tracker", type=str, default=str(DEFAULT_TRACKER))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=60,
                        help="Only table/tile tracks with at least this many usable "
                             "frames -- below that they are almost always detector "
                             "noise, and tiling them all makes the sheet unreadable.")
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def load_or_collect_tracks(args, video_path, cache_dir):
    cache_path = cache_dir / f"tracks_{video_path.stem}.pkl"
    if cache_path.is_file():
        with cache_path.open("rb") as handle:
            blob = pickle.load(handle)
        print(f"Loaded cached tracks from {cache_path} "
              f"({len(blob['tracks'])} tracks, {blob['n_frames']} frames).")
        return blob["tracks"], blob["n_frames"]

    from ultralytics import YOLO

    print(f"No cache at {cache_path}; running the tracker (this is the slow part)...")
    yolo = YOLO(args.yolo_model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")
    try:
        tracks, n_frames = collect_person_tracks(
            yolo, cap, imgsz=args.yolo_imgsz, max_frames=args.max_frames,
            tracker=args.tracker, device=args.device)
    finally:
        cap.release()

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump({"tracks": tracks, "n_frames": n_frames}, handle)
    print(f"Cached tracks to {cache_path}.")
    return tracks, n_frames


def describe_tracks(tracks, fps, min_frames):
    """(sorted rows, auto-selected ids) for the table and the contact sheet."""
    rows = []
    for track_id, frames in tracks.items():
        usable = _usable_frames_sorted(frames, 0.2, 4)
        if len(usable) < min_frames:
            continue
        middle = usable[len(usable) // 2]
        center = _tracking_center(*frames[middle], 0.2)
        rows.append({
            "id": track_id,
            "start": usable[0],
            "end": usable[-1],
            "count": len(usable),
            "scale": _robust_scale(frames, usable, 0.2),
            "center": center,
            "sample": middle,
        })
    rows.sort(key=lambda row: row["start"])

    target_id, _ = select_dominant_track_id(tracks)
    auto_ids = set()
    if target_id is not None:
        _merged, merged_ids = link_track_fragments(tracks, target_id)
        auto_ids = {target_id, *merged_ids}
    return rows, auto_ids


def draw_track(frame, keypoints_xy, keypoints_conf, color, label):
    for a, b in SKELETON_EDGES:
        if keypoints_conf[a] >= 0.3 and keypoints_conf[b] >= 0.3:
            cv2.line(frame, tuple(keypoints_xy[a].astype(int)),
                     tuple(keypoints_xy[b].astype(int)), color, 3)
    for point, conf in zip(keypoints_xy, keypoints_conf):
        if conf >= 0.3:
            cv2.circle(frame, tuple(point.astype(int)), 4, color, -1)
    visible = keypoints_xy[keypoints_conf >= 0.3]
    if len(visible):
        cv2.putText(frame, label, tuple(visible.min(axis=0).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)


def build_contact_sheet(video_path, tracks, rows, auto_ids, fps, args):
    """One tile per track: its representative frame, that track in green and
    everyone else visible in that frame in red -- so a tile answers both
    "who is this id" and "who else was around"."""
    wanted = {row["sample"]: row for row in rows}
    tiles = []
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    last_wanted = max(wanted) if wanted else -1
    while frame_idx <= last_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        row = wanted.get(frame_idx)
        if row is not None:
            for other_id, frames in tracks.items():
                if frame_idx not in frames:
                    continue
                xy, conf = frames[frame_idx]
                if int((conf >= 0.3).sum()) < 4:
                    continue
                is_this = other_id == row["id"]
                draw_track(frame, xy, conf,
                           (0, 255, 0) if is_this else (0, 0, 255), f"id={other_id}")
            banner = (f"id={row['id']}  {row['start']/fps:.0f}-{row['end']/fps:.0f}s  "
                      f"n={row['count']}  scale={row['scale']:.0f}"
                      f"{'  [auto]' if row['id'] in auto_ids else ''}")
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
            cv2.putText(frame, banner, (16, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)
            height = int(frame.shape[0] * args.tile_width / frame.shape[1])
            tiles.append(cv2.resize(frame, (args.tile_width, height)))
        frame_idx += 1
    cap.release()

    if not tiles:
        return None
    columns = max(1, args.columns)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(grid)


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30) if cap.isOpened() else 30
    cap.release()

    tracks, n_frames = load_or_collect_tracks(args, video_path, cache_dir)
    rows, auto_ids = describe_tracks(tracks, fps, args.min_frames)

    print(f"\n=== {video_path.stem} ===")
    print(f"{n_frames} frames, {n_frames/fps:.1f}s, {len(tracks)} raw tracks, "
          f"{len(rows)} with >={args.min_frames} usable frames\n")
    print(f"{'id':>6} {'start':>9} {'end':>9} {'frames':>8} {'scale':>7} "
          f"{'center':>15}  auto")
    for row in rows:
        center = row["center"]
        print(f"{row['id']:>6} {row['start']/fps:>8.1f}s {row['end']/fps:>8.1f}s "
              f"{row['count']:>8} {row['scale']:>7.0f} "
              f"({center[0]:>6.0f},{center[1]:>6.0f})"
              f"{'   <-- auto' if row['id'] in auto_ids else ''}")

    sheet = build_contact_sheet(video_path, tracks, rows, auto_ids, fps, args)
    if sheet is not None:
        sheet_path = out_dir / f"{video_path.stem}_tracks.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"\nContact sheet: {sheet_path}")
        print("  green = the tile's own track, red = everyone else in that frame.")

    auto_line = json.dumps({video_path.stem: sorted(auto_ids)})
    all_line = json.dumps({video_path.stem: [row["id"] for row in rows]})
    print("\nPaste into your --target-tracks file, then delete the ids that are "
          "not the subject:")
    print(f"  what automatic selection chose : {auto_line}")
    print(f"  every substantial track        : {all_line}")


if __name__ == "__main__":
    main()
