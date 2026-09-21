"""Convert ELAN CSV exports into the label__*.json files build_training_pairs.py reads.

Reads one CSV per (user, take) from annotations/03.09.2026/CSV and writes
annotations/ceiling_installation/label__uid-XX_take-YY[.R].{json,csv,eaf}. The
source name user_XX_take_YY[.R].csv supplies the ids; the optional .R suffix is
a take split across two files (user_07_take_02.1.csv -> label__uid-07_take-02.1).

Each CSV is an ELAN export: some commented "#file:///... -- offset: ...,
duration: ..., ms per sample: ..." lines, one per linked camera, then a table
whose first six columns are begin/end/duration times and whose remaining
columns are the annotation tiers. Every row fills exactly one tier, and that
cell's value is the piece id.

Output JSON:

    {"meta_data": {...},
     "labels": [{step_id, piece_id, timestamp,
                 start_frame, end_frame, start_progress, end_progress}, ...]}

  step_id     the row's tier, mapped through labels.ANNOTATION_STEP_IDS
  piece_id    the value in that tier's cell
  *_frame     converted with the CSV's own "ms per sample"
  *_progress  0..100 per annotation, so each labelled span runs 0% -> 100% on
              its own (--progress-mode per-group restores the older behaviour
              of splitting one 0..100 across repeats of a (step_id, piece_id))

A single annotation covers all three cameras, so the output names carry no
camera id and the files sit flat in one folder. Every media line is kept in
meta_data["media"], and their durations disagree -- the cameras were started
and stopped by hand, by up to 172s on one take -- because the labels are on
the ELAN timeline, which is not necessarily aligned to any one camera's video.

skeleton_pipeline/dataset/labels.py builds its per-frame task_progress from
these step labels.

Usage:
    cd data_proc_3d/app
    uv run python proc_annotations.py                       # dry run, prints what it would write
    uv run python proc_annotations.py --write --overwrite
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skeleton_pipeline.dataset.labels import ANNOTATION_STEP_IDS


VIDEOS_ROOT = Path(
    r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos")
ANNOTATIONS_ROOT = VIDEOS_ROOT / "annotations"
DEFAULT_SOURCE_DIR = ANNOTATIONS_ROOT / "03.09.2026" / "CSV"
DEFAULT_ELAN_DIR = ANNOTATIONS_ROOT / "03.09.2026" / "ELAN"
DEFAULT_OUTPUT_DIR = ANNOTATIONS_ROOT / "ceiling_installation"
DEFAULT_RAW_DIR = VIDEOS_ROOT / "raw"

# user_07_take_02.1.csv -> uid 07, take 02, round 1 (a take split across files)
SOURCE_NAME_PATTERN = re.compile(
    r"^user_(?P<user_id>\d+)_take_(?P<take_id>\d+)(?:\.(?P<round_id>\d+))?$")

# ELAN's linked media name, e.g. cam-05_user-07_take-02.1.mp4. The separator
# before "user"/"take" is usually "_" but one link was typed with "-"
# (cam-07-user-08_take-01.mp4), so accept either rather than failing to
# resolve over a typo in the annotator's filename.
MEDIA_NAME_PATTERN = re.compile(
    r"^cam-(?P<camera_id>\d+)[-_]user-(?P<user_id>\d+)[-_]take-(?P<take_id>\d+)"
    r"(?:\.(?P<round_id>\d+))?\.mp4$", re.IGNORECASE)

MEDIA_LINE_PATTERN = re.compile(
    r'^"?#(?P<file_url>file:///.*?)\s+--\s+'
    r'offset:\s+(?P<offset>\d+(?:\.\d+)?)\s*,\s*'
    r'duration:\s+(?P<duration_hms>\d{2}:\d{2}:\d{2}\.\d+)\s*/\s*'
    r'(?P<duration_sec>\d+(?:\.\d+)?)\s*/\s*'
    r'(?P<duration_ms>\d+)\s*,\s*'
    r'ms per sample:\s+(?P<ms_per_sample>\d+(?:\.\d+)?)"?\s*$')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--elan-dir", type=str, default=str(DEFAULT_ELAN_DIR),
                        help="Where the matching .eaf files live; they are copied "
                             "next to the JSON so the annotation stays re-openable.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR),
                        help="Root of the per-camera video folders (raw/cam-XX). "
                             "Each media line's ELAN path is rewritten to the "
                             "matching video here and checked for existence.")
    parser.add_argument("--progress-mode", choices=PROGRESS_MODES, default="per-label",
                        help="How start_progress/end_progress are filled. "
                             "'per-label' (default) gives every annotation its own "
                             "0..100; 'per-group' splits one 0..100 across repeats of "
                             "the same (step_id, piece_id). See assign_progress_ranges.")
    parser.add_argument("--write", action="store_true",
                        help="Actually write. Without it this is a dry run that "
                             "reports what would be produced -- the default, since "
                             "this writes into the shared Drive folder.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing outputs instead of skipping them.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    elan_dir = Path(args.elan_dir)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)

    csv_files = sorted(p for p in source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files in {source_dir}")
    print(f"{len(csv_files)} CSV file(s) in {source_dir}")
    print(f"output -> {output_dir}{'' if args.write else '   (DRY RUN -- pass --write)'}\n")

    written = skipped = failed = 0
    all_unknown = set()
    missing_videos = []
    for csv_path in csv_files:
        match = SOURCE_NAME_PATTERN.match(csv_path.stem)
        if not match:
            print(f"  SKIP {csv_path.name}: name does not match user_XX_take_YY[.R]")
            failed += 1
            continue

        user_id = int(match.group("user_id"))
        take_id = int(match.group("take_id"))
        round_id = match.group("round_id")
        base = f"label__uid-{user_id:02d}_take-{take_id:02d}"
        if round_id:
            base = f"{base}-{round_id}"

        media, header_index = read_media_and_header(csv_path, raw_dir)
        if not media:
            print(f"  SKIP {csv_path.name}: no parsable '#file:///...' media line")
            failed += 1
            continue

        ms_per_sample = media[0]["ms_per_sample"]
        try:
            labels, unknown, total_rows = load_label_rows(
                csv_path, ms_per_sample, header_index, args.progress_mode)
        except Exception as error:                       # noqa: BLE001 - report and continue
            print(f"  FAIL {csv_path.name}: {error!r}")
            failed += 1
            continue
        all_unknown.update(unknown)

        spread = max(m["duration_sec"] for m in media) - min(m["duration_sec"] for m in media)
        payload = {
            "meta_data": {
                "source_csv": csv_path.name,
                "user_id": user_id,
                "take_id": take_id,
                "round_id": int(round_id) if round_id else None,
                "ms_per_sample": ms_per_sample,
                "offset": media[0]["offset"],
                "duration_hms": media[0]["duration_hms"],
                "duration_sec": media[0]["duration_sec"],
                "duration_ms": media[0]["duration_ms"],
                # One annotation, several cameras -- keep them all, and keep the
                # disagreement visible rather than averaging it away.
                "media": media,
                "media_duration_spread_sec": round(spread, 3),
            },
            "labels": labels,
        }

        json_path = output_dir / f"{base}.json"
        if json_path.exists() and not args.overwrite:
            print(f"  SKIP {csv_path.name} -> {json_path.name} (exists; --overwrite to replace)")
            skipped += 1
            continue

        if args.write:
            output_dir.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

        copy_alongside(csv_path, output_dir / f"{base}.csv", args.write, args.overwrite)
        copy_alongside(elan_dir / f"{csv_path.stem}.eaf", output_dir / f"{base}.eaf",
                       args.write, args.overwrite)

        absent = [m for m in media if not m["video_exists"]]
        missing_videos.extend(
            (m["file_name"], m["video_path"]) for m in absent)
        flags = ""
        if spread > 10:
            flags += "  <-- camera durations differ by %.0fs" % spread
        if absent:
            flags += f"  <-- {len(absent)} video(s) not found"
        print(f"  {'WRITE' if args.write else 'would write'} {json_path.name}"
              f"  ({len(labels)}/{total_rows} rows, {len(media)} media){flags}")
        written += 1

    print(f"\n{written} written, {skipped} skipped, {failed} failed")
    if all_unknown:
        print(f"\nWARNING: tier column(s) not in ANNOTATION_STEP_IDS, so never labelled: "
              f"{sorted(all_unknown)}")
    if missing_videos:
        unparsed = sorted({name for name, path in missing_videos if path is None})
        not_on_disk = sorted({path for _name, path in missing_videos if path is not None})
        print(f"\nWARNING: {len(missing_videos)} media line(s) have no usable video. "
              f"video_path/video_exists\n         are still written, so the gap is "
              f"visible in the JSON rather than silent.")
        if unparsed:
            print("       ELAN media name did not parse (so no path could be built):")
            for name in unparsed:
                print(f"         {name}")
        if not_on_disk:
            print(f"       resolved, but no such file under {raw_dir}:")
            for path in not_on_disk:
                print(f"         {Path(path).relative_to(raw_dir)}")


def resolve_video_path(media_file_name, raw_dir):
    """Map ELAN's linked media name onto the video in Videos/raw/cam-XX.

    ELAN points at whatever copy the annotator had locally
    (.../videos/momo_01/cam-05_user-02_take-01.mp4), which is not where the
    pipeline reads from. The ids match, the spelling does not:

        cam-05_user-02_take-01.mp4    ->  raw/cam-05/video__cam-05_uid-02_take-01.mp4
        cam-05_user-07_take-02.1.mp4  ->  raw/cam-05/video__cam-05_uid-07_take-02-1.mp4

    i.e. "user-" becomes "uid-", the "video__" prefix is added, and a take
    split into rounds separates the round with "." in ELAN but "-" on Drive.

    Returns (camera_id, path) with path None when the name does not parse.
    """
    match = MEDIA_NAME_PATTERN.match(media_file_name)
    if not match:
        return None, None

    camera_id = match.group("camera_id")
    stem = (f"video__cam-{camera_id}"
            f"_uid-{match.group('user_id')}"
            f"_take-{match.group('take_id')}")
    if match.group("round_id"):
        stem = f"{stem}-{match.group('round_id')}"
    return camera_id, raw_dir / f"cam-{camera_id}" / f"{stem}.mp4"


def read_media_and_header(csv_path, raw_dir):
    """Return (media entries, index of the header row).

    ELAN writes one commented "#file:///... -- offset: ..., duration: ...,
    ms per sample: ..." line per linked medium, then the header. There is one
    per camera and takes do not all link the same number, so the header is
    found by scanning for the first uncommented line rather than by a fixed
    skiprows.
    """
    media = []
    header_index = 0
    with csv_path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped.lstrip('"').startswith("#"):
                header_index = index
                break
            match = MEDIA_LINE_PATTERN.match(stripped)
            if match:
                entry = match.groupdict()
                entry["offset"] = float(entry["offset"])
                entry["duration_sec"] = float(entry["duration_sec"])
                entry["duration_ms"] = int(entry["duration_ms"])
                entry["ms_per_sample"] = float(entry["ms_per_sample"])
                entry["file_name"] = entry["file_url"].rsplit("/", 1)[-1]
                camera_id, video_path = resolve_video_path(entry["file_name"], raw_dir)
                entry["camera_id"] = camera_id
                entry["video_path"] = str(video_path) if video_path else None
                entry["video_exists"] = bool(video_path and video_path.is_file())
                media.append(entry)
    return media, header_index


def time_to_frame(series, ms_per_sample):
    return series.apply(
        lambda value: int(round(value * 1000 / ms_per_sample)) if pd.notnull(value) else None)


PROGRESS_MODES = ("per-label", "per-group")


def assign_progress_ranges(frame, mode="per-label"):
    """Fill start_progress/end_progress. See PROGRESS_MODES.

    "per-label" (default): every annotation carries its own full 0..100, so a
    span runs from 0% at its start frame to 100% at its end frame regardless of
    what else is annotated. A repeat of the same (step_id, piece_id) means the
    subject did that step again -- after a mistake, or on a second pass -- and
    is therefore a fresh execution starting over at 0.

    "per-group": the original behaviour. Repeats of one (step_id, piece_id) are
    treated as consecutive slices of a single shared 0..100 range, in time
    order, so only the last repeat reaches 100. Note this makes the target
    depend on how many times the step happened to be annotated in that take,
    which is not something a model watching the video can observe -- keep it
    for reproducing earlier datasets rather than for new ones.
    """
    if mode not in PROGRESS_MODES:
        raise ValueError(f"progress mode must be one of {PROGRESS_MODES}, got {mode!r}")

    result = frame.copy()
    result["start_progress"] = 0
    result["end_progress"] = 100
    if mode == "per-label" or result.empty:
        return result

    for _key, group in result.groupby(["step_id", "piece_id"], sort=False):
        ordered = group.sort_values(
            by=["start_frame", "end_frame", "timestamp"], kind="stable")
        size = len(ordered)
        previous_end = -1
        for position, index in enumerate(ordered.index):
            end_progress = 100 if position == size - 1 else int(((position + 1) * 100) // size)
            start_progress = 0 if position == 0 else previous_end + 1
            result.at[index, "start_progress"] = start_progress
            result.at[index, "end_progress"] = end_progress
            previous_end = end_progress

    return result


def load_label_rows(csv_path, ms_per_sample, header_index, progress_mode="per-label"):
    """Parse one ELAN CSV into the list of label dicts the JSON carries."""
    frame = pd.read_csv(csv_path, skiprows=header_index)

    for column in ("Begin Time - ss.msec", "End Time - ss.msec", "Duration - ss.msec"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Tier columns are everything past the six time columns -- read them off the
    # CSV before any column of ours is added, or start_frame/end_frame below get
    # reported as unrecognised tiers.
    tier_columns = list(frame.columns[6:])

    frame["start_frame"] = time_to_frame(frame["Begin Time - ss.msec"], ms_per_sample)
    frame["end_frame"] = time_to_frame(frame["End Time - ss.msec"], ms_per_sample)

    # Each annotation row fills exactly one tier column; that cell's value is
    # the piece id. Restrict to tiers this file actually has: a lookup that
    # defaults a missing tier to 0 reads as "present, piece 0" and would
    # silently label every row with whichever absent tier is checked first.
    tiers = [tier for tier in ANNOTATION_STEP_IDS if tier in tier_columns]
    unknown = [c for c in tier_columns if c not in ANNOTATION_STEP_IDS]

    frame["step_id"] = pd.NA
    frame["piece_id"] = pd.NA
    for index, row in frame.iterrows():
        for tier in tiers:
            value = row[tier]
            if pd.isna(value):
                continue
            frame.at[index, "step_id"] = ANNOTATION_STEP_IDS[tier]
            frame.at[index, "piece_id"] = value
            break

    labelled = frame.dropna(subset=["step_id", "piece_id"]).copy()
    labelled = labelled.rename(columns={"Begin Time - ss.msec": "timestamp"})
    labelled = labelled[["step_id", "piece_id", "timestamp", "start_frame", "end_frame"]]
    labelled["step_id"] = labelled["step_id"].astype(int)
    labelled["piece_id"] = labelled["piece_id"].astype(int)

    labelled = assign_progress_ranges(labelled, progress_mode)
    return labelled.to_dict(orient="records"), unknown, len(frame)


def copy_alongside(source, destination, write, overwrite):
    if not source.exists():
        return None
    if destination.exists() and not overwrite:
        return "exists"
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return "copied"


if __name__ == "__main__":
    main()
