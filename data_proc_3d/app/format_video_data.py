"""Rename video files to the project's standard mp4 naming convention.

Expected input filename pattern:
    cam-04_user-01_take-01.mp4
    cam-07_user-07_take-02.1.mp4

Output path pattern:
    <save-folder>/cam-04/video__cam-04_uid-01_take-01.mp4
    <save-folder>/cam-07/video__cam-07_uid-07_take-02-1.mp4

Edit the configuration values below and run this file directly from the editor.
By default it runs as a dry-run and only prints the planned actions.
"""

import re
import shutil
from pathlib import Path


DEFAULT_VIDEO_ROOT_FOLDER = Path(
    r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\cropped (1)"
)
DEFAULT_SAVE_FOLDER = Path(
    r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw"
)

EXECUTE_CHANGES = True
MOVE_FILES = False
OVERWRITE_EXISTING = False

SOURCE_PATTERN = re.compile(
    r"^cam-(?P<cam_id>\d{2})_user-(?P<user_id>\d{2})_take-(?P<take_id>\d{2})(?:\.(?P<round_id>\d+))?\.mp4$",
    re.IGNORECASE,
)

MAX_SEARCH_DEPTH = 2


def iter_video_files(video_root_folder: Path, max_depth: int):
    for src_path in sorted(video_root_folder.rglob("*.mp4")):
        relative_parts = src_path.relative_to(video_root_folder).parts
        depth = len(relative_parts) - 1
        if depth <= max_depth:
            yield src_path


def build_output_path(
    save_folder: Path,
    cam_id: str,
    user_id: str,
    take_id: str,
    round_id=None,
) -> Path:
    camera_folder = save_folder / f"cam-{cam_id}"
    round_suffix = f"-{round_id}" if round_id is not None else ""
    output_name = f"video__cam-{cam_id}_uid-{user_id}_take-{take_id}{round_suffix}.mp4"
    return camera_folder / output_name


def rename_and_save_videos(
    video_root_folder: Path,
    save_folder: Path,
    *,
    execute: bool,
    move_files: bool,
    overwrite: bool,
) -> int:
    if not video_root_folder.exists():
        raise FileNotFoundError(f"Source folder does not exist: {video_root_folder}")

    matched_count = 0
    operation_name = "move" if move_files else "copy"

    for src_path in iter_video_files(video_root_folder, MAX_SEARCH_DEPTH):

        match = SOURCE_PATTERN.match(src_path.name)
        if match is None:
            print(f"skip: {src_path.name} (does not match expected pattern)")
            continue

        matched_count += 1
        output_path = build_output_path(save_folder, **match.groupdict())

        if output_path.exists() and not overwrite:
            print(f"skip: {output_path} already exists")
            continue

        print(f"{operation_name}: {src_path} -> {output_path}")

        if not execute:
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if move_files:
            if output_path.exists() and overwrite:
                output_path.unlink()
            shutil.move(str(src_path), str(output_path))
        else:
            shutil.copy2(src_path, output_path)

    return matched_count


def main() -> None:
    matched_count = rename_and_save_videos(
        DEFAULT_VIDEO_ROOT_FOLDER,
        DEFAULT_SAVE_FOLDER,
        execute=EXECUTE_CHANGES,
        move_files=MOVE_FILES,
        overwrite=OVERWRITE_EXISTING,
    )
    if matched_count == 0:
        print("No matching mp4 files were found.")


if __name__ == "__main__":
    main()


