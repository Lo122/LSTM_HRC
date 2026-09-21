"""Utilities for renaming/formatting camera video files."""

from pathlib import Path


def prefix_filenames_with_folder_name(folder: Path, dry_run: bool = True) -> None:
    """Rename every file in `folder` by prefixing it with the folder's own name.

    Example:
        .../raw/cam-05/user-01_take-01_incomplete.MP4
        -> .../raw/cam-05/cam-05_user-01_take-01_incomplete.MP4

    Args:
        folder: Directory whose files should be renamed.
        dry_run: If True (default), only print what would be renamed without
            actually renaming anything. Set to False to perform the rename.
    """
    folder = Path(folder)
    prefix = folder.name

    for path in folder.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(f"{prefix}_"):
            # Already prefixed, skip to avoid double-prefixing.
            continue

        new_path = path.with_name(f"{prefix}_{path.name}")

        if dry_run:
            print(f"[dry-run] {path.name} -> {new_path.name}")
        else:
            path.rename(new_path)
            print(f"renamed: {path.name} -> {new_path.name}")


if __name__ == "__main__":
    target_folder = Path(
        r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-07"
    )

    # First run with dry_run=True to check the planned renames, then set to
    # False to actually rename the files.
    prefix_filenames_with_folder_name(target_folder, dry_run=False)
