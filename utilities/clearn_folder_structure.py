import argparse
import re
from dataclasses import dataclass
from pathlib import Path


VIDEO_ROOT_PATH = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos\cropped")
DEFAULT_PREFIX = "video"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
DEFAULT_LEGACY_USER_MAP = {
    "B": "01",
    "Y": "02",
    "M": "03",
}

NORMALIZED_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)__cam-(?P<camera_id>\d+)_uid-(?P<user_id>\d+)_take-(?P<take_id>\d+)$",
    re.IGNORECASE,
)

CID_UID_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)__CID-(?P<camera_id>\d+)_UID-(?P<user_id>\d+)_(?P<take_id>\d+)$",
    re.IGNORECASE,
)

LEGACY_CAM_PATTERN = re.compile(
    r"^cam(?:era)?[-_ ]?(?P<camera_id>\d+)(?:[-_ ][A-Za-z0-9]+)*[-_ ](?P<user_code>[A-Za-z]+)(?P<take_id>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoNameParts:
    prefix: str
    camera_id: str
    user_id: str
    take_id: str
    extension: str

    def to_filename(self) -> str:
        return (
            f"{self.prefix}__cam-{self.camera_id}_uid-{self.user_id}_take-{self.take_id}"
            f".{self.extension}"
        )

    def camera_folder(self) -> str:
        return f"cam-{self.camera_id}"


def pad_id(value: str) -> str:
    return f"{int(value):02d}"


def parse_user_map(raw_entries: list[str]) -> dict[str, str]:
    user_map: dict[str, str] = dict(DEFAULT_LEGACY_USER_MAP)
    for entry in raw_entries:
        legacy_code, user_id = entry.split("=", maxsplit=1)
        user_map[legacy_code.strip().upper()] = pad_id(user_id.strip())
    return user_map


def parse_video_name(
    video_file: Path,
    prefix: str,
    legacy_user_map: dict[str, str],
) -> VideoNameParts | None:
    suffix = video_file.suffix.lower()
    stem = video_file.stem

    normalized_match = NORMALIZED_PATTERN.fullmatch(stem)
    if normalized_match:
        return VideoNameParts(
            prefix=normalized_match.group("prefix").lower(),
            camera_id=pad_id(normalized_match.group("camera_id")),
            user_id=pad_id(normalized_match.group("user_id")),
            take_id=pad_id(normalized_match.group("take_id")),
            extension=suffix.lstrip("."),
        )

    cid_uid_match = CID_UID_PATTERN.fullmatch(stem)
    if cid_uid_match:
        return VideoNameParts(
            prefix=prefix,
            camera_id=pad_id(cid_uid_match.group("camera_id")),
            user_id=pad_id(cid_uid_match.group("user_id")),
            take_id=pad_id(cid_uid_match.group("take_id")),
            extension=suffix.lstrip("."),
        )

    legacy_match = LEGACY_CAM_PATTERN.fullmatch(stem)
    if not legacy_match:
        return None

    user_code = legacy_match.group("user_code").upper()
    user_id = legacy_user_map.get(user_code)
    if user_id is None:
        raise ValueError(
            f"Missing --user-map entry for legacy participant code '{user_code}' in {video_file.name}"
        )

    return VideoNameParts(
        prefix=prefix,
        camera_id=pad_id(legacy_match.group("camera_id")),
        user_id=user_id,
        take_id=pad_id(legacy_match.group("take_id")),
        extension=suffix.lstrip("."),
    )


def rename_videos(
    root_path: Path,
    prefix: str,
    legacy_user_map: dict[str, str],
    apply_changes: bool,
) -> int:
    rename_count = 0

    for video_file in sorted(root_path.rglob("*")):
        if not video_file.is_file() or video_file.suffix.lower() not in VIDEO_SUFFIXES:
            continue

        try:
            parts = parse_video_name(video_file, prefix=prefix, legacy_user_map=legacy_user_map)
        except ValueError as exc:
            print(f"SKIP {video_file}: {exc}")
            continue

        if parts is None:
            print(f"SKIP {video_file}: unsupported filename pattern")
            continue

        target_dir = root_path / parts.camera_folder()
        target_path = target_dir / parts.to_filename()
        if target_path == video_file:
            print(f"OK   {video_file.name}")
            continue

        if target_path.exists():
            print(f"SKIP {video_file}: target already exists -> {target_path.name}")
            continue

        print(f"MOVE {video_file} -> {target_path}")
        rename_count += 1
        if apply_changes:
            target_dir.mkdir(parents=True, exist_ok=True)
            video_file.rename(target_path)

    if apply_changes:
        remove_empty_directories(root_path)

    return rename_count


def remove_empty_directories(root_path: Path) -> None:
    directories = sorted(
        (path for path in root_path.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rename video files to the standard {prefix}__cam-XX_uid-YY_take-ZZ.ext format "
            "and place them in cropped/cam-XX style folders."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=VIDEO_ROOT_PATH,
        type=Path,
        help="Root directory that contains the videos.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Prefix to use in renamed files. Default: video.",
    )
    parser.add_argument(
        "--user-map",
        action="append",
        default=[],
        metavar="LEGACY=UID",
        help=(
            "Override legacy participant codes, for example B=01. "
            "Defaults are B=01, Y=02, and M=03."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag the script performs a dry run.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    user_map = parse_user_map(args.user_map)
    args.apply = True
    
    rename_count = rename_videos(
        root_path=args.root,
        prefix=args.prefix.lower(),
        legacy_user_map=user_map,
        apply_changes=args.apply,
    )

    mode = "applied" if args.apply else "planned"
    print(f"{mode.capitalize()} {rename_count} rename(s).")


if __name__ == "__main__":
    main()
