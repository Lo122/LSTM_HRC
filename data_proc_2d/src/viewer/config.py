"""
viewer/config.py
----------------
All tunable constants for the Pose Sync Viewer.

Edit this file to customise paths, colours, default joint selection,
playback speeds, and layout — without touching any of the UI modules.

Import the global singleton:
    from viewer.config import cfg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ── Project-level paths (derived from this file's location) ───────────────
#   config.py  →  viewer/  →  application/  →  project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_ROOT:     Path = PROJECT_ROOT / "src"


@dataclass
class ViewerConfig:
    # ── Data source ─────────────────────────────────────────────────────────
    # Root directory that contains .pt pose files (sub-directories are scanned).
    dataset_root: Path = field(
        default_factory=lambda: (
            Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos") / ".pt"
        )
    )
    step_label_root: Path = field(
        default_factory=lambda: (
            Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos") / ".json"
        )
    )
    elan_label_root: Path = field(
        default_factory=lambda: Path(r"G:\My Drive\University of Stuttgart\ITECH_Thesis\ELAN")
    )
    # Maximum .pt files shown in the sidebar dropdown.  0 = no limit.
    max_files_in_picker: int = 0
    # YOLO pose model used by the one-click raw-keypoint extractor in the overlay tab.
    yolo_model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "yolo26n-pose.pt")

    # ── Page ────────────────────────────────────────────────────────────────
    page_title: str  = "Pose Sync Viewer"
    page_icon:  str  = ""
    layout:     Literal["centered", "wide"] = "wide"
    left_col_weight:  int = 1
    right_col_weight: int = 2

    # ── Skeleton rendering ──────────────────────────────────────────────────
    # Padding (in normalised coords) around the pose bounding box.
    axis_pad:    float = 0.2
    bg_colour:   str   = "#0d1117"
    joint_colour: str  = "#ff6b35"

    # Limb-segment colours keyed by body region.
    seg_colours: dict[str, str] = field(default_factory=lambda: {
        "head":  "#f9c74f",
        "arm":   "#90be6d",
        "torso": "#4ecdc4",
        "leg":   "#577590",
    })

    # Body-centre marker definitions: list of (label, hex_colour).
    body_center_cfg: list[tuple[str, str]] = field(default_factory=lambda: [
        ("mid-hip",      "#ff6b35"),
        ("mid-shoulder", "#00b4d8"),
        ("half-body",    "#c77dff"),
    ])

    # ── Chart (right panel) ──────────────────────────────────────────────────
    # Joints shown by default in the multiselect; others are visible via legend.
    default_visible_joints: list[str] = field(default_factory=lambda: [
        "nose",
        "left_wrist",
        "right_wrist",
        "left_shoulder",
        "right_shoulder",
    ])

    # ── Playback ────────────────────────────────────────────────────────────
    speed_options:       list[float] = field(default_factory=lambda: [
        0.125, 0.25, 0.5, 1.0, 2.0, 4.0
    ])
    default_speed:       float = 1.0
    # Never sleep shorter than this (keeps the UI responsive).
    min_sleep_s:         float = 0.02
    default_fps_fallback: float = 30.0


# Global singleton — import and use this everywhere instead of instantiating
# ViewerConfig manually.
cfg: ViewerConfig = ViewerConfig()
