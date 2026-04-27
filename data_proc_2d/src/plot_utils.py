import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

PROJECT_SRC_ROOT = Path(__file__).resolve().parent
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

from yolo_pose_config import (
    JOINT_ANGLE_TRIPLETS,
    JOINT_ANGLE_TRIPLETS_CAL,
    RATIO_BETWEEN_DISTS,
    STEP_ID_LABEL,
)
    

# ============================================================
# Plotting  –  all panels in one image
# ============================================================
# ── Plot defaults ─────────────────────────────────────────────────────────────
# Y-axis scale per panel in plot_pose_analysis:
#   [speed, acceleration, dist_half_body, dist_hip, dist_shoulder, joint_angles, distance_ratios]
DEFAULT_YSCALES: list[str] = ["log", "log", "linear", "linear", "linear", "linear", "log"]


# ── Keypoint names (index 0-16, COCO order) ──────────────────────────────────
PLOT_KEYPOINT_NAMES: list[str] = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
]

PLOT_ANGLE_NAMES: list[str] = [name for name, *_ in JOINT_ANGLE_TRIPLETS] +\
    [name for name, *_ in JOINT_ANGLE_TRIPLETS_CAL]
PLOT_RATIO_NAMES: list[str] = [name for name, *_ in RATIO_BETWEEN_DISTS]



# Default y-axis scales per panel: log for speed & accel (high variance), linear for distances
_DEFAULT_YSCALES = DEFAULT_YSCALES


@dataclass
class PanelData:
    df: pd.DataFrame
    names: list[str]
    cols: list[str]
    ylabel: str
    title: str
    yscale: str = "linear"


@dataclass
class VideoFeaturePanels:
    video_name: str
    feature_panel: PanelData
    annotations: dict[str, list[dict]] | None = None

def _infer_total_frames(*dfs: pd.DataFrame | None) -> int:
    max_frame = -1
    for df in dfs:
        if df is None or df.empty or "frame" not in df.columns:
            continue
        max_frame = max(max_frame, int(df["frame"].max()))
    return max_frame + 1 if max_frame >= 0 else 0


def _frame_axis_range(total_frames: int) -> tuple[int, int]:
    """Return a stable frame-axis range shared by all panels."""
    return 0, max(total_frames, 1)


def _infer_total_frames_from_panel_data(
    panel_data: list[PanelData],
) -> int:
    total_frames = 0
    for panel in panel_data:
        df = panel.df
        if df is None or df.empty:
            continue
        if "frame" in df.columns:
            total_frames = max(total_frames, int(df["frame"].max()) + 1)
        else:
            total_frames = max(total_frames, len(df.index))
    return total_frames


def _build_step_segments(
    step_labels: dict[str, list[dict]] | None,
    total_frames: int,
) -> list[dict[str, int]]:
    """Convert discrete step markers into frame-span segments for plotting."""
    if not step_labels or total_frames <= 0:
        return []

    markers: list[dict[str, int]] = []
    for label in step_labels.get("step_markers", []):
        frame = label.get("frame")
        step_id = label.get("step_id")
        if frame is None or step_id is None:
            continue
        markers.append({
            "frame": max(0, min(int(round(frame)), total_frames - 1)),
            "step_id": int(step_id),
        })

    if not markers:
        return []

    markers.sort(key=lambda item: item["frame"])
    segments: list[dict[str, int]] = []
    for index, marker in enumerate(markers):
        start_frame = marker["frame"]
        end_frame = total_frames
        if index + 1 < len(markers):
            end_frame = markers[index + 1]["frame"]
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, total_frames)
        if end_frame <= start_frame:
            continue
        segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "step_id": marker["step_id"],
        })

    return segments


def _normalise_segment_labels(
    labels: dict[str, list[dict]] | None,
    total_frames: int,
) -> list[dict[str, int]]:
    if not labels or total_frames <= 0:
        return []

    segments: list[dict[str, int]] = []
    for label in labels.get("step_markers", []):
        step_id = label.get("step_id")
        start_frame = label.get("start_frame", label.get("frame"))
        end_frame = label.get("end_frame")
        if step_id is None or start_frame is None:
            continue

        start = max(0, min(int(round(start_frame)), total_frames - 1))
        if end_frame is None:
            end = min(start + 1, total_frames)
        else:
            end = max(0, min(int(round(end_frame)), total_frames))
        if end <= start:
            end = min(start + 1, total_frames)
        if end <= start:
            continue

        segments.append({
            "start_frame": start,
            "end_frame": end,
            "step_id": int(step_id),
        })

    return segments


def _plot_segment_panel(
    ax,
    segments: list[dict[str, int]],
    title: str,
    id_to_label: dict[int, str] | None = None,
) -> None:
    """Render a compact horizontal bar panel for step segments."""
    step_ids = sorted({segment["step_id"] for segment in segments})
    colour_map = plt.get_cmap("tab10", max(len(step_ids), 1))
    step_colours = {step_id: colour_map(index) for index, step_id in enumerate(step_ids)}

    for segment in segments:
        width = segment["end_frame"] - segment["start_frame"]
        step_id = segment["step_id"]
        label = id_to_label.get(step_id, str(step_id)) if id_to_label else str(step_id)
        ax.barh(
            y=step_id,
            width=width,
            left=segment["start_frame"],
            height=0.75,
            color=step_colours[step_id],
            edgecolor="black",
            linewidth=0.3,
            label=label,
        )

    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylabel("Step ID", fontsize=8)
    ax.set_yticks(step_ids)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.tick_params(labelsize=7)


def _plot_annotation_panel(
    ax,
    annotations: dict[str, list[dict]] | None,
    video_name: str,
    total_frames: int,
) -> None:
    annotations = annotations or {}
    elan_segments = _normalise_segment_labels(
        annotations.get("elan_step_labels", {}),
        total_frames,
    )
    if elan_segments:
        _plot_segment_panel(ax, elan_segments, f"Annotations: {video_name} (ELAN)", STEP_ID_LABEL)
        return

    step_segments = _build_step_segments(
        annotations.get("step_labels", {}),
        total_frames,
    )
    if step_segments:
        _plot_segment_panel(ax, step_segments, f"Annotations: {video_name} (Step Labels)", STEP_ID_LABEL)
        return

    ax.set_title(f"Annotations: {video_name} (empty)", fontsize=9)
    ax.set_ylabel("Step ID", fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.tick_params(labelsize=7)


def _plot_feature(ax, panel, x_axis_min, x_axis_max) -> None:
    
    df = panel.df
    if df is None or df.empty:
        ax.set_ylabel(panel.ylabel, fontsize=8)
        ax.set_title(f"{panel.title} (empty)", fontsize=9)
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.tick_params(labelsize=7)
        return

    frame_values = df["frame"] if "frame" in df.columns else df.index
    valid_pairs = [(name, col) for name, col in zip(panel.names, panel.cols) if col in df.columns]
    for name, col in valid_pairs:
        ax.plot(frame_values, df[col], linewidth=0.8, label=name)
    # log scale requires strictly positive values; fall back to linear if any zeros present
    if panel.yscale == "log":
        valid_cols = [col for _, col in valid_pairs]
        data_vals = df[valid_cols].values if valid_cols else None
        if data_vals is not None and (data_vals <= 0).any():
            panel.yscale = "symlog"
    ax.set_yscale(panel.yscale)
    ax.set_ylabel(panel.ylabel, fontsize=8)
    ax.set_title(panel.title, fontsize=9)
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.tick_params(labelsize=7)
    if valid_pairs:
        ax.legend(fontsize=6, ncol=3, loc="upper right")



def plot_features(
    panel_data: list[PanelData],
    annotations,
    save_path: str | None = None,
    suptitle: str = "",
    show: bool = False,
) -> None:
    
    total_frames = _infer_total_frames_from_panel_data(panel_data)
    x_axis_min, x_axis_max = _frame_axis_range(total_frames)
    step_segments = _build_step_segments(annotations.get('step_labels', {}), total_frames)
    elan_step_segments = _normalise_segment_labels(annotations.get('elan_step_labels', {}), total_frames)

    height_ratios = [5.0] * len(panel_data)
    if step_segments:
        height_ratios.append(1.0)
    if elan_step_segments:
        height_ratios.append(1.0)

    panel_count = len(height_ratios)
    fig_height = 1.1 * sum(height_ratios)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(16, fig_height),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if panel_count == 1:
        axes = [axes]
    else:
        axes = list(axes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.002)

    for ax, panel in zip(axes[:len(panel_data)], panel_data):
        _plot_feature(ax, panel, x_axis_min, x_axis_max)
        
    next_panel_index = len(panel_data)
    if step_segments:
        _plot_segment_panel(axes[next_panel_index], step_segments, "Step Labels", STEP_ID_LABEL)
        axes[next_panel_index].set_xlim(x_axis_min, x_axis_max)
        next_panel_index += 1
    if elan_step_segments:
        _plot_segment_panel(axes[next_panel_index], elan_step_segments, "ELAN Step Labels", STEP_ID_LABEL)
        axes[next_panel_index].set_xlim(x_axis_min, x_axis_max)

    axes[-1].set_xlabel("Frame", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_features_by_video(
    video_panels: list[VideoFeaturePanels],
    save_path: str | None = None,
    suptitle: str = "",
    show: bool = False,
) -> None:
    """Render alternating feature and annotation panels for each video."""
    if not video_panels:
        return

    height_ratios: list[float] = []
    for _ in video_panels:
        height_ratios.extend([5.0, 1.2])

    panel_count = len(height_ratios)
    fig_height = 1.1 * sum(height_ratios)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(16, fig_height),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = [axes] if panel_count == 1 else list(axes)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.002)

    for index, video_panel in enumerate(video_panels):
        feature_ax = axes[index * 2]
        annotation_ax = axes[index * 2 + 1]
        total_frames = _infer_total_frames_from_panel_data([video_panel.feature_panel])
        x_axis_min, x_axis_max = _frame_axis_range(total_frames)

        _plot_feature(feature_ax, video_panel.feature_panel, x_axis_min, x_axis_max)
        feature_ax.set_title(video_panel.feature_panel.title, fontsize=9)

        _plot_annotation_panel(
            annotation_ax,
            video_panel.annotations,
            video_panel.video_name,
            total_frames,
        )
        annotation_ax.set_xlim(x_axis_min, x_axis_max)
        annotation_ax.set_xlabel("Frame", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)



def plot_pose_analysis(
    speed_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    dist_half_df: pd.DataFrame,
    dist_hip_df: pd.DataFrame,
    dist_shoulder_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    save_path: str,
    suptitle: str = "",
    yscales: list[str] | None = None,
    step_labels: list[dict] | None = None,
    elan_step_labels: list[dict] | None = None,
) -> None:
    """Render speed, distances, and acceleration as a single multi-panel image.

    Panels (top to bottom):
        1. Joint speed
        2. Joint acceleration
        3. Distance from half-body centre
        4. Distance from mid-hip
        5. Distance from mid-shoulder
        6. Joint angles
        7. Distance ratios
        8. Step labels (optional)
        9. ELAN step labels (optional)

    Args:
        speed_df:         output of :func:`compute_joint_speeds_df`.
        accel_df:         output of :func:`compute_joint_acceleration_df`.
        angle_df:         output of :func:`compute_joint_angles_df`.
        dist_hip_df:      output of :func:`compute_joint_distances_df` with ``center="hip"``.
        dist_shoulder_df: output of :func:`compute_joint_distances_df` with ``center="shoulder"``.
        dist_half_df:     output of :func:`compute_joint_distances_df` with ``center="half_body"``.
        ratio_df:         output of :func:`compute_distance_ratios_df`.
        save_path:        destination PNG path.
        suptitle:         overall figure title (optional).
        yscales:          list of 7 y-axis scale strings (``"linear"`` or ``"log"``) for each
                          panel in order.
        step_labels:      optional list of step markers loaded from JSON.
        elan_step_labels: optional list of ELAN step markers loaded from JSON.
    """
    panel_data = [
        (speed_df,         PLOT_KEYPOINT_NAMES, [f"{n}_speed" for n in PLOT_KEYPOINT_NAMES], "Speed (norm. units / frame)",            "Joint Speed"),
        (accel_df,         PLOT_KEYPOINT_NAMES, [f"{n}_accel" for n in PLOT_KEYPOINT_NAMES], "Acceleration (norm. units / frame²)",      "Joint Acceleration"),
        (dist_half_df,     PLOT_KEYPOINT_NAMES, [f"{n}_dist_half_body"  for n in PLOT_KEYPOINT_NAMES], "Distance from half-body (norm. units)",   "Distance from Half-Body"),
        (dist_hip_df,      PLOT_KEYPOINT_NAMES, [f"{n}_dist_hip"  for n in PLOT_KEYPOINT_NAMES], "Distance from mid-hip (norm. units)",     "Distance from Mid-Hip"),
        (dist_shoulder_df, PLOT_KEYPOINT_NAMES, [f"{n}_dist_shoulder"  for n in PLOT_KEYPOINT_NAMES], "Distance from mid-shoulder (norm. units)", "Distance from Mid-Shoulder"),
        (angle_df,         PLOT_ANGLE_NAMES,    [f"{n}_angle_deg" for n in PLOT_ANGLE_NAMES], "Angle (deg)",                              "Joint Angles"),
        (ratio_df,         PLOT_RATIO_NAMES,    [f"{n}_ratio" for n in PLOT_RATIO_NAMES], "Ratio",                                         "Distance Ratios"),
    ]

    if yscales is None:
        scales = list(_DEFAULT_YSCALES)
    else:
        scales = list(yscales)
        if len(scales) < len(panel_data):
            scales.extend(_DEFAULT_YSCALES[len(scales):len(panel_data)])
        elif len(scales) > len(panel_data):
            scales = scales[:len(panel_data)]

    total_frames = _infer_total_frames(
        speed_df,
        accel_df,
        angle_df,
        ratio_df,
        dist_half_df,
        dist_hip_df,
        dist_shoulder_df,
    )
    x_axis_min, x_axis_max = _frame_axis_range(total_frames)
    step_segments = _build_step_segments(step_labels, total_frames)
    elan_step_segments = _normalise_segment_labels(elan_step_labels, total_frames)

    height_ratios = [4.0] * len(panel_data)
    if step_segments:
        height_ratios.append(1.0)
    if elan_step_segments:
        height_ratios.append(1.0)

    panel_count = len(height_ratios)
    fig_height = 1.1 * sum(height_ratios)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(16, fig_height),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if panel_count == 1:
        axes = [axes]
    else:
        axes = list(axes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.002)

    for ax, (df, names, cols, ylabel, panel_title), scale in zip(axes[:len(panel_data)], panel_data, scales):
        for col, name in zip(cols, names):
            ax.plot(df["frame"], df[col], linewidth=0.8, label=name)
        # log scale requires strictly positive values; fall back to linear if any zeros present
        if scale == "log":
            data_vals = df[cols].values
            if (data_vals <= 0).any():
                scale = "symlog"
        ax.set_yscale(scale)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper right")

    next_panel_index = len(panel_data)
    if step_segments:
        _plot_segment_panel(axes[next_panel_index], step_segments, "Step Labels", STEP_ID_LABEL)
        axes[next_panel_index].set_xlim(x_axis_min, x_axis_max)
        next_panel_index += 1
    if elan_step_segments:
        _plot_segment_panel(axes[next_panel_index], elan_step_segments, "ELAN Step Labels", STEP_ID_LABEL)
        axes[next_panel_index].set_xlim(x_axis_min, x_axis_max)

    axes[-1].set_xlabel("Frame", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_plotly_pose_figure(
    speed_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    dist_half_df: pd.DataFrame,
    dist_hip_df: pd.DataFrame,
    dist_shoulder_df: pd.DataFrame,
    step_labels: list[dict] | None = None,
    elan_step_labels: list[dict] | None = None,
    yscales: list[str] | None = None,
    title: str = "",
    visible_joints: list[str] | None = None,
    center_speed_df: pd.DataFrame | None = None,
    center_accel_df: pd.DataFrame | None = None,
) -> go.Figure:
    from viewer.view.pose_chart import build_plotly_pose_figure as _impl

    return _impl(
        speed_df=speed_df,
        accel_df=accel_df,
        angle_df=angle_df,
        ratio_df=ratio_df,
        dist_half_df=dist_half_df,
        dist_hip_df=dist_hip_df,
        dist_shoulder_df=dist_shoulder_df,
        step_labels=step_labels,
        elan_step_labels=elan_step_labels,
        yscales=yscales,
        title=title,
        visible_joints=visible_joints,
        center_speed_df=center_speed_df,
        center_accel_df=center_accel_df,
    )


def set_frame_indicator(fig: go.Figure, frame_idx: int, n_panels: int | None = None) -> None:
    from viewer.view.pose_chart import set_frame_indicator as _impl

    _impl(fig, frame_idx, n_panels=n_panels)