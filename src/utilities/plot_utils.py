import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))
    

# ============================================================
# Plotting  –  all panels in one image
# ============================================================
# ── Plot defaults ─────────────────────────────────────────────────────────────
# Y-axis scale per panel in plot_pose_analysis:
#   [speed, dist_hip, dist_shoulder, dist_half_body, acceleration]
DEFAULT_YSCALES: list[str] = ["log", "log", "linear", "linear", "linear"]


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



# Default y-axis scales per panel: log for speed & accel (high variance), linear for distances
_DEFAULT_YSCALES = DEFAULT_YSCALES


def plot_pose_analysis(
    speed_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    dist_half_df: pd.DataFrame,
    dist_hip_df: pd.DataFrame,
    dist_shoulder_df: pd.DataFrame,
    save_path: str,
    suptitle: str = "",
    yscales: list[str] | None = None,
) -> None:
    """Render speed, distances, and acceleration as a single multi-panel image.

    Panels (top to bottom):
        1. Joint speed
        2. Distance from mid-hip
        3. Distance from mid-shoulder
        4. Distance from half-body centre
        5. Joint acceleration

    Args:
        speed_df:         output of :func:`compute_joint_speeds_df`.
        dist_hip_df:      output of :func:`compute_joint_distances_df` with ``center="hip"``.
        dist_shoulder_df: output of :func:`compute_joint_distances_df` with ``center="shoulder"``.
        dist_half_df:     output of :func:`compute_joint_distances_df` with ``center="half_body"``.
        accel_df:         output of :func:`compute_joint_acceleration_df`.
        save_path:        destination PNG path.
        suptitle:         overall figure title (optional).
        yscales:          list of 5 y-axis scale strings (``"linear"`` or ``"log"``) for each
                          panel in order. Defaults to ``["log", "linear", "linear", "linear", "log"]``.
    """
    scales = yscales if yscales is not None else _DEFAULT_YSCALES

    panel_data = [
        (speed_df,         [f"{n}_speed" for n in PLOT_KEYPOINT_NAMES], "Speed (norm. units / frame)",            "Joint Speed"),
        (accel_df,         [f"{n}_accel" for n in PLOT_KEYPOINT_NAMES], "Acceleration (norm. units / frame²)",      "Joint Acceleration"),
        (dist_half_df,     [f"{n}_dist_half_body"  for n in PLOT_KEYPOINT_NAMES], "Distance from half-body (norm. units)",   "Distance from Half-Body"),
        (dist_hip_df,      [f"{n}_dist_hip"  for n in PLOT_KEYPOINT_NAMES], "Distance from mid-hip (norm. units)",     "Distance from Mid-Hip"),
        (dist_shoulder_df, [f"{n}_dist_shoulder"  for n in PLOT_KEYPOINT_NAMES], "Distance from mid-shoulder (norm. units)", "Distance from Mid-Shoulder"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(16, 22), sharex=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.002)

    for ax, (df, cols, ylabel, panel_title), scale in zip(axes, panel_data, scales):
        for col, name in zip(cols, PLOT_KEYPOINT_NAMES):
            ax.plot(df["frame"], df[col], linewidth=0.8, label=name)
        # log scale requires strictly positive values; fall back to linear if any zeros present
        if scale == "log":
            data_vals = df[cols].values
            if (data_vals <= 0).any():
                scale = "symlog"
        ax.set_yscale(scale)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(panel_title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper right")

    axes[-1].set_xlabel("Frame", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plotly interactive figure (for Streamlit / browser)
# ============================================================

def build_plotly_pose_figure(
    speed_df: pd.DataFrame,
    dist_hip_df: pd.DataFrame,
    dist_shoulder_df: pd.DataFrame,
    dist_half_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    yscales: list[str] | None = None,
    title: str = "",
    visible_joints: list[str] | None = None,
    center_speed_df: pd.DataFrame | None = None,
    center_accel_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Build a 5-panel interactive Plotly figure (no frame indicator).

    Call :func:`set_frame_indicator` separately to add/update the vertical
    current-frame line.  This split allows the heavy figure (all traces) to
    be cached while only the cheap shape update runs on every frame change.

    Panels (top to bottom):
        1. Joint speed
        2. Distance from mid-hip
        3. Distance from mid-shoulder
        4. Distance from half-body centre
        5. Joint acceleration

    Args:
        speed_df:         output of ``compute_joint_speeds_df``.
        dist_hip_df:      output of ``compute_joint_distances_df(center='hip')``.
        dist_shoulder_df: output of ``compute_joint_distances_df(center='shoulder')``.
        dist_half_df:     output of ``compute_joint_distances_df(center='half_body')``.
        accel_df:         output of ``compute_joint_acceleration_df``.
        current_frame:    frame index at which to draw the vertical indicator line.
        yscales:          list of 5 strings, each ``\"linear\"`` or ``\"log\"``.
                          Defaults to ``DEFAULT_YSCALES``.
        title:            overall figure title.
        visible_joints:   list of joint names (from ``PLOT_KEYPOINT_NAMES``) to
                          draw as solid lines; all others are set to
                          ``"legendonly"`` (hidden but toggleable in the legend).
                          Pass ``None`` to show all joints.
        center_speed_df:  output of ``compute_body_center_speed_df``; adds a
                          dashed overlay trace per centre to the speed panel.
        center_accel_df:  output of ``compute_body_center_accel_df``; adds a
                          dashed overlay trace per centre to the accel panel.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    scales = yscales if yscales is not None else _DEFAULT_YSCALES

    panel_cfg = [
        (speed_df,         [f"{n}_speed"         for n in PLOT_KEYPOINT_NAMES], "Speed (norm / frame)",    "Joint Speed"),
        (accel_df,         [f"{n}_accel"          for n in PLOT_KEYPOINT_NAMES], "Accel (norm / frame²)",   "Joint Acceleration"),
        (dist_half_df,     [f"{n}_dist_half_body" for n in PLOT_KEYPOINT_NAMES], "Dist half-body",          "Distance from Half-Body"),
        (dist_hip_df,      [f"{n}_dist_hip"       for n in PLOT_KEYPOINT_NAMES], "Dist mid-hip",            "Distance from Mid-Hip"),
        (dist_shoulder_df, [f"{n}_dist_shoulder"  for n in PLOT_KEYPOINT_NAMES], "Dist mid-shoulder",       "Distance from Mid-Shoulder"),
    ]

    n_panels = len(panel_cfg)
    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=False,
        subplot_titles=[cfg[3] for cfg in panel_cfg],
        vertical_spacing=0.06,
    )

    # One colour per joint, reused across panels
    colours = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
        "#c49c94","#f7b6d2",
    ]

    # Body-centre overlay config: (column_suffix, display_label, colour)
    _CENTER_OVERLAY = [
        ("hip_center",       "center: mid-hip",       "#5A5A5A"),
        ("shoulder_center",  "center: mid-shoulder",  "#5A5A5A"),
        ("half_body_center", "center: half-body",     "#5A5A5A"),
    ]

    for row, (df, cols, ylabel, _) in enumerate(panel_cfg, start=1):
        show_legend = (row == 1)
        for col, name, colour in zip(cols, PLOT_KEYPOINT_NAMES, colours):
            is_visible = visible_joints is None or name in visible_joints
            fig.add_trace(
                go.Scatter(
                    x=df["frame"],
                    y=df[col],
                    mode="lines",
                    name=name,
                    line=dict(width=1, color=colour),
                    showlegend=show_legend,
                    legendgroup=name,
                    visible=True if is_visible else "legendonly",
                ),
                row=row, col=1,
            )

        # ── body-centre overlay on speed panel (row 1) ────────────────────
        if row == 1 and center_speed_df is not None:
            for key, label, colour in _CENTER_OVERLAY:
                col_name = f"{key}_speed"
                if col_name in center_speed_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=center_speed_df["frame"],
                            y=center_speed_df[col_name],
                            mode="lines",
                            name=label,
                            line=dict(width=0.5, color=colour),
                            showlegend=True,
                            legendgroup=label,
                            visible=True,
                        ),
                        row=row, col=1,
                    )

        # ── body-centre overlay on accel panel (row 2) ────────────────────
        if row == 2 and center_accel_df is not None:
            for key, label, colour in _CENTER_OVERLAY:
                col_name = f"{key}_accel"
                if col_name in center_accel_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=center_accel_df["frame"],
                            y=center_accel_df[col_name],
                            mode="lines",
                            name=label,
                            line=dict(width=0.5, color=colour),
                            showlegend=False,
                            legendgroup=label,
                            visible=True,
                        ),
                        row=row, col=1,
                    )

        scale = scales[row - 1]
        fig.update_yaxes(title_text=ylabel, type=scale, row=row, col=1)

    fig.update_xaxes(title_text="Frame", row=n_panels, col=1)

    fig.update_layout(
        height=280 * n_panels,
        title_text=title,
        title_font_size=13,
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=9),
            tracegroupgap=2,
        ),
        margin=dict(l=60, r=160, t=60, b=40),
    )

    return fig


def set_frame_indicator(fig: go.Figure, frame_idx: int, n_panels: int = 5) -> None:
    """Update the vertical current-frame indicator on an existing figure.

    Replaces ``fig.layout.shapes`` with one vertical dashed line per panel.
    This is the only part that changes on every frame seek; the traces remain
    untouched, so Plotly's browser diff is near-zero.

    Args:
        fig:        A figure previously built by :func:`build_plotly_pose_figure`.
        frame_idx:  The frame index at which to draw the indicator.
        n_panels:   Number of subplot rows (must match the figure; default 5).
    """
    shapes = []
    for i in range(1, n_panels + 1):
        suffix = str(i) if i > 1 else ""
        shapes.append(dict(
            type="line",
            xref=f"x{suffix}",
            yref=f"y{suffix} domain",
            x0=frame_idx, x1=frame_idx,
            y0=0, y1=1,
            line=dict(color="red", width=1.5, dash="dash"),
        ))
    # datarevision changing tells Plotly to re-draw shapes/data while
    # uirevision (set constant on the figure) preserves zoom/pan/selection.
    fig.update_layout(shapes=shapes, datarevision=frame_idx)

