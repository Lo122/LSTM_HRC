import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_utils import (
    PLOT_ANGLE_NAMES,
    PLOT_KEYPOINT_NAMES,
    PLOT_RATIO_NAMES,
    STEP_ID_LABEL,
    _DEFAULT_YSCALES,
    _build_step_segments,
    _frame_axis_range,
    _infer_total_frames,
    _normalise_segment_labels,
)


def _add_plotly_segment_panel(
    fig: go.Figure,
    row: int,
    segments: list[dict[str, int]],
    id_to_label: dict[int, str] | None = None,
    showlegend: bool = True,
) -> None:
    step_ids = sorted({segment["step_id"] for segment in segments})
    colour_sequence = ["#577590", "#43aa8b", "#f3722c", "#f94144", "#277da1", "#9c89b8"]

    for index, step_id in enumerate(step_ids):
        matching_segments = [segment for segment in segments if segment["step_id"] == step_id]
        label = id_to_label.get(step_id, str(step_id)) if id_to_label else str(step_id)
        fig.add_trace(
            go.Bar(
                x=[segment["end_frame"] - segment["start_frame"] for segment in matching_segments],
                y=[label] * len(matching_segments),
                base=[segment["start_frame"] for segment in matching_segments],
                orientation="h",
                name=label,
                marker=dict(color=colour_sequence[index % len(colour_sequence)]),
                showlegend=showlegend,
                legendgroup=f"step:{label}",
                width=0.65,
            ),
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="Step", type="category", row=row, col=1)


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
    line_panel_cfg = [
        (speed_df, PLOT_KEYPOINT_NAMES, [f"{name}_speed" for name in PLOT_KEYPOINT_NAMES], "Speed (norm / frame)", "Joint Speed", 1),
        (accel_df, PLOT_KEYPOINT_NAMES, [f"{name}_accel" for name in PLOT_KEYPOINT_NAMES], "Accel (norm / frame²)", "Joint Acceleration", 2),
        (dist_half_df, PLOT_KEYPOINT_NAMES, [f"{name}_dist_half_body" for name in PLOT_KEYPOINT_NAMES], "Dist half-body", "Distance from Half-Body", 3),
        (ratio_df, PLOT_RATIO_NAMES, [f"{name}_ratio" for name in PLOT_RATIO_NAMES], "Ratio", "Distance Ratios", 7),
        (angle_df, PLOT_ANGLE_NAMES, [f"{name}_angle_deg" for name in PLOT_ANGLE_NAMES], "Angle (deg)", "Joint Angles", 6),
    ]

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
    elan_segments = _normalise_segment_labels(elan_step_labels, total_frames)

    if yscales is None:
        scales = list(_DEFAULT_YSCALES)
    else:
        scales = list(yscales)
        if len(scales) < len(line_panel_cfg):
            scales.extend(_DEFAULT_YSCALES[len(scales):len(line_panel_cfg)])
        elif len(scales) > len(line_panel_cfg):
            scales = scales[:len(line_panel_cfg)]

    panel_titles = [cfg[4] for cfg in line_panel_cfg]
    row_heights = [4.0] * len(line_panel_cfg)
    if step_segments:
        panel_titles.append("Step Labels")
        row_heights.append(1.0)
    if elan_segments:
        panel_titles.append("ELAN Step Labels")
        row_heights.append(1.0)
    panel_count = len(panel_titles)

    fig = make_subplots(
        rows=panel_count,
        cols=1,
        shared_xaxes=False,
        subplot_titles=panel_titles,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2",
    ]
    ratio_colours = ["#003049", "#d62828", "#f77f00", "#588157", "#6a4c93"]

    center_overlay_cfg = [
        ("hip_center", "center: mid-hip", "#5A5A5A"),
        ("shoulder_center", "center: mid-shoulder", "#7A7A7A"),
        ("half_body_center", "center: half-body", "#9A9A9A"),
    ]

    for row, (df, names, cols, ylabel, panel_title, panel_index) in enumerate(line_panel_cfg, start=1):
        scale = scales[row - 1]
        if scale == "log" and not df.empty:
            data_vals = df[cols].to_numpy()
            if (data_vals <= 0).any():
                scale = "linear"

        for trace_index, (name, col_name) in enumerate(zip(names, cols)):
            if panel_index <= 5:
                colour = colours[trace_index % len(colours)]
                is_visible = visible_joints is None or name in visible_joints
                show_legend = (row == 1)
                legend_group = f"joint:{name}"
                visible = True if is_visible else "legendonly"
            elif panel_index == 6:
                colour = colours[trace_index % len(colours)]
                show_legend = True
                legend_group = f"angle:{name}"
                visible = True
            else:
                colour = ratio_colours[trace_index % len(ratio_colours)]
                show_legend = True
                legend_group = f"ratio:{name}"
                visible = True

            fig.add_trace(
                go.Scatter(
                    x=df["frame"],
                    y=df[col_name],
                    mode="lines",
                    name=name,
                    line=dict(width=1, color=colour),
                    showlegend=show_legend,
                    legendgroup=legend_group,
                    visible=visible,
                ),
                row=row,
                col=1,
            )

        if panel_index == 1 and center_speed_df is not None:
            for key, label, colour in center_overlay_cfg:
                col_name = f"{key}_speed"
                if col_name in center_speed_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=center_speed_df["frame"],
                            y=center_speed_df[col_name],
                            mode="lines",
                            name=label,
                            line=dict(width=1, color=colour, dash="dash"),
                            showlegend=True,
                            legendgroup=f"overlay:{label}",
                            visible=True,
                        ),
                        row=row,
                        col=1,
                    )

        if panel_index == 2 and center_accel_df is not None:
            for key, label, colour in center_overlay_cfg:
                col_name = f"{key}_accel"
                if col_name in center_accel_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=center_accel_df["frame"],
                            y=center_accel_df[col_name],
                            mode="lines",
                            name=label,
                            line=dict(width=1, color=colour, dash="dash"),
                            showlegend=False,
                            legendgroup=f"overlay:{label}",
                            visible=True,
                        ),
                        row=row,
                        col=1,
                    )

        fig.update_yaxes(title_text=ylabel, type=scale, row=row, col=1)

    next_row = len(line_panel_cfg) + 1
    if step_segments:
        _add_plotly_segment_panel(fig, next_row, step_segments, STEP_ID_LABEL, showlegend=True)
        next_row += 1
    if elan_segments:
        _add_plotly_segment_panel(fig, next_row, elan_segments, STEP_ID_LABEL, showlegend=not step_segments)

    for row in range(1, panel_count + 1):
        fig.update_xaxes(range=[x_axis_min, x_axis_max], row=row, col=1)
    fig.update_xaxes(title_text="Frame", row=panel_count, col=1)
    fig.update_layout(
        height=220 * panel_count,
        title_text=title,
        title_font_size=13,
        barmode="overlay",
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=9),
            tracegroupgap=2,
        ),
        margin=dict(l=60, r=180, t=70, b=40),
        meta={"n_panels": panel_count},
    )

    return fig


def set_frame_indicator(fig: go.Figure, frame_idx: int, n_panels: int | None = None) -> None:
    if n_panels is None:
        meta = getattr(fig.layout, "meta", None)
        if isinstance(meta, dict) and "n_panels" in meta:
            n_panels = int(meta["n_panels"])
        else:
            n_panels = 7

    shapes = []
    for panel_index in range(1, n_panels + 1):
        suffix = str(panel_index) if panel_index > 1 else ""
        shapes.append({
            "type": "line",
            "xref": f"x{suffix}",
            "yref": f"y{suffix} domain",
            "x0": frame_idx,
            "x1": frame_idx,
            "y0": 0,
            "y1": 1,
            "line": {"color": "red", "width": 1.5, "dash": "dash"},
        })
    fig.update_layout(shapes=shapes, datarevision=frame_idx)