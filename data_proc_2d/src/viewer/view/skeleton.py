"""
viewer/view/skeleton.py
------------------
Pure Plotly skeleton renderer.  No Streamlit dependency  Ejust in/out.

The caller is responsible for passing per-clip axis limits (computed once
from the full landmark array) so the skeleton stays stable across frames.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


# ── Segment group for each keypoint-index pair ───────────────────────────────
# Indices follow the COCO 17-keypoint layout used by YOLO-Pose.
_PAIR_TO_GROUP: dict[tuple[int, int], str] = {
    (0, 1): "head",  (0, 2): "head",
    (1, 3): "head",  (2, 4): "head",
    (5, 6): "torso",
    (5, 7): "arm",   (7, 9):  "arm",
    (6, 8): "arm",   (8, 10): "arm",
    (5, 11): "torso", (6, 12): "torso",
    (11, 12): "torso",
    (11, 13): "leg",  (13, 15): "leg",
    (12, 14): "leg",  (14, 16): "leg",
}

# Short display labels for each of the 17 COCO keypoints.
_JOINT_LABELS: list[str] = [
    "nose", "L.eye", "R.eye", "L.ear", "R.ear",
    "L.sh",  "R.sh",  "L.el",  "R.el",  "L.wr", "R.wr",
    "L.hp",  "R.hp",  "L.kn",  "R.kn",  "L.an", "R.an",
]


def render_skeleton_plotly(
    kpts:            np.ndarray,              # (17, 2) for a single frame
    x_lim:          tuple[float, float],
    y_lim:          tuple[float, float],
    frame_idx:      int,
    total_frames:   int,
    seg_colours:    dict[str, str],           # {"head": "#…", "arm": "#…", …}
    joint_colour:   str,
    bg_colour:      str,
    body_center_cfg: list[tuple[str, str]],  # [(label, hex_colour), …]
) -> go.Figure:
    """Build a Plotly skeleton figure for one frame.

    Pure function  Eno Streamlit calls, no global state.
    """
    # ── body-centre points ────────────────────────────────────────────────
    mid_hip      = (kpts[11] + kpts[12]) / 2.0
    mid_shoulder = (kpts[5]  + kpts[6])  / 2.0
    half_body    = (kpts[11] + kpts[12] + kpts[5] + kpts[6]) / 4.0
    centers      = [mid_hip, mid_shoulder, half_body]

    fig = go.Figure()

    # ── limb segments: group by colour ↁEone trace per colour ────────────
    # Using the None-gap trick so each colour group is a single WebGL path.
    colour_segs: dict[str, tuple[list, list]] = {}
    for (a, b), group in _PAIR_TO_GROUP.items():
        colour = seg_colours[group]
        if colour not in colour_segs:
            colour_segs[colour] = ([], [])
        xs, ys = colour_segs[colour]
        xs += [float(kpts[a, 0]), float(kpts[b, 0]), None]
        ys += [float(kpts[a, 1]), float(kpts[b, 1]), None]

    colour_to_group = {v: k for k, v in seg_colours.items()}
    for colour, (xs, ys) in colour_segs.items():
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=colour, width=2.5),
            name=colour_to_group.get(colour, ""),
            hoverinfo="skip",
            showlegend=True,
        ))

    # ── joint dots ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=kpts[:, 0].tolist(),
        y=kpts[:, 1].tolist(),
        mode="markers+text",
        marker=dict(color=joint_colour, size=8, line=dict(color="white", width=1)),
        text=_JOINT_LABELS,
        textposition="top right",
        textfont=dict(size=7, color="#aaaaaa"),
        name="joints",
        showlegend=False,
        hovertemplate="%{text}<extra></extra>",
    ))

    # ── body-centre diamonds ──────────────────────────────────────────────
    for (label, colour), pt in zip(body_center_cfg, centers):
        fig.add_trace(go.Scatter(
            x=[float(pt[0])],
            y=[float(pt[1])],
            mode="markers+text",
            marker=dict(
                color=colour, size=13, symbol="diamond",
                line=dict(color="white", width=1.5),
            ),
            text=[label],
            textposition="top right",
            textfont=dict(size=7, color=colour),
            name=label,
            showlegend=True,
            hovertemplate=f"{label}<extra></extra>",
        ))

    # ── layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        xaxis=dict(
            range=[x_lim[0], x_lim[1]],
            visible=False,
            fixedrange=False,
        ),
        yaxis=dict(
            range=[y_lim[1], y_lim[0]],   # inverted: image coords
            visible=False,
            fixedrange=False,
            scaleanchor="x",
            scaleratio=1.0,
        ),
        paper_bgcolor=bg_colour,
        plot_bgcolor=bg_colour,
        autosize=True,                     # fills any container (normal + expand)
        margin=dict(l=5, r=120, t=28, b=5),
        title=dict(
            text=f"Frame {frame_idx} / {total_frames - 1}",
            font=dict(color="#cccccc", size=9),
            x=0.5,
        ),
        legend=dict(
            font=dict(size=7, color="#cccccc"),
            bgcolor="rgba(0,0,0,0)",
            x=1.01, y=1.0,
        ),
        showlegend=True,
        uirevision="skeleton",             # preserves zoom/pan across reruns
    )

    return fig
