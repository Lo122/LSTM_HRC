"""
viewer/controller/left_panel.py
--------------------------------
Left-column controller: streaming controls, frame slider, skeleton display,
optional video frame, and metadata expander (Controller layer).
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..config          import ViewerConfig
from ..model.app_state import ViewerState
from ..view.skeleton   import render_skeleton_plotly
from ..view.video      import grab_video_frame


def render_left_panel(
    cfg:          ViewerConfig,
    state:        ViewerState,
    landmarks_np: np.ndarray,          # (T, 17, 2)
    total_frames: int,
    fps_meta:     float | None,
    x_lim:        tuple[float, float],
    y_lim:        tuple[float, float],
    video_path:   str,
    video_ok:     bool,
    metadata:     dict,
) -> tuple[int, float]:
    """Render the left column.

    Returns
    -------
    frame_idx  : int   – the currently selected frame
    play_speed : float – the selected playback speed multiplier
    """
    st.subheader("Frame Inspector")

    # ── Streaming controls ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Play", width="stretch"):
            state.playing = True
    with c2:
        if st.button("⏸ Pause", width="stretch"):
            state.playing = False
    with c3:
        if st.button("⏹ Stop", width="stretch"):
            state.playing = False
            state.frame   = 0

    play_speed: float = st.select_slider(
        "Playback speed",
        options=cfg.speed_options,
        value=cfg.default_speed,
        key="play_speed",
        format_func=lambda v: f"{v}×",
    )

    # The slider is driven by value=state.frame so that any external seek
    # (stream loop, chart click, Stop button) is reflected immediately without
    # needing a separate widget key.  This also prevents the slider from
    # being reset mid-drag, which is what the two-key sync caused.
    frame_idx: int = st.slider(
        "Frame",
        min_value=0,
        max_value=total_frames - 1,
        value=state.frame,
        help="Drag to scrub. Click a data point on the chart (right) to seek.",
    )
    # Persist manual drag back to the backing key.
    state.frame = frame_idx

    if fps_meta:
        st.caption(f"t = {frame_idx / fps_meta:.3f} s  (@ {fps_meta:.1f} fps)")

    # ── Plotly skeleton ───────────────────────────────────────────────────
    fig_skeleton = render_skeleton_plotly(
        kpts=landmarks_np[frame_idx],
        x_lim=x_lim,
        y_lim=y_lim,
        frame_idx=frame_idx,
        total_frames=total_frames,
        seg_colours=cfg.seg_colours,
        joint_colour=cfg.joint_colour,
        bg_colour=cfg.bg_colour,
        body_center_cfg=cfg.body_center_cfg,
    )
    st.plotly_chart(fig_skeleton, width="stretch", key="skeleton_chart")

    # ── Optional video frame ──────────────────────────────────────────────
    if video_ok:
        rgb = grab_video_frame(video_path, frame_idx)
        if rgb is not None:
            st.image(rgb, caption=f"Video — frame {frame_idx}", width="stretch")
        else:
            st.warning("Could not read video frame.")
    elif video_path:
        st.caption(f"⚠ Video not accessible at stored path:\n`{video_path}`")

    # ── Metadata expander ─────────────────────────────────────────────────
    if metadata:
        with st.expander("Metadata", expanded=False):
            st.json({k: str(v) for k, v in metadata.items()})

    return frame_idx, play_speed
