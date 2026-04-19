"""
viewer/controller/right_panel.py
---------------------------------
Right-column controller: joint multiselect, interactive pose-analysis chart,
and the click-to-seek callback factory (Controller layer).
"""

from __future__ import annotations

from types import ModuleType

import streamlit as st

from config          import ViewerConfig
from model.app_state import ViewerState


# ── Seek callback factory ─────────────────────────────────────────────────────

def _make_seek_callback(frame_key: str, total_frames: int):
    """Return a Streamlit on_select callback that updates *frame_key*.

    The factory captures *frame_key* and *total_frames* by value so the
    returned callable is self-contained and safe to use as an ``on_select``
    argument (Streamlit calls it before the next script re-run).
    """
    def _callback() -> None:
        raw = st.session_state.get("pose_chart")
        if not raw:
            return
        # Streamlit may deliver the event as a plain dict or as a proxy object.
        try:
            pts = raw["selection"]["points"]
        except (TypeError, KeyError):
            try:
                pts = raw.selection.points
            except AttributeError:
                return
        if not pts:
            return
        pt    = pts[0]
        raw_x = pt.get("x") if isinstance(pt, dict) else getattr(pt, "x", None)
        if raw_x is not None:
            st.session_state[frame_key] = max(
                0, min(total_frames - 1, int(round(float(raw_x))))
            )

    return _callback


def _label_signature(labels: list[dict] | None) -> tuple[tuple[int | None, int | None, int | None], ...]:
    if not labels:
        return ()

    signature: list[tuple[int | None, int | None, int | None]] = []
    for label in labels:
        step_id = label.get("step_id")
        start_frame = label.get("start_frame", label.get("frame"))
        end_frame = label.get("end_frame")
        signature.append((
            int(step_id) if step_id is not None else None,
            int(round(float(start_frame))) if start_frame is not None else None,
            int(round(float(end_frame))) if end_frame is not None else None,
        ))
    return tuple(signature)


# ── Panel renderer ────────────────────────────────────────────────────────────

def render_right_panel(
    cfg:                ViewerConfig,
    state:              ViewerState,
    total_frames:       int,
    frame_idx:          int,
    speed_df,
    accel_df,
    angle_df,
    ratio_df,
    dist_half_df,
    dist_hip_df,
    dist_shoulder_df,
    center_speed_df,
    center_accel_df,
    step_labels,
    elan_step_labels,
    pt_stem:            str,
    supports_on_select: bool,
    plot_utils:         ModuleType,
) -> None:
    """Render the right column with the joint selector and analysis chart."""
    if supports_on_select:
        st.subheader("Pose Analysis — click a data point to seek to that frame")
    else:
        st.subheader("Pose Analysis")
        st.caption(
            f"Upgrade Streamlit to ≥ 1.31 to enable click-to-seek "
            f"(current: {st.__version__})."
        )

    # ── Joint multiselect ─────────────────────────────────────────────────
    all_joints = plot_utils.PLOT_KEYPOINT_NAMES
    selected: list[str] = st.multiselect(
        "Visible joints",
        options=all_joints,
        default=cfg.default_visible_joints,
        help=(
            "Select which joints to plot. "
            "Others are hidden but can be toggled via the chart legend."
        ),
    )
    visible_joints = selected if selected else None   # None → show all

    # ── Base figure: build once, cache in session state ───────────────────
    # The heavy work (iterating all joints × all frames) only runs when the
    # file or joint selection changes.  Frame seeks only update the vline.
    _cache_key = (
        pt_stem,
        tuple(sorted(visible_joints or [])),
        _label_signature(step_labels),
        _label_signature(elan_step_labels),
    )
    _cached    = st.session_state.get("_pose_chart_cache")

    if _cached is None or _cached["key"] != _cache_key:
        fig = plot_utils.build_plotly_pose_figure(
            speed_df=speed_df,
            accel_df=accel_df,
            angle_df=angle_df,
            ratio_df=ratio_df,
            dist_half_df=dist_half_df,
            dist_hip_df=dist_hip_df,
            dist_shoulder_df=dist_shoulder_df,
            step_labels=step_labels,
            elan_step_labels=elan_step_labels,
            title=pt_stem,
            visible_joints=visible_joints,
            center_speed_df=center_speed_df,
            center_accel_df=center_accel_df,
        )
        fig.update_layout(
            clickmode="event+select",
            # Constant uirevision tells Plotly to do in-place diff updates
            # instead of full DOM re-renders.  This preserves zoom, pan,
            # legend, and — critically — pending click-selection events so
            # on_select fires reliably even during streaming playback.
            uirevision="pose_chart",
        )
        st.session_state["_pose_chart_cache"] = {"key": _cache_key, "fig": fig}
    else:
        fig = _cached["fig"]

    # ── Frame indicator: cheap shape-only update (runs every frame) ───────
    plot_utils.set_frame_indicator(fig, frame_idx)

    # ── Render with or without click-to-seek ──────────────────────────────
    if supports_on_select:
        st.plotly_chart(
            fig,
            width="stretch",
            on_select=_make_seek_callback(state.frame_key, total_frames),
            key="pose_chart",
            selection_mode="points",
        )
    else:
        st.plotly_chart(fig, width="stretch")
