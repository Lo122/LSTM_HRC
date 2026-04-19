"""
sync_viewer.py
--------------
Streamlit app for synchronised pose-analysis visualisation.

Each .pt file contains:
    landmarks  (T, 17, 2)  – per-frame normalised keypoints
    metadata               – video path, fps, model info …
    t_steps    (T,)        – per-frame timestamps (seconds)

Layout
------
    Sidebar   : .pt file selector
    Left col  : frame slider + skeleton canvas + (optional) real video frame
    Right col : interactive Plotly chart with optional step-label rows
                clicking a data-point seeks the left panel to that frame

Usage
-----
    streamlit run application/sync_viewer.py
"""

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress the "missing ScriptRunContext" noise that PyTorch background
# threads trigger when they call into Streamlit logging.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)

import streamlit as st

# ── package path ──────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parents[1] / "src" / "viewer"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from config                      import cfg, SRC_ROOT
from model.pose_data             import load_pose_data
from model.app_state             import ViewerState
from view.video                  import resolve_video_path
from controller.sidebar          import render_sidebar, render_metrics
from controller.left_panel       import render_left_panel
from controller.right_panel      import render_right_panel
from controller.overlay_page     import render_overlay_page
from controller.stream           import run_stream_loop

# ── ensure src/ is on the path for utilities imports ─────────────────────────
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_proc_2d.src import plot_utils   # imported here so it's available to right_panel

# ── Streamlit version guard for on_select ────────────────────────────────────
_ST_VER             = tuple(int(x) for x in st.__version__.split(".")[:2])
_SUPPORTS_ON_SELECT = _ST_VER >= (1, 31)

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    layout=cfg.layout,
    page_title=cfg.page_title,
    page_icon=cfg.page_icon,
)
st.title("Pose Analysis — Synchronised Frame Viewer")

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
pt_path = render_sidebar(cfg)

# ══════════════════════════════════════════════════════════════════════════════
# Load & pre-compute (cached per unique path)
# ══════════════════════════════════════════════════════════════════════════════
(
    landmarks_np,
    metadata,
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
    raw_landmarks_np,     # (T, 17, 2) xyn in [0,1], or None for older .pt files
) = load_pose_data(str(pt_path))

T        = landmarks_np.shape[0]
fps_meta = metadata.get("fps")

render_metrics(metadata, T)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
state = ViewerState(pt_path.stem)

# ══════════════════════════════════════════════════════════════════════════════
# Stable axis limits for the skeleton
# ══════════════════════════════════════════════════════════════════════════════
_pad   = cfg.axis_pad
x_lim  = (landmarks_np[:, :, 0].min() - _pad, landmarks_np[:, :, 0].max() + _pad)
y_lim  = (landmarks_np[:, :, 1].min() - _pad, landmarks_np[:, :, 1].max() + _pad)

# ══════════════════════════════════════════════════════════════════════════════
# Video
# ══════════════════════════════════════════════════════════════════════════════
video_path, video_ok = resolve_video_path(metadata)

# ══════════════════════════════════════════════════════════════════════════════
# Main layout  (two tabs: Analysis  |  Video Overlay)
# ══════════════════════════════════════════════════════════════════════════════
tab_analysis, tab_overlay = st.tabs(["📊 Analysis", "🎥 Video Overlay"])

with tab_analysis:
    col_left, col_right = st.columns(
        [cfg.left_col_weight, cfg.right_col_weight], gap="medium"
    )

    with col_left:
        frame_idx, play_speed = render_left_panel(
            cfg=cfg,
            state=state,
            landmarks_np=landmarks_np,
            total_frames=T,
            fps_meta=fps_meta,
            x_lim=x_lim,
            y_lim=y_lim,
            video_path=video_path,
            video_ok=video_ok,
            metadata=metadata,
        )

    with col_right:
        render_right_panel(
            cfg=cfg,
            state=state,
            total_frames=T,
            frame_idx=frame_idx,
            speed_df=speed_df,
            accel_df=accel_df,
            angle_df=angle_df,
            ratio_df=ratio_df,
            dist_half_df=dist_half_df,
            dist_hip_df=dist_hip_df,
            dist_shoulder_df=dist_shoulder_df,
            center_speed_df=center_speed_df,
            center_accel_df=center_accel_df,
            step_labels=step_labels,
            elan_step_labels=elan_step_labels,
            pt_stem=pt_path.stem,
            supports_on_select=_SUPPORTS_ON_SELECT,
            plot_utils=plot_utils,
        )

with tab_overlay:
    render_overlay_page(
        cfg=cfg,
        state=state,
        raw_landmarks_np=raw_landmarks_np,
        total_frames=T,
        fps_meta=fps_meta,
        video_path=video_path,
        video_ok=video_ok,
        pt_path=str(pt_path),
    )

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    f"File: `{pt_path.name}` — {T} frames"
    + (f" · {T / fps_meta:.1f} s" if fps_meta else "")
    + "  |  Streamlit "
    + st.__version__
)

# ══════════════════════════════════════════════════════════════════════════════
# Streaming loop (runs after the full UI is rendered each tick)
# ══════════════════════════════════════════════════════════════════════════════
run_stream_loop(cfg, state, T, fps_meta, play_speed)
