"""
viewer/controller/overlay_page.py
----------------------------------
Video Overlay page controller (Controller layer).

Renders the current video frame with the skeleton pose drawn directly on top.

If ``raw_landmarks`` are missing from the .pt file (extracted before the
extractor update), offers a one-click extraction button that re-runs YOLO on
the video, patches the .pt file in-place, then reloads — so the step is only
needed once per file.

The frame slider shares ``state.frame`` with the Analysis tab so both views
are always synchronised.  Playback controls live in the Analysis tab; this
page is a passive view that updates on every stream-loop rerun.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from config              import ViewerConfig
from model.app_state     import ViewerState
from model.pose_data     import extract_and_save_raw_landmarks
from view.overlay        import draw_pose_overlay
from view.video          import grab_video_frame

_EXTRACTING_KEY = "_ov_extracting"


def render_overlay_page(
    cfg:              ViewerConfig,
    state:            ViewerState,
    raw_landmarks_np: np.ndarray | None,   # (T, 17, 2) xyn  or  None
    total_frames:     int,
    fps_meta:         float | None,
    video_path:       str,
    video_ok:         bool,
    pt_path:          str = "",            # absolute path to .pt file (needed for patching)
) -> None:
    """Render the video overlay page inside the active Streamlit tab."""

    # ── Prerequisite checks ───────────────────────────────────────────────
    if not video_ok:
        st.warning("Video file not accessible — cannot render overlay.")
        if video_path:
            st.caption(f"Expected at: `{video_path}`")
        return

    if raw_landmarks_np is None:
        _render_extraction_ui(cfg, video_path, pt_path)
        return

    # ── Frame seek slider (shared state with Analysis tab) ────────────────
    frame_idx: int = st.slider(
        "Frame",
        min_value=0,
        max_value=total_frames - 1,
        value=state.frame,
        key="ov_frame_slider",
        help="Drag to seek. Synchronised with the Analysis tab frame slider.",
    )
    state.frame = frame_idx

    if fps_meta:
        st.caption(f"t = {frame_idx / fps_meta:.3f} s  (@ {fps_meta:.1f} fps)")

    # ── Grab video frame + draw overlay ──────────────────────────────────
    frame_rgb = grab_video_frame(video_path, frame_idx)
    if frame_rgb is None:
        st.error(f"Could not read frame {frame_idx} from video.")
        return

    overlaid = draw_pose_overlay(
        frame_rgb=frame_rgb,
        kpts_xyn=raw_landmarks_np[frame_idx],
        seg_colours=cfg.seg_colours,
        joint_colour=cfg.joint_colour,
        body_center_cfg=cfg.body_center_cfg,
    )

    st.image(overlaid, caption=f"Frame {frame_idx} — skeleton overlay", width="stretch")


# ── private helper ─────────────────────────────────────────────────────────

def _render_extraction_ui(
    cfg:        ViewerConfig,
    video_path: str,
    pt_path:    str,
) -> None:
    """Prompt the user to extract raw keypoints, or run the extraction."""
    extracting: bool = st.session_state.get(_EXTRACTING_KEY, False)

    if not extracting:
        st.warning(
            "This .pt file was extracted before raw keypoints were added. "
            "Click the button below to run YOLO on the video once — the result "
            "is saved back into the .pt file so this step is only needed once."
        )
        if pt_path:
            if st.button("🔍 Extract raw keypoints", type="primary"):
                st.session_state[_EXTRACTING_KEY] = True
                st.rerun()
        else:
            st.caption("(pt_path not available — cannot extract)")
        return

    # ── Extraction in progress ────────────────────────────────────────────
    st.info("Running YOLO inference on every video frame … this runs once and saves to the .pt file.")
    progress_bar = st.progress(0, text="Starting YOLO inference …")

    def _on_progress(current: int, total: int) -> None:
        pct = current / max(total, 1)
        progress_bar.progress(pct, text=f"Frame {current} / {total}")

    raw_np = extract_and_save_raw_landmarks(
        video_path=video_path,
        pt_path=pt_path,
        yolo_model_path=str(cfg.yolo_model_path),
        on_progress=_on_progress,
    )

    st.session_state[_EXTRACTING_KEY] = False

    if raw_np is None:
        st.error(
            "Extraction failed. Check that the YOLO model and video file are "
            "accessible, then try again."
        )
        return

    progress_bar.progress(1.0, text=f"Done — {len(raw_np)} frames extracted.")
    st.success("Raw keypoints saved to .pt file. Reloading …")
    st.rerun()

