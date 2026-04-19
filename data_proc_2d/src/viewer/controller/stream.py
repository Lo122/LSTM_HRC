"""
viewer/controller/stream.py
----------------------------
Streaming playback loop controller (Controller layer).

Must be called AFTER all UI elements have been rendered so that the
``st.rerun()`` at the end of a tick picks up a fully-rendered page.
"""

from __future__ import annotations

import time

import streamlit as st

from config          import ViewerConfig
from model.app_state import ViewerState


def run_stream_loop(
    cfg:          ViewerConfig,
    state:        ViewerState,
    total_frames: int,
    fps_meta:     float | None,
    play_speed:   float,
) -> None:
    """Advance one frame and trigger a rerun if the player is active.

    The function is a no-op while playback is paused or stopped, so it is
    always safe to call at the bottom of the main script.

    Responsiveness
    --------------
    Instead of sleeping for the full inter-frame interval (which blocks the
    server thread and delays Pause/Stop clicks), we always sleep for at most
    ``cfg.min_sleep_s`` (20 ms).  Frame advancement is gated on wall-clock
    time: the frame only increments when the elapsed time since the last
    advance is >= the desired inter-frame interval.  This means the Pause
    button is processed within one poll cycle (~20 ms) regardless of speed.
    """
    if not state.playing:
        return

    frame_interval = 1.0 / (float(fps_meta or cfg.default_fps_fallback) * play_speed)

    now    = time.monotonic()
    last_t = st.session_state.get("_stream_last_t", 0.0)

    if now - last_t >= frame_interval:
        # Enough real time has passed — advance the frame.
        next_frame = state.frame + 1
        if next_frame >= total_frames:
            state.playing = False      # auto-stop at last frame
            return
        state.frame = next_frame
        st.session_state["_stream_last_t"] = now

    # Short fixed sleep keeps the poll loop responsive (≤ min_sleep_s lag
    # on Pause/Stop) without spinning the CPU at 100 %.
    time.sleep(cfg.min_sleep_s)
    st.rerun()
