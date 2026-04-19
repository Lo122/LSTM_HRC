"""
viewer/model/app_state.py
-------------------------
Session-state management for the sync viewer (Model layer).

Frame state lives in ``_frame_key``.  All external seeks (chart click,
stream loop, Stop button) write to ``_frame_key`` freely — at any point in
the script run — because the slider uses ``value=state.frame`` rather than
binding its own session-state key.  This removes the need for the two-key
dance and eliminates the slider-reset bug that two-key sync causes.
"""

from __future__ import annotations

import streamlit as st


class ViewerState:
    """Manages per-file session-state keys for frame navigation and playback."""

    def __init__(self, pt_stem: str) -> None:
        self._frame_key   = f"_fv_{pt_stem}"
        self._playing_key = f"playing_{pt_stem}"

        # Initialise defaults on first visit.
        if self._frame_key not in st.session_state:
            st.session_state[self._frame_key] = 0
        if self._playing_key not in st.session_state:
            st.session_state[self._playing_key] = False

    # ── frame ────────────────────────────────────────────────────────────────

    @property
    def frame(self) -> int:
        return int(st.session_state[self._frame_key])

    @frame.setter
    def frame(self, value: int) -> None:
        st.session_state[self._frame_key] = int(value)

    # ── playing flag ─────────────────────────────────────────────────────────

    @property
    def playing(self) -> bool:
        return bool(st.session_state[self._playing_key])

    @playing.setter
    def playing(self, value: bool) -> None:
        st.session_state[self._playing_key] = bool(value)

    # ── key accessor (read-only) ──────────────────────────────────────────────

    @property
    def frame_key(self) -> str:
        """The session-state key for the frame index (always safe to write)."""
        return self._frame_key
