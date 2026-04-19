"""
viewer/controller/sidebar.py
----------------------------
Sidebar controller: .pt file selector and summary metrics (Controller layer).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import ViewerConfig


def render_sidebar(cfg: ViewerConfig) -> Path:
    """Render the sidebar file-picker and return the selected .pt Path.

    Falls back to a manual text-input when the dataset root is unreachable.
    Calls ``st.stop()`` if no valid file has been chosen.
    """
    st.sidebar.header("📂 File Selection")

    pt_files: list[Path] = []
    if cfg.dataset_root.exists():
        pt_files = sorted(cfg.dataset_root.rglob("*.pt"))
        if cfg.max_files_in_picker > 0:
            pt_files = pt_files[: cfg.max_files_in_picker]

    if pt_files:
        labels         = [str(f.relative_to(cfg.dataset_root)) for f in pt_files]
        selected_label = st.sidebar.selectbox("Dataset .pt file", labels)
        return cfg.dataset_root / selected_label

    # ── fallback: manual path entry ──────────────────────────────────────
    st.sidebar.warning(f"No .pt files found under\n{cfg.dataset_root}")
    manual = st.sidebar.text_input("Enter absolute .pt path")

    if not manual:
        st.info("Please select or enter a .pt file path in the sidebar.")
        st.stop()

    pt_path = Path(manual)
    if not pt_path.exists():
        st.error(f"File not found: {pt_path}")
        st.stop()

    return pt_path


def render_metrics(metadata: dict, total_frames: int) -> None:
    """Render summary metrics in the sidebar below the file selector."""
    st.sidebar.divider()
    st.sidebar.metric("Total frames", total_frames)

    fps = metadata.get("fps")
    if fps:
        st.sidebar.metric("FPS", f"{fps:.2f}")
        st.sidebar.metric("Duration (s)", f"{total_frames / fps:.2f}")
