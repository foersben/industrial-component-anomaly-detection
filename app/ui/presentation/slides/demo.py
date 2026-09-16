"""Compact live inspection slide."""

# ruff: noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from app.ui.presentation.theme import RESULTS

if TYPE_CHECKING:
    from pathlib import Path


def _available_ids(directory: Path, suffix: str) -> list[str]:
    return sorted(path.name.removeprefix("image_").removesuffix(suffix) for path in directory.glob(f"image_*{suffix}"))


def _open_dashboard() -> None:
    st.session_state["application_mode"] = "Technical Dashboard"


def render() -> None:
    """Render a bounded, interactive inspection surface inside the slide stage."""
    st.markdown(
        '<div class="demo-marker"><div class="def-eyebrow">08 · Live demo</div>'
        '<div style="font:790 3.2cqw/.98 Inter,sans-serif;letter-spacing:-.045em">Live inspection</div></div>',
        unsafe_allow_html=True,
    )
    controls, visual = st.columns([0.3, 0.7], gap="large")
    categories = sorted(path.name for path in (RESULTS / "patchcore").iterdir() if path.is_dir())
    with controls:
        category = st.selectbox("Component category", categories, index=categories.index("bottle"))
        pred_dir = RESULTS / "patchcore" / category / "heatmaps" / "prediction"
        gt_dir = RESULTS / "patchcore" / category / "heatmaps" / "ground_truth_overlay"
        sample_ids = sorted(set(_available_ids(pred_dir, "_prediction.png")) & set(_available_ids(gt_dir, "_gt_overlay.png")), key=int)
        sample = st.select_slider("Inspection sample", options=sample_ids, value=sample_ids[0])
        st.markdown(
            "**Operator reading**  \nWarm regions carry anomaly evidence. The contour shows the frozen decision boundary."
        )
        st.button("Open technical dashboard", on_click=_open_dashboard, width="stretch")
    with visual:
        left, right = st.columns(2, gap="medium")
        left.image(str(gt_dir / f"image_{sample}_gt_overlay.png"), caption="Ground-truth reference", width="stretch")
        right.image(str(pred_dir / f"image_{sample}_prediction.png"), caption="PatchCore prediction", width="stretch")
