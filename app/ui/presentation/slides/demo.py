"""Compact live inspection slide."""

# ruff: noqa: E501

from __future__ import annotations

import streamlit as st

from app.ui.presentation.theme import RESULTS


def _open_dashboard() -> None:
    st.session_state["application_mode"] = "Technical Dashboard"


def _select_category(category: str) -> None:
    st.session_state["presentation_demo_category"] = category


def render() -> None:
    """Render a bounded, interactive inspection surface inside the slide stage."""
    st.markdown(
        '<div class="demo-marker"><div class="def-eyebrow">08 · Live demo</div>'
        '<div style="font:790 3.2cqw/.98 Inter,sans-serif;letter-spacing:-.045em">Live inspection</div></div>',
        unsafe_allow_html=True,
    )
    controls, visual = st.columns([0.36, 0.64], gap="large")
    categories = sorted(path.name for path in (RESULTS / "patchcore").iterdir() if path.is_dir())
    if st.session_state.get("presentation_demo_category") not in categories:
        st.session_state["presentation_demo_category"] = "bottle"
    category = st.session_state["presentation_demo_category"]
    with controls:
        st.markdown("**Component category**")
        with st.container():
            for row_start in range(0, len(categories), 3):
                columns = st.columns(3, gap="small")
                for column, option in zip(columns, categories[row_start : row_start + 3], strict=True):
                    column.button(
                        option.replace("_", " ").title(),
                        key=f"presentation_demo_category_button_{option}",
                        type="primary" if option == category else "secondary",
                        on_click=_select_category,
                        args=(option,),
                        width="stretch",
                    )
        panel_dir = RESULTS / "patchcore" / category / "four_panel"
        samples = sorted(panel_dir.glob("slide_09_*_sample_*.png"))
        st.markdown(
            "**Two inspection examples**  \nEach row reads left to right: source image, dataset mask, anomaly evidence, and the thresholded decision."
        )
        st.button("Open technical dashboard", on_click=_open_dashboard, width="stretch")
    with visual:
        for sample_number, sample in enumerate(samples, start=1):
            defect = sample.stem.split("_sample_", maxsplit=1)[-1].split("_", maxsplit=1)[-1].replace("_", " ").title()
            st.image(str(sample), caption=f"Sample {sample_number} · {defect}", width="stretch")
