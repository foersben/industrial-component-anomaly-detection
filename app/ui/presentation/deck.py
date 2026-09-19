"""Navigation and fixed-stage orchestration for the defense deck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from app.ui.presentation.slides import demo, pdf_parity
from app.ui.presentation.theme import inject_theme

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Slide:
    """One ordered defense slide."""

    title: str
    render: Callable[[], None]


SLIDES = (
    Slide("Industrial Component Anomaly Detection", pdf_parity.render_title),
    Slide("Data Exploration", pdf_parity.render_data_section),
    Slide("The MVTec AD Dataset - Class Imbalance", pdf_parity.render_dataset_imbalance),
    Slide("Defect Taxonomy & Spatial Distribution", pdf_parity.render_taxonomy),
    Slide("Metric Selection: AUPIMO & F1-Score", pdf_parity.render_metrics),
    Slide("Models", pdf_parity.render_models_section),
    Slide("Model Selection & Architecture Overview", pdf_parity.render_model_overview),
    Slide("PatchCore: Transfer Learning & Architecture", pdf_parity.render_patchcore),
    Slide("Keras CAE - Autoencoder Learning Process", pdf_parity.render_cae_learning),
    Slide("Keras CAE - End-to-End Pipeline", pdf_parity.render_cae_pipeline),
    Slide("Evaluation Protocol & Benchmarks", pdf_parity.render_evaluation_section),
    Slide("Zero-Leakage Evaluation Protocol", pdf_parity.render_protocol),
    Slide("Qualitative Results - Localisation Comparison", pdf_parity.render_qualitative),
    Slide("Business Interpretation", pdf_parity.render_business_section),
    Slide("Economic Error Asymmetry - The Cost Model", pdf_parity.render_cost_model),
    Slide("Threshold Calibration & Risk-Tiered Operating Profiles", pdf_parity.render_thresholds),
    Slide("SLA Interpretation & Operational Value", pdf_parity.render_sla),
    Slide("Summary & Future Prospects", pdf_parity.render_summary),
)


def clamp_slide_index(index: int, slide_count: int) -> int:
    """Keep a requested slide index inside deck bounds."""
    return max(0, min(index, max(0, slide_count - 1)))


def _go_to_slide(index: int) -> None:
    st.session_state["defense_slide_index"] = clamp_slide_index(index, len(SLIDES))
    st.session_state["defense_demo_open"] = False


def _set_demo_open(is_open: bool) -> None:
    st.session_state["defense_demo_open"] = is_open


def render_defense_presentation() -> None:
    """Render exactly one fixed-size slide and a stable navigation row."""
    inject_theme()
    st.session_state.setdefault("defense_slide_index", 0)
    st.session_state.setdefault("defense_demo_open", False)
    index = clamp_slide_index(int(st.session_state["defense_slide_index"]), len(SLIDES))
    st.session_state["defense_slide_index"] = index
    demo_open = bool(st.session_state["defense_demo_open"])
    progress_label = "Interactive demo" if demo_open else f"{index + 1} / {len(SLIDES)} · {SLIDES[index].title}"

    with st.container(key="defense_stage"):
        if demo_open:
            demo.render()
        else:
            SLIDES[index].render()

    with st.container(key="defense_navigation"):
        previous, progress, next_col, live_demo, dashboard = st.columns([1, 2.7, 1, 0.8, 0.4])
        previous.button(
            "◀ Previous",
            disabled=index == 0 or demo_open,
            width="stretch",
            on_click=_go_to_slide,
            args=(index - 1,),
        )
        progress.markdown(
            f'<div class="def-progress">{progress_label}</div>',
            unsafe_allow_html=True,
        )
        next_col.button(
            "Next ▶",
            type="primary",
            disabled=index == len(SLIDES) - 1 or demo_open,
            width="stretch",
            on_click=_go_to_slide,
            args=(index + 1,),
        )
        live_demo.button(
            "Slides" if demo_open else "Live demo",
            on_click=_set_demo_open,
            args=(not demo_open,),
            width="stretch",
        )
        dashboard.button(
            "⚙",
            help="Open technical dashboard",
            on_click=demo._open_dashboard,
            width="stretch",
        )
