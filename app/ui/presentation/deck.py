"""Navigation and fixed-stage orchestration for the defense deck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from app.ui.presentation.slides import (
    conclusion,
    demo,
    eda,
    models,
    patchcore,
    problem,
    protocol,
    results,
    thresholds,
    title,
)
from app.ui.presentation.theme import inject_theme

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Slide:
    """One ordered defense slide."""

    title: str
    render: Callable[[], None]


SLIDES = (
    Slide("Industrial Component Anomaly Detection", title.render),
    Slide("Why the problem is hard", problem.render),
    Slide("What EDA taught us", eda.render),
    Slide("Evaluation without leakage", protocol.render),
    Slide("Model evolution", models.render),
    Slide("How PatchCore works", patchcore.render),
    Slide("Why PatchCore won", results.render),
    Slide("Threshold as a business decision", thresholds.render),
    Slide("Live inspection", demo.render),
    Slide("Deployment recommendation", conclusion.render),
)


def clamp_slide_index(index: int, slide_count: int) -> int:
    """Keep a requested slide index inside deck bounds."""
    return max(0, min(index, max(0, slide_count - 1)))


def _go_to_slide(index: int) -> None:
    st.session_state["defense_slide_index"] = clamp_slide_index(index, len(SLIDES))


def render_defense_presentation() -> None:
    """Render exactly one fixed-size slide and a stable navigation row."""
    inject_theme()
    st.session_state.setdefault("defense_slide_index", 0)
    index = clamp_slide_index(int(st.session_state["defense_slide_index"]), len(SLIDES))
    st.session_state["defense_slide_index"] = index

    with st.container(key="defense_stage"):
        SLIDES[index].render()

    with st.container(key="defense_navigation"):
        previous, progress, next_col, dashboard = st.columns(
            [1, 3.2, 1, 0.4]
        )
        previous.button(
            "◀ Previous",
            disabled=index == 0,
            width="stretch",
            on_click=_go_to_slide,
            args=(index - 1,),
        )
        progress.markdown(
            f'<div class="def-progress">{index + 1} / {len(SLIDES)} &nbsp;·&nbsp; {SLIDES[index].title}</div>',
            unsafe_allow_html=True,
        )
        next_col.button(
            "Next ▶",
            type="primary",
            disabled=index == len(SLIDES) - 1,
            width="stretch",
            on_click=_go_to_slide,
            args=(index + 1,),
        )
        dashboard.button(
            "⚙",
            help="Open technical dashboard",
            on_click=demo._open_dashboard,
            width="stretch",
        )
