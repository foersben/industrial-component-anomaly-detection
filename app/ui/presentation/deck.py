"""Navigation and fixed-stage orchestration for the defense deck."""

from __future__ import annotations

import streamlit as st

from app.ui.presentation.pdf_pages import get_pdf_page_count, render_pdf_page
from app.ui.presentation.slides import demo
from app.ui.presentation.theme import inject_theme


def clamp_slide_index(index: int, slide_count: int) -> int:
    """Keep a requested slide index inside deck bounds."""
    return max(0, min(index, max(0, slide_count - 1)))


def _go_to_slide(index: int, slide_count: int) -> None:
    st.session_state["defense_slide_index"] = clamp_slide_index(index, slide_count)
    st.session_state["defense_demo_open"] = False


def _set_demo_open(is_open: bool) -> None:
    st.session_state["defense_demo_open"] = is_open


def render_defense_presentation() -> None:
    """Render exactly one fixed-size slide and a stable navigation row."""
    inject_theme()
    try:
        slide_count = get_pdf_page_count()
    except (OSError, RuntimeError) as exc:
        st.error(str(exc))
        return
    if slide_count < 1:
        st.error("The presentation PDF has no pages.")
        return
    st.session_state.setdefault("defense_slide_index", 0)
    st.session_state.setdefault("defense_demo_open", False)
    index = clamp_slide_index(int(st.session_state["defense_slide_index"]), slide_count)
    st.session_state["defense_slide_index"] = index
    demo_open = bool(st.session_state["defense_demo_open"])
    progress_label = "Interactive demo" if demo_open else f"{index + 1} / {slide_count}"

    with st.container(key="defense_stage"):
        if demo_open:
            demo.render()
        else:
            render_pdf_page(index + 1)

    with st.container(key="defense_navigation"):
        previous, progress, next_col, live_demo, dashboard = st.columns([1, 2.7, 1, 0.8, 0.4])
        previous.button(
            "◀ Previous",
            disabled=index == 0 or demo_open,
            width="stretch",
            on_click=_go_to_slide,
            args=(index - 1, slide_count),
        )
        progress.markdown(
            f'<div class="def-progress">{progress_label}</div>',
            unsafe_allow_html=True,
        )
        next_col.button(
            "Next ▶",
            type="primary",
            disabled=index == slide_count - 1 or demo_open,
            width="stretch",
            on_click=_go_to_slide,
            args=(index + 1, slide_count),
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
