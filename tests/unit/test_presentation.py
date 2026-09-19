from typing import Any

import pytest

from app.ui.presentation import SLIDES, clamp_slide_index
from app.ui.presentation.slides import pdf_parity
from app.ui.presentation.theme import math_html


def test_defense_deck_matches_eighteen_page_pdf() -> None:
    """The defense navigation should mirror every rendered page in the PDF deck."""
    assert len(SLIDES) == 18
    assert len({slide.title for slide in SLIDES}) == 18
    for slide in SLIDES:
        assert callable(slide.render)


def test_slide_index_is_clamped_to_deck_bounds() -> None:
    """Navigation cannot address a slide outside the deck."""
    assert clamp_slide_index(-1, len(SLIDES)) == 0
    assert clamp_slide_index(4, len(SLIDES)) == 4
    assert clamp_slide_index(18, len(SLIDES)) == 17


def test_all_slides_render_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each slide's render() callable should execute safely."""
    import streamlit as st

    def _dummy(*_args: Any, **_kwargs: Any) -> None:
        pass

    def _dummy_button(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def _dummy_columns(spec: Any, **_kwargs: Any) -> list[Any]:
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [st.container() for _ in range(count)]

    monkeypatch.setattr(st, "markdown", _dummy)
    monkeypatch.setattr(st, "button", _dummy_button)
    monkeypatch.setattr(st, "image", _dummy)
    monkeypatch.setattr(st, "columns", _dummy_columns)

    for slide in SLIDES:
        slide.render()


def test_slide_math_is_typeset_as_mathml() -> None:
    """LaTeX expressions should render inside fixed HTML slides."""
    rendered = math_html(r"t^*=\frac{c_{\mathrm{FP}}}{c_{\mathrm{FP}}+c_{\mathrm{FN}}}")
    assert "<math" in rendered
    assert "<mfrac" in rendered
    assert "\\frac" not in rendered.split("</span>")[0].split('">', 1)[-1]


def test_pipeline_has_connected_branch_and_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CAE pipeline should retain its editable nodes and connector graph."""
    markup: list[str] = []
    monkeypatch.setattr("streamlit.markdown", lambda body, **_kwargs: markup.append(body))
    pdf_parity.render_cae_pipeline()
    assert len(markup) == 1
    assert markup[0].count('class="pdf-pipeline-node') == 11
    assert 'd="M690 235 H710 V85 H730"' in markup[0]
    assert 'd="M985 615 H960"' in markup[0]
