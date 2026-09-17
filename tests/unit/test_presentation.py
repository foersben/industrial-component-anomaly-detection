from typing import Any

import pytest

from app.ui.presentation import SLIDES, clamp_slide_index


def test_defense_deck_has_twelve_unique_slides() -> None:
    """The defense navigation should expose the complete planned 12-slide deck."""
    assert len(SLIDES) == 12
    assert len({slide.title for slide in SLIDES}) == 12
    for slide in SLIDES:
        assert callable(slide.render)


def test_slide_index_is_clamped_to_deck_bounds() -> None:
    """Navigation cannot address a slide outside the deck."""
    assert clamp_slide_index(-1, len(SLIDES)) == 0
    assert clamp_slide_index(4, len(SLIDES)) == 4
    assert clamp_slide_index(12, len(SLIDES)) == 11


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
