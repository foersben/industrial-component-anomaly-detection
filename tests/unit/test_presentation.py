"""Tests for the reusable defense presentation model."""

from app.ui.presentation import SLIDES, clamp_slide_index


def test_defense_deck_has_ten_unique_slides() -> None:
    """The defense navigation should expose the complete planned deck."""
    assert len(SLIDES) == 10
    assert len({slide.title for slide in SLIDES}) == 10


def test_slide_index_is_clamped_to_deck_bounds() -> None:
    """Navigation cannot address a slide outside the deck."""
    assert clamp_slide_index(-1, len(SLIDES)) == 0
    assert clamp_slide_index(4, len(SLIDES)) == 4
    assert clamp_slide_index(10, len(SLIDES)) == 9
