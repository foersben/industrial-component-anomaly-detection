"""Tests for the PDF-backed defense presentation."""

from pathlib import Path

import pytest

from app.ui.presentation import clamp_slide_index, deck, pdf_pages


def test_slide_index_is_clamped_to_current_pdf_length() -> None:
    """Navigation bounds come from the compiled PDF page count."""
    assert clamp_slide_index(-1, 18) == 0
    assert clamp_slide_index(4, 18) == 4
    assert clamp_slide_index(18, 18) == 17
    assert clamp_slide_index(18, 19) == 18


def test_go_to_slide_uses_dynamic_page_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a PDF page should make it reachable without defining a Slide."""
    state: dict[str, int | bool] = {}
    monkeypatch.setattr(deck.st, "session_state", state)

    deck._go_to_slide(18, 19)

    assert state == {"defense_slide_index": 18, "defense_demo_open": False}


def test_pdf_page_count_reads_pdfinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compiled PDF is the source for navigation length."""
    monkeypatch.setattr(pdf_pages, "_run_poppler", lambda _command: b"Title: Demo\nPages: 18\n")
    assert pdf_pages.pdf_page_count.__wrapped__("deck.pdf", 1, 42) == 18


def test_get_pdf_page_count_uses_current_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Metadata passed to the cache changes after a PDF rebuild."""
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-test")
    arguments: list[tuple[str, int, int]] = []

    def fake_count(path: str, modified_ns: int, size: int) -> int:
        arguments.append((path, modified_ns, size))
        return 19

    monkeypatch.setattr(pdf_pages, "PDF_PATH", pdf)
    monkeypatch.setattr(pdf_pages, "pdf_page_count", fake_count)

    assert pdf_pages.get_pdf_page_count() == 19
    assert arguments == [(str(pdf), pdf.stat().st_mtime_ns, pdf.stat().st_size)]


def test_pdf_page_png_renders_only_requested_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app should not rasterize the entire deck on every navigation step."""
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> bytes:
        commands.append(command)
        return b"png"

    monkeypatch.setattr(pdf_pages, "_run_poppler", fake_run)
    assert pdf_pages.pdf_page_png.__wrapped__("deck.pdf", 1, 42, 5) == b"png"
    assert commands == [
        [
            "pdftoppm",
            "-f",
            "5",
            "-l",
            "5",
            "-singlefile",
            "-scale-to-x",
            "3840",
            "-scale-to-y",
            "-1",
            "-png",
            "deck.pdf",
        ]
    ]


def test_pdf_page_is_embedded_in_slide_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The current PDF page should become one full-stage image."""
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-test")
    markup: list[str] = []
    monkeypatch.setattr(pdf_pages, "PDF_PATH", pdf)
    monkeypatch.setattr(pdf_pages, "pdf_page_png", lambda *_args: b"png")
    monkeypatch.setattr(pdf_pages.st, "markdown", lambda body, **_kwargs: markup.append(body))

    pdf_pages.render_pdf_page(5)

    assert len(markup) == 1
    assert 'src="data:image/png;base64,cG5n"' in markup[0]
    assert 'alt="Presentation slide 5"' in markup[0]
    assert "object-fit:contain" in markup[0]
