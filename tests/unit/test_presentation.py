"""Tests for the PDF-backed defense presentation."""

from concurrent.futures import Future
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
    assert pdf_pages.pdf_page_count.__wrapped__("deck.pdf", "version") == 18


def test_get_pdf_page_count_uses_current_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The page-count cache uses the current PDF contents."""
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-test")
    arguments: list[tuple[str, str]] = []

    def fake_count(path: str, version: str) -> int:
        arguments.append((path, version))
        return 19

    monkeypatch.setattr(pdf_pages, "PDF_PATH", pdf)
    monkeypatch.setattr(pdf_pages, "pdf_page_count", fake_count)

    assert pdf_pages.get_pdf_page_count() == 19
    assert arguments == [(str(pdf), pdf_pages._pdf_version())]


def test_pdf_page_png_is_cached_on_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A page is rasterized once and read from disk on later calls."""
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> bytes:
        commands.append(command)
        return b"png"

    monkeypatch.setattr(pdf_pages, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(pdf_pages, "_run_poppler", fake_run)
    assert pdf_pages.pdf_page_png("deck.pdf", "version", 5) == b"png"
    assert pdf_pages.pdf_page_png("deck.pdf", "version", 5) == b"png"
    assert pdf_pages._cache_path("version", 5).read_bytes() == b"png"
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


def test_pdf_change_uses_new_cache_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Even a same-size PDF edit must select a fresh page cache."""
    pdf = tmp_path / "main.pdf"
    monkeypatch.setattr(pdf_pages, "PDF_PATH", pdf)
    monkeypatch.setattr(pdf_pages, "CACHE_ROOT", tmp_path / "pages")
    pdf.write_bytes(b"version A")
    original = pdf_pages._cache_path(pdf_pages._pdf_version(), 1)
    pdf.write_bytes(b"version B")
    updated = pdf_pages._cache_path(pdf_pages._pdf_version(), 1)
    assert original != updated


def test_precache_schedules_missing_pages_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Render all missing pages in the background, prioritizing nearby ones."""
    scheduled: list[tuple[str, str, int]] = []

    class ImmediatePool:
        def submit(self, _function, *args):
            scheduled.append(args)
            job: Future[bytes] = Future()
            job.set_result(b"png")
            return job

    monkeypatch.setattr(pdf_pages, "_PREFETCH_POOL", ImmediatePool())
    monkeypatch.setattr(pdf_pages, "_PREFETCH_JOBS", {})
    monkeypatch.setattr(pdf_pages, "CACHE_ROOT", tmp_path)

    pdf_pages._cache_path("version", 2).parent.mkdir(parents=True)
    pdf_pages._cache_path("version", 2).write_bytes(b"png")
    pdf_pages._precache_deck("deck.pdf", "version", 2, 4)
    pdf_pages._precache_deck("deck.pdf", "version", 2, 4)

    assert scheduled == [("deck.pdf", "version", 3), ("deck.pdf", "version", 1), ("deck.pdf", "version", 4)]


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
