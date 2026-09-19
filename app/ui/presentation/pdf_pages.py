"""Render and persist pages of the compiled Beamer deck for Streamlit."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = ROOT / "docs" / "latex" / "latex_beamer_presentation" / "main.pdf"
CACHE_ROOT = ROOT / "app" / "ui" / "static" / "presentation_pages"
RENDER_WIDTH = 3840
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slide-prefetch")
_PREFETCH_LOCK = Lock()
_PREFETCH_JOBS: dict[Path, Future[bytes]] = {}


def _run_poppler(command: list[str]) -> bytes:
    """Run a Poppler command with an actionable error if it is unavailable."""
    try:
        return subprocess.run(command, check=True, capture_output=True).stdout
    except FileNotFoundError as exc:
        raise RuntimeError(
            "PDF slide rendering requires Poppler (pdfinfo and pdftoppm). Install poppler-utils."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not read the presentation PDF: {detail}") from exc


def _pdf_version() -> str:
    """Use PDF contents to invalidate cached pages after any deck change."""
    return hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()[:20]


@st.cache_data(show_spinner=False, max_entries=4)
def pdf_page_count(path: str, version: str) -> int:
    """Read the page count, cached for this exact PDF version."""
    del version
    output = _run_poppler(["pdfinfo", path]).decode("utf-8", errors="replace")
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.partition(":")[2].strip())
    raise RuntimeError("Could not determine the number of pages in the presentation PDF.")


def _cache_path(version: str, page_number: int) -> Path:
    return CACHE_ROOT / f"{version}-{RENDER_WIDTH}" / f"page-{page_number:03d}.png"


def _page_url(version: str, page_number: int) -> str:
    return f"app/static/presentation_pages/{version}-{RENDER_WIDTH}/page-{page_number:03d}.png"


def pdf_page_png(path: str, version: str, page_number: int) -> bytes:
    """Read a rendered page from disk or render and save it atomically."""
    cached = _cache_path(version, page_number)
    if cached.is_file():
        return cached.read_bytes()

    png = _run_poppler(
        [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-scale-to-x",
            str(RENDER_WIDTH),
            "-scale-to-y",
            "-1",
            "-png",
            path,
        ]
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cached.parent, suffix=".tmp", delete=False) as temporary:
        temporary.write(png)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(cached)
    finally:
        temporary_path.unlink(missing_ok=True)
    return png


def get_pdf_page_count() -> int:
    """Return the compiled deck length, refreshing after a PDF change."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Presentation PDF not found: {PDF_PATH}. Compile main.tex first.")
    return pdf_page_count(str(PDF_PATH), _pdf_version())


def _precache_deck(path: str, version: str, current_page: int, page_count: int) -> None:
    """Render the rest of the deck in the background, with nearby pages first."""
    order = sorted(range(1, page_count + 1), key=lambda page: (abs(page - current_page), page < current_page))
    with _PREFETCH_LOCK:
        for page in order:
            cached = _cache_path(version, page)
            if cached.is_file():
                continue
            job = _PREFETCH_JOBS.get(cached)
            if job is None or (job.done() and job.exception() is not None):
                _PREFETCH_JOBS[cached] = _PREFETCH_POOL.submit(pdf_page_png, path, version, page)


def render_pdf_page(page_number: int, page_count: int | None = None) -> None:
    """Show one PDF page inside the existing fixed 16:9 presentation stage."""
    if not PDF_PATH.is_file():
        st.error(f"Presentation PDF not found: {PDF_PATH}. Compile main.tex first.")
        return

    try:
        version = _pdf_version()
        path = str(PDF_PATH)
        pdf_page_png(path, version, page_number)
    except (OSError, RuntimeError) as exc:
        st.error(str(exc))
        return

    st.markdown(
        '<div style="width:100%;height:100%;overflow:hidden">'
        f'<img src="{_page_url(version, page_number)}" alt="Presentation slide {page_number}" '
        f'data-page-number="{page_number}" data-page-count="{page_count or page_number}" '
        f'data-pdf-version="{version}" '
        'style="display:block;width:100%;height:100%;object-fit:contain">'
        "</div>",
        unsafe_allow_html=True,
    )
    if page_count is not None:
        _precache_deck(path, version, page_number, page_count)
