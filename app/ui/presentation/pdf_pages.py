"""Render individual pages of the compiled Beamer deck in Streamlit."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import streamlit as st

PDF_PATH = Path(__file__).resolve().parents[3] / "docs" / "latex" / "latex_beamer_presentation" / "main.pdf"
RENDER_WIDTH = 3840


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


@st.cache_data(show_spinner=False, max_entries=4)
def pdf_page_count(path: str, modified_ns: int, size: int) -> int:
    """Read the number of PDF pages; file metadata invalidates the cache."""
    del modified_ns, size
    output = _run_poppler(["pdfinfo", path]).decode("utf-8", errors="replace")
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.partition(":")[2].strip())
    raise RuntimeError("Could not determine the number of pages in the presentation PDF.")


@st.cache_data(show_spinner=False, max_entries=18)
def pdf_page_png(path: str, modified_ns: int, size: int, page_number: int) -> bytes:
    """Rasterize only the requested page, at presentation resolution."""
    del modified_ns, size
    return _run_poppler(
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


def get_pdf_page_count() -> int:
    """Return the current compiled deck length, refreshing after a rebuild."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Presentation PDF not found: {PDF_PATH}. Compile main.tex first.")
    stat = PDF_PATH.stat()
    return pdf_page_count(str(PDF_PATH), stat.st_mtime_ns, stat.st_size)


def render_pdf_page(page_number: int) -> None:
    """Show one PDF page inside the existing fixed 16:9 presentation stage."""
    if not PDF_PATH.is_file():
        st.error(f"Presentation PDF not found: {PDF_PATH}. Compile main.tex first.")
        return

    try:
        stat = PDF_PATH.stat()
        path = str(PDF_PATH)
        png = pdf_page_png(path, stat.st_mtime_ns, stat.st_size, page_number)
    except (OSError, RuntimeError) as exc:
        st.error(str(exc))
        return

    source = base64.b64encode(png).decode("ascii")
    st.markdown(
        '<div style="width:100%;height:100%;overflow:hidden">'
        f'<img src="data:image/png;base64,{source}" alt="Presentation slide {page_number}" '
        'style="display:block;width:100%;height:100%;object-fit:contain">'
        "</div>",
        unsafe_allow_html=True,
    )
