"""Presentation theme and small rendering helpers."""

# Long CSS literals are intentionally kept readable as presentation markup.
# ruff: noqa: E501

from __future__ import annotations

import base64
import html
import mimetypes
from functools import lru_cache
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "raw" / "mvtec_ad"
RESULTS = REPO_ROOT / "results" / "evaluation"


@lru_cache(maxsize=64)
def image_uri(path: Path) -> str:
    """Return a local image as a browser-safe data URI."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def img(path: Path, alt: str, css_class: str = "") -> str:
    """Build escaped presentation image markup."""
    return (
        f'<img class="{html.escape(css_class)}" src="{image_uri(path)}" '
        f'alt="{html.escape(alt)}">'
    )


def slide_html(title: str, body: str, *, eyebrow: str, extra_class: str = "") -> None:
    """Render one static slide as a single, height-bounded HTML composition."""
    st.markdown(
        f"""
        <div class="def-slide {extra_class}">
          <div class="def-eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <div class="def-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PRESENTATION_CSS = """
<style>
:root {
  --paper: #f5f1e8;
  --ink: #10212b;
  --muted: #64747a;
  --teal: #007f7a;
  --mint: #8dd8c9;
  --coral: #f06c54;
  --gold: #e9b949;
  --night: #09171f;
}
html, body, #root, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main { height: 100%; overflow: hidden !important; }
[data-testid="stAppViewContainer"] {
  background: #071117;
  overscroll-behavior: none;
}
[data-testid="stHeader"], #MainMenu, footer, [data-testid="stStatusWidget"] { display: none !important; }
.block-container, [data-testid="stAppViewBlockContainer"] {
  box-sizing: border-box;
  width: 100%;
  max-width: none !important;
  height: 100dvh;
  padding: 8px 12px 0 !important;
  overflow: hidden !important;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.element-container:has(.presentation-style-marker) { display: none !important; }
.st-key-defense_stage {
  box-sizing: border-box;
  width: min(96vw, calc((100dvh - 94px) * 16 / 9));
  aspect-ratio: 16 / 9;
  align-self: center;
  flex: 0 0 auto;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 10px;
  background: var(--paper);
  box-shadow: 0 16px 55px rgba(0,0,0,.42);
  container-type: size;
}
.st-key-defense_stage > div,
.st-key-defense_stage [data-testid="stVerticalBlock"],
.st-key-defense_stage .stMarkdown,
.st-key-defense_stage [data-testid="stMarkdownContainer"] { height: 100%; min-height: 0; }
.st-key-defense_stage .stMarkdown > div,
.st-key-defense_stage .stMarkdown > div > div { height: 100%; margin: 0 !important; }
.st-key-defense_stage [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-defense_navigation {
  width: min(96vw, calc((100dvh - 94px) * 16 / 9));
  align-self: center;
  height: 54px;
  flex: 0 0 54px;
  padding-top: 8px;
  color: #dce7e9;
}
.st-key-defense_navigation [data-testid="stHorizontalBlock"] { align-items: center; gap: 12px; }
.st-key-defense_navigation button {
  min-height: 38px;
  height: 38px;
  border-radius: 999px;
  border-color: rgba(255,255,255,.24);
  background: transparent;
  color: #edf4f4;
}
.st-key-defense_navigation button[kind="primary"] { background: #008f88; border-color: #008f88; }
.def-progress { text-align: center; color: #d5e0e2; font: 650 14px/38px Inter, sans-serif; }
.def-slide {
  box-sizing: border-box;
  height: 100%;
  overflow: hidden;
  padding: 5.6cqh 5.4cqw 4.8cqh;
  color: var(--ink);
  background:
    radial-gradient(circle at 86% 8%, rgba(141,216,201,.28), transparent 28%),
    linear-gradient(135deg, #fbf8f1, #efe9dc);
  font-family: Inter, "Segoe UI", sans-serif;
}
.def-slide.dark {
  color: #eef7f6;
  background:
    radial-gradient(circle at 83% 20%, rgba(0,127,122,.42), transparent 33%),
    linear-gradient(135deg, #0b2029, #071117 70%);
}
.def-eyebrow {
  color: var(--teal);
  font-size: 1.45cqw;
  line-height: 1;
  font-weight: 760;
  letter-spacing: .16em;
  text-transform: uppercase;
  margin-bottom: 1.35cqh;
}
.dark .def-eyebrow { color: #87e4d7; }
.def-slide h1 {
  margin: 0;
  font-size: clamp(25px, 4.1cqw, 64px);
  line-height: .98;
  letter-spacing: -.045em;
  font-weight: 790;
}
.def-slide h2, .def-slide h3, .def-slide p { margin-top: 0; }
.def-body { height: calc(100% - 11.3cqh); min-height: 0; margin-top: 4cqh; }
.def-sub { color: var(--muted); font-size: 1.65cqw; line-height: 1.35; }
.dark .def-sub { color: #aec2c7; }
.accent { color: var(--teal); }
.dark .accent { color: #83e0d2; }
.coral { color: var(--coral); }
.metric-number { font-size: 4.4cqw; line-height: .9; font-weight: 820; letter-spacing: -.055em; }
.metric-label { margin-top: 1.1cqh; font-size: 1.25cqw; font-weight: 720; }
.metric-note { margin-top: .5cqh; color: var(--muted); font-size: 1.02cqw; }
.photo { display: block; width: 100%; height: 100%; object-fit: cover; }
.contain { object-fit: contain; }
.image-frame { overflow: hidden; border-radius: 1.2cqw; background: #d8ddd9; box-shadow: 0 1.2cqw 3cqw rgba(28,42,48,.18); }
.tiny-label { font-size: .92cqw; font-weight: 760; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.source { position: absolute; right: 3.4cqw; bottom: 2.2cqh; color: var(--muted); font-size: .72cqw; }
.slide-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4cqw; height: 100%; min-height: 0; }
.def-rule { width: 6cqw; height: .55cqh; border-radius: 99px; background: var(--coral); margin: 2.1cqh 0; }
.flat-note { border-left: .35cqw solid var(--coral); padding-left: 1.3cqw; font-size: 1.3cqw; line-height: 1.35; }
.stage-chip { display: inline-block; border: 1px solid currentColor; border-radius: 999px; padding: .55cqh .8cqw; font-size: .9cqw; font-weight: 720; }
.st-key-defense_stage:has(.demo-marker) {
  padding: 4.2cqh 4.3cqw 3.4cqh;
  background: linear-gradient(135deg, #f8f5ed, #ece6d9);
  color: var(--ink);
}
.st-key-defense_stage:has(.demo-marker) .stMarkdown,
.st-key-defense_stage:has(.demo-marker) [data-testid="stMarkdownContainer"] { height: auto; }
.st-key-defense_stage:has(.demo-marker) > div { height: auto; }
.st-key-defense_stage:has(.demo-marker) .stMarkdown > div,
.st-key-defense_stage:has(.demo-marker) .stMarkdown > div > div { height: auto; }
.st-key-defense_stage:has(.demo-marker) [data-testid="stVerticalBlock"] { gap: 1.2cqh; height: auto; }
.st-key-defense_stage:has(.demo-marker) [data-testid="stImage"] img {
  max-height: 38cqh;
  width: 100%;
  object-fit: contain;
  border-radius: 1cqw;
  background: #071117;
}
.st-key-defense_stage:has(.demo-marker) label { font-size: 1.03cqw; color: var(--ink) !important; }
.st-key-defense_stage:has(.demo-marker) [data-baseweb="select"] { font-size: 1.02cqw; }
.st-key-defense_stage:has(.demo-marker) button { min-height: 36px; }
.st-key-defense_stage:has(.demo-marker) [data-testid="stButton"] button,
.st-key-defense_stage:has(.demo-marker) [data-testid="stButton"] button * {
    color: #ffffff !important;
}
.st-key-defense_stage:has(.demo-marker) [data-testid="stButton"] button:hover {
    background: #008f88 !important;
    border-color: #008f88 !important;
    color: #ffffff !important;
}
@media (max-width: 800px) {
  .st-key-defense_stage { border-radius: 5px; }
  .st-key-defense_navigation button { font-size: 0; }
  .st-key-defense_navigation button::first-letter { font-size: 16px; }
}
</style>
"""


def inject_theme() -> None:
    """Apply the fixed-stage contract to the Streamlit page."""
    st.markdown(f'{PRESENTATION_CSS}<span class="presentation-style-marker"></span>', unsafe_allow_html=True)
