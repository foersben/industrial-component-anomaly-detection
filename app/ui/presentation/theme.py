"""Styling for PDF pages, navigation, and the interactive demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

RESULTS = Path(__file__).resolve().parent / "assets" / "demo_results"

PRESENTATION_CSS = """
<style>
:root { --paper: #f5f1e8; --ink: #10212b; --teal: #007f7a; }
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
.def-eyebrow {
  color: var(--teal);
  font-size: 1.45cqw;
  line-height: 1;
  font-weight: 760;
  letter-spacing: .16em;
  text-transform: uppercase;
  margin-bottom: 1.35cqh;
}
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
[class*="st-key-presentation_demo_category_button_"] button {
  min-height: 32px !important;
  padding: 0 .3cqw !important;
  font-size: 1cqw !important;
  white-space: nowrap !important;
}
[class*="st-key-presentation_demo_category_button_"] button * {
  font-size: 1cqw !important;
  white-space: nowrap !important;
  overflow-wrap: normal !important;
  word-break: keep-all !important;
}
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
