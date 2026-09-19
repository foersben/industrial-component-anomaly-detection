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
from latex2mathml.converter import convert as latex_to_mathml

ASSETS = Path(__file__).resolve().parent / "assets"
DEFECT_EXAMPLES = ASSETS / "defect_examples"
GENERATED_CHARTS = ASSETS / "generated_charts"
RESULTS = ASSETS / "demo_results"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PDF_FIGURES = REPOSITORY_ROOT / "docs" / "latex" / "latex_beamer_presentation" / "figures"


@lru_cache(maxsize=64)
def image_uri(path: Path) -> str:
    """Return a local image as a browser-safe data URI."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def img(path: Path, alt: str, css_class: str = "") -> str:
    """Build escaped presentation image markup."""
    return f'<img class="{html.escape(css_class)}" src="{image_uri(path)}" alt="{html.escape(alt)}">'


@lru_cache(maxsize=128)
def math_html(latex: str, *, display: bool = False) -> str:
    """Render trusted LaTeX as native MathML inside a fixed HTML slide."""
    kind = "block" if display else "inline"
    content = latex_to_mathml(latex, display=kind)
    return f'<span class="pdf-math {kind}" aria-label="{html.escape(latex)}">{content}</span>'


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


def beamer_slide_html(title: str, body: str, *, frame_number: int, extra_class: str = "") -> None:
    """Render a content slide using the visual contract of the LaTeX Beamer deck."""
    st.markdown(
        f"""
        <div class="pdf-slide {extra_class}">
          <div class="pdf-frame-title">{title}</div>
          <div class="pdf-frame-body">{body}</div>
          <div class="pdf-frame-number">{frame_number}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def beamer_section_html(title: str) -> None:
    """Render a non-numbered section divider matching the PDF deck."""
    st.markdown(
        f"""
        <div class="pdf-section-slide">
          <div class="pdf-section-title">{title}</div>
          <div class="pdf-section-rule"></div>
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
.badge {
  display: inline-block;
  padding: .35cqh .7cqw;
  border-radius: 999px;
  font-size: .82cqw;
  font-weight: 750;
  letter-spacing: .06em;
  text-transform: uppercase;
  background: rgba(0,127,122,.12);
  color: var(--teal);
  border: 1px solid rgba(0,127,122,.28);
}
.badge-coral {
  background: rgba(240,108,84,.14);
  color: var(--coral);
  border-color: rgba(240,108,84,.32);
}
.badge-dark {
  background: rgba(16,33,43,.85);
  color: #edf6f4;
  border-color: rgba(255,255,255,.2);
}
.card-box {
  background: rgba(255,255,255,.65);
  border: 1px solid rgba(16,33,43,.08);
  border-radius: 1.1cqw;
  padding: 2.2cqh 1.5cqw;
  box-shadow: 0 .6cqw 1.8cqw rgba(16,33,43,.04);
}
.card-box-dark {
  background: #0d232c;
  color: #eef7f6;
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 1.1cqw;
  padding: 2.2cqh 1.5cqw;
  box-shadow: 0 .8cqw 2.4cqw rgba(0,0,0,.25);
}
.slide-grid-3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2cqw;
  height: 100%;
  min-height: 0;
}
.pdf-slide,
.pdf-section-slide {
  position: relative;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  overflow: hidden;
  color: #1a1a2e;
  background: #ffffff;
  font-family: "Segoe UI", Arial, sans-serif;
}
.pdf-frame-title {
  box-sizing: border-box;
  height: 11.2cqh;
  padding: 2.35cqh 3.8cqw 1.7cqh;
  color: #ffffff;
  background: #007f7a;
  font-size: 2.05cqw;
  line-height: 1;
  font-weight: 700;
}
.pdf-frame-body {
  box-sizing: border-box;
  height: calc(100% - 11.2cqh);
  padding: 4.2cqh 4.2cqw 4.7cqh;
  font-size: 1.58cqw;
  line-height: 1.36;
}
.pdf-frame-body h2,
.pdf-frame-body h3,
.pdf-frame-body p { margin: 0; }
.pdf-frame-body h3 { font-size: 1.62cqw !important; }
.pdf-frame-body ul { margin: .7cqh 0 0; padding-left: 1.45cqw; }
.pdf-frame-body li { margin: .9cqh 0; }
.pdf-frame-body li::marker { color: #007f7a; }
.pdf-frame-number {
  position: absolute;
  right: 2.1cqw;
  bottom: 1.55cqh;
  color: #6b7280;
  font-size: .74cqw;
}
.pdf-section-slide {
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 15.5cqw;
}
.pdf-section-title {
  font-size: 3.05cqw;
  line-height: 1.15;
  font-weight: 650;
  color: #26343a;
}
.pdf-section-rule {
  width: 31cqw;
  height: .32cqh;
  margin-top: 2.2cqh;
  background: #f06c54;
}
.pdf-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 3.4cqw; height: 100%; min-height: 0; }
.pdf-slide.balanced .pdf-frame-body { display: flex; align-items: center; }
.pdf-slide.balanced .pdf-frame-body > .pdf-grid-2 { width: 100%; height: auto; }
.pdf-pipeline-layout { display: grid; grid-template-rows: minmax(0,1fr) auto; gap: 2.1cqh; height: 100%; min-height: 0; }
.pdf-pipeline-map { position: relative; box-sizing: border-box; width: 87cqw; height: 59cqh; margin: 0 auto; }
.pdf-pipeline-lines { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; fill: none; }
.pdf-pipeline-lines > path { stroke: #258f90; stroke-width: 2.5; vector-effect: non-scaling-stroke; }
.pdf-pipeline-lines > path.arrow { marker-end: url(#pdf-pipeline-arrow); }
.pdf-pipeline-lines marker path { fill: #258f90; stroke: none; }
.pdf-pipeline-node { position: absolute; box-sizing: border-box; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .3cqh; height: 17%; border: 1.5px solid #008e89; border-radius: .55cqw; background: #f4fcfc; color: #26343a; text-align: center; line-height: 1.14; }
.pdf-pipeline-node strong { font-size: 1.26cqw; }
.pdf-pipeline-node span { font-size: 1.16cqw; white-space: nowrap; }
.pdf-pipeline-node.tint { background: #e4f5f5; }
.pdf-pipeline-node.strong { background: #d7eeee; }
.pdf-pipeline-node.accent { border-color: #f06c54; background: #fffaf8; }
.pdf-pipeline-node.accent-tint { border-color: #f06c54; background: #fff1ed; }
.pdf-pipeline-node.data { left: 2%; top: 15%; width: 19%; }
.pdf-pipeline-node.prep { left: 25%; top: 15%; width: 23%; }
.pdf-pipeline-node.category { left: 52%; top: 15%; width: 17%; }
.pdf-pipeline-node.texture { left: 73%; top: 0; width: 23%; }
.pdf-pipeline-node.object { left: 73%; top: 29%; width: 23%; }
.pdf-pipeline-node.masking { left: 73%; top: 53%; width: 23%; }
.pdf-pipeline-node.build { left: 37%; top: 53%; width: 24%; }
.pdf-pipeline-node.train { left: 2%; top: 53%; width: 23%; }
.pdf-pipeline-node.score { left: 2%; top: 82%; width: 23%; }
.pdf-pipeline-node.threshold { left: 37%; top: 82%; width: 24%; }
.pdf-pipeline-node.evaluation { left: 73%; top: 82%; width: 23%; }
.pdf-pipeline-note { padding: 0 1.2cqw; font-size: 1.48cqw; line-height: 1.32; }
.pdf-pipeline-note b:first-child { color: #007f7a; font-size: 1.66cqw; }
.pdf-pipeline-note p { margin-top: .7cqh; }
.pdf-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.4cqw; }
.pdf-kicker { color: #007f7a; font-weight: 750; }
.pdf-accent { color: #f06c54; }
.pdf-muted { color: #6b7280; }
.pdf-small { font-size: 1.27cqw; line-height: 1.34; }
.pdf-tiny { font-size: 1.03cqw; line-height: 1.3; }
.pdf-label { color: #6b7280; font-size: 1.04cqw; font-weight: 750; letter-spacing: .04em; text-transform: uppercase; }
.pdf-box {
  box-sizing: border-box;
  border: 1px solid rgba(0,127,122,.55);
  border-radius: .45cqw;
  background: rgba(0,127,122,.07);
  padding: 1.05cqh .9cqw;
}
.pdf-box.coral { border-color: rgba(240,108,84,.6); background: rgba(240,108,84,.09); }
.pdf-box.neutral { border-color: #aeb5b8; background: #f7f6f3; }
.pdf-image { width: 100%; height: 100%; object-fit: contain; display: block; }
.pdf-math { white-space: nowrap; font-family: "STIX Two Math", "Cambria Math", serif; }
.pdf-math.inline { display: inline-block; vertical-align: -.13em; }
.pdf-math.block { display: block; margin: 1.4cqh 0; font-size: 1.45cqw; }
.pdf-math math { font-size: inherit; }
.pdf-table { width: 100%; border-collapse: collapse; font-size: 1.27cqw; }
.pdf-table th { text-align: left; border-bottom: 2px solid #1a1a2e; padding: .55cqh .4cqw; }
.pdf-table td { border-bottom: 1px solid #c9cdcf; padding: .58cqh .4cqw; }
.pdf-flow { display: flex; align-items: center; justify-content: center; gap: .65cqw; font-size: 1.17cqw; }
.pdf-arrow { color: #007f7a; font-size: 1.55cqw; font-weight: 800; }
.pdf-comparison-row > div { height: 100%; min-height: 0; overflow: hidden; }
.pdf-comparison-row .pdf-image { height: 100% !important; max-height: 100%; object-fit: contain; }
.pdf-reading-column { font-size: 1.36cqw; line-height: 1.42; }
.pdf-reading-column li { margin: 1.1cqh 0; }
.pdf-taxonomy-layout { display: grid; grid-template-columns: 1.05fr .85fr; gap: 8.1cqw; height: 100%; min-height: 0; }
.pdf-taxonomy-copy { font-size: 1.68cqw; line-height: 1.38; }
.pdf-taxonomy-copy h2 { font-size: 1.86cqw; line-height: 1.25; margin-bottom: 2.8cqh; }
.pdf-taxonomy-card { padding: 1.8cqh 1.35cqw; border: 1px solid #007f7a; border-radius: .7cqw; background: #f1fbfb; }
.pdf-taxonomy-card + .pdf-taxonomy-card { margin-top: 2.4cqh; }
.pdf-taxonomy-card > div { white-space: nowrap; }
.pdf-taxonomy-card b { color: #007f7a; font-size: 1.82cqw; }
.pdf-taxonomy-card ul { margin: 1.2cqh 0 0; padding-left: 1.7cqw; }
.pdf-taxonomy-card li { margin: .62cqh 0; }
.pdf-taxonomy-card li::marker { color: #26343a; }
.pdf-taxonomy-card.coral { border-color: #f06c54; background: #fff7f5; }
.pdf-taxonomy-card.coral b, .pdf-taxonomy-card.coral li::marker { color: #f06c54; }
.pdf-taxonomy-copy p { margin-top: 3.7cqh; }
.pdf-taxonomy-gallery { display: flex; flex-direction: column; gap: 3.4cqh; padding-top: 3.5cqh; }
.pdf-taxonomy-group-title { color: #007f7a; font-size: 1.13cqw; font-weight: 750; letter-spacing: .025em; text-transform: uppercase; white-space: nowrap; margin-bottom: 1.9cqh; }
.pdf-taxonomy-group-title span { font-size: .96cqw; letter-spacing: 0; text-transform: none; }
.pdf-taxonomy-group.coral .pdf-taxonomy-group-title { color: #f06c54; }
.pdf-taxonomy-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 1.25cqw; }
.pdf-taxonomy-figure { min-width: 0; margin: 0; text-align: center; }
.pdf-taxonomy-figure .pdf-image { height: 19cqh; object-fit: contain; }
.pdf-taxonomy-figure figcaption { color: #6b7280; font-size: 1.04cqw; margin-top: 1.7cqh; white-space: nowrap; }
.pdf-summary-layout { display: grid; grid-template-rows: minmax(0,1fr) auto; height: 100%; }
.pdf-summary-close { display: flex; flex-direction: column; align-items: center; gap: .5cqh; border-top: 1px solid #cbd8d8; padding: 2.5cqh 0 1.5cqh; color: #007f7a; }
.pdf-summary-close b { font-size: 2.15cqw; line-height: 1.1; }
.pdf-summary-close span { color: #26343a; font-size: 1.05cqw; }
.pdf-summary-close small { color: #6b7280; font-size: .88cqw; margin-top: .3cqh; }
.pdf-title-slide {
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  padding: 6cqh 5cqw;
  color: #1a1a2e;
  background: #ffffff;
  font-family: "Segoe UI", Arial, sans-serif;
}
.pdf-title-grid { display: grid; grid-template-columns: 1.12fr .88fr; gap: 4.5cqw; height: 100%; align-items: start; }
.pdf-title-copy { padding-top: 4.2cqh; }
.pdf-title-grid h1 { margin: 0; font-size: 3cqw; line-height: 1.11; letter-spacing: -.028em; }
.pdf-title-subhead { color: #f06c54; font-size: 1.62cqw; margin-top: 2.1cqh; }
.pdf-title-grid .title-rule { width: 20cqw; height: .42cqh; margin: 6.2cqh 0 4.3cqh; background: #26343a; }
.pdf-title-grid .subtitle { font-size: 1.51cqw; line-height: 1.48; max-width: 48cqw; }
.pdf-title-pills { margin-top: 2.7cqh; }
.pdf-pill { display: inline-block; margin: 0 .5cqw .6cqh 0; padding: .53cqh .66cqw; border-radius: .35cqw; color: #007f7a; background: rgba(0,127,122,.12); font-size: .96cqw; font-weight: 700; }
.pdf-title-team { margin-top: 6.3cqh; }
.pdf-title-team > div:last-child { color: #6b7280; font-size: 1.3cqw; margin-top: 2.5cqh; }
.pdf-examples { padding-top: 1.1cqh; }
.pdf-examples-heading { display: flex; justify-content: space-between; align-items: baseline; gap: 1cqw; margin-bottom: 3.2cqh; font-size: 1.05cqw; font-weight: 700; letter-spacing: .045em; color: #6b7280; }
.pdf-examples-heading span:last-child { color: #007f7a; font-size: .95cqw; letter-spacing: 0; white-space: nowrap; }
.pdf-examples-columns, .pdf-examples-row { display: grid; grid-template-columns: 1.15fr .75fr .75fr; gap: 1cqw; align-items: center; }
.pdf-examples-columns { color: #6b7280; font-size: .86cqw; font-weight: 750; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 1.4cqh; }
.pdf-examples-row { margin-bottom: 1.25cqh; }
.pdf-example-name { display: flex; align-items: baseline; gap: 1.15cqw; white-space: nowrap; font-size: 1.38cqw; }
.pdf-example-name span { color: #6b7280; font-size: 1.01cqw; }
.pdf-example-image { position: relative; height: 16.1cqh; min-width: 0; }
.pdf-example-image .pdf-image { object-fit: cover; aspect-ratio: 1; margin: auto; }
.pdf-example-badge { position: absolute; left: 0; bottom: 0; background: #007f7a; color: #fff; font-size: .82cqw; font-weight: 700; padding: .15cqh .36cqw; line-height: 1.2; }
.pdf-example-badge.defect { background: #f06c54; }
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
