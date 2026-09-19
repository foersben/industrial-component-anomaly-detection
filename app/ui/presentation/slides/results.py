"""Benchmark results slide."""

# ruff: noqa: E501

from __future__ import annotations

import csv

from app.ui.presentation.theme import RESULTS, slide_html


def _category_rows() -> str:
    rows: list[tuple[str, float]] = []
    for path in sorted((RESULTS / "patchcore").glob("*/metrics_summary.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            value = float(next(csv.DictReader(handle))["image_level_f1_score"])
        rows.append((path.parent.name.replace("_", " "), value))
    rows.sort(key=lambda item: item[1], reverse=True)
    return "".join(
        f"""<div style="display:grid;grid-template-columns:7.5cqw 1fr 3.2cqw;gap:.65cqw;align-items:center">
        <span style="font-size:.73cqw;white-space:nowrap">{name}</span>
        <span style="height:.85cqh;background:#d7ddd8;border-radius:99px;overflow:hidden"><i style="display:block;width:{score * 100:.1f}%;height:100%;background:{"#007f7a" if score >= 0.9 else "#f06c54"}"></i></span>
        <b style="font-size:.73cqw">{score:.2f}</b></div>"""
        for name, score in rows
    )


def render() -> None:
    """Compare the two core models under fair-eval-v1 and show category-level variation."""
    body = f"""
    <div style="display:grid;grid-template-columns:1.05fr .95fr;gap:3.5cqw;height:100%">
      <div>
        <div class="tiny-label" style="margin-bottom:1cqh">Macro Average Comparison (15 Categories)</div>
        <div class="card-box" style="padding:1.8cqh 1.4cqw;margin-bottom:2cqh">
          <div style="display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:.5cqw;align-items:center;border-bottom:1px solid rgba(16,33,43,.1);padding-bottom:.8cqh;margin-bottom:1cqh">
            <div></div>
            <div class="tiny-label" style="color:var(--teal);font-size:.85cqw">PatchCore (Transfer)</div>
            <div class="tiny-label" style="font-size:.85cqw">Keras CAE (Scratch)</div>
          </div>
          <div style="display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:.5cqw;align-items:baseline;margin-bottom:.8cqh">
            <div style="font-size:1.12cqw;font-weight:750">Image F1 Score</div>
            <div class="metric-number accent" style="font-size:2.4cqw">0.929</div>
            <div style="font-size:1.8cqw;color:var(--muted)">0.495</div>
          </div>
          <div style="display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:.5cqw;align-items:baseline;margin-bottom:.8cqh">
            <div style="font-size:1.12cqw;font-weight:750">Image PR-AUC</div>
            <div class="metric-number accent" style="font-size:2.4cqw">0.988</div>
            <div style="font-size:1.8cqw;color:var(--muted)">0.867</div>
          </div>
          <div style="display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:.5cqw;align-items:baseline">
            <div style="font-size:1.12cqw;font-weight:750">Pixel AUPIMO</div>
            <div class="metric-number accent" style="font-size:2.4cqw">0.603</div>
            <div style="font-size:1.8cqw;color:var(--coral);font-weight:700">0.058</div>
          </div>
        </div>
        <div class="flat-note" style="font-size:1.02cqw;line-height:1.4">
          Transfer learning yields a <b>10&times; leap in AUPIMO</b> localization and raises Image F1 by <b>+43.4 points</b>, beating the trivial all-defective reference in 13/15 categories.
        </div>
      </div>
      <div>
        <div class="tiny-label" style="margin-bottom:1cqh">PatchCore Image F1 by Category (Validation-Calibrated &alpha; = 0.05)</div>
        <div style="display:grid;gap:.5cqh">{_category_rows()}</div>
      </div>
    </div>
    <div class="source">Protocol: fair-eval-v1 · Metrics from official test split · Seed 42</div>
    """
    slide_html("Empirical Benchmark: Why Transfer Learning Won", body, eyebrow="08 · Empirical Performance")
