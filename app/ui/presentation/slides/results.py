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
        <span style="height:.85cqh;background:#d7ddd8;border-radius:99px;overflow:hidden"><i style="display:block;width:{score * 100:.1f}%;height:100%;background:{'#007f7a' if score >= .9 else '#f06c54'}"></i></span>
        <b style="font-size:.73cqw">{score:.2f}</b></div>"""
        for name, score in rows
    )


def render() -> None:
    """Compare the two core models and show category-level variation."""
    body = f"""
    <div style="display:grid;grid-template-columns:1.08fr .92fr;gap:4.3cqw;height:100%">
      <div>
        <div style="display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:.5cqw;align-items:end;margin-bottom:1.5cqh">
          <div></div><div class="tiny-label" style="color:var(--teal)">PatchCore</div><div class="tiny-label">CAE</div>
          <div style="font-size:1.2cqw;font-weight:720">Image F1</div><div class="metric-number accent" style="font-size:2.8cqw">0.929</div><div style="font-size:2.15cqw;color:var(--muted)">0.495</div>
          <div style="font-size:1.2cqw;font-weight:720">PR-AUC</div><div class="metric-number accent" style="font-size:2.8cqw">0.988</div><div style="font-size:2.15cqw;color:var(--muted)">0.867</div>
          <div style="font-size:1.2cqw;font-weight:720">AUPIMO</div><div class="metric-number accent" style="font-size:2.8cqw">0.603</div><div style="font-size:2.15cqw;color:var(--muted)">0.058</div>
        </div>
        <div class="flat-note" style="margin-top:3.2cqh">PatchCore combines strong image decisions with useful low-false-positive localisation.</div>
      </div>
      <div>
        <div class="tiny-label" style="margin-bottom:1.2cqh">Image F1 by category</div>
        <div style="display:grid;gap:.55cqh">{_category_rows()}</div>
      </div>
    </div>
    <div class="source">Source: category metrics_summary.csv files · fair-eval-v1</div>
    """
    slide_html("Why PatchCore won", body, eyebrow="06 · Results")
