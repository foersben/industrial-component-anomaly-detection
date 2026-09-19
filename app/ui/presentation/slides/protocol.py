"""Evaluation protocol slide."""

# ruff: noqa: E501, RUF001

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Render the leakage-safe split, canonical resolution, and metric roles."""
    body = r"""
    <div style="display:grid;grid-template-rows:22cqh 1fr;gap:3.5cqh;height:100%">
      <div style="display:grid;grid-template-columns:1fr 5cqw 1fr 5cqw 1fr;align-items:stretch">
        <div style="background:#dceee9;border-radius:1.2cqw;padding:1.8cqh 1.5cqw;text-align:center">
          <div class="metric-number accent" style="font-size:2.9cqw">85%</div>
          <b style="font-size:1.08cqw">Normal Fit</b>
          <div class="metric-note">Model training & memory bank</div>
        </div>
        <div style="display:grid;place-items:center;font-size:2.2cqw;color:var(--coral)">→</div>
        <div style="background:#f8e5d8;border-radius:1.2cqw;padding:1.8cqh 1.5cqw;text-align:center">
          <div class="metric-number coral" style="font-size:2.9cqw">15%</div>
          <b style="font-size:1.08cqw">Normal Validation</b>
          <div class="metric-note">Operating threshold calibration</div>
        </div>
        <div style="display:grid;place-items:center;font-size:2.2cqw;color:var(--coral)">→</div>
        <div style="background:#152d38;color:#edf6f4;border-radius:1.2cqw;padding:1.8cqh 1.5cqw;text-align:center">
          <div class="metric-number" style="font-size:2.9cqw;color:#8dd8c9">TEST</div>
          <b style="font-size:1.08cqw">Final Evaluation</b>
          <div class="metric-note" style="color:#a9bdc1">Unseen split · scored strictly once</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2.5cqw">
        <div class="card-box">
          <span class="badge" style="margin-bottom:.8cqh">Primary Decision</span>
          <div style="font-size:2cqw;font-weight:800;margin-top:.4cqh">Image F1</div>
          <div class="def-rule" style="margin:.8cqh 0"></div>
          <div style="font-size:1.02cqw;line-height:1.4;color:var(--muted)">
            Evaluates binary pass/fail decision quality at the frozen validation-derived 95th-percentile threshold (&alpha; = 0.05).
          </div>
        </div>
        <div class="card-box">
          <span class="badge" style="margin-bottom:.8cqh">Threshold-Free</span>
          <div style="font-size:2cqw;font-weight:800;margin-top:.4cqh">PR-AUC</div>
          <div class="def-rule" style="margin:.8cqh 0"></div>
          <div style="font-size:1.02cqw;line-height:1.4;color:var(--muted)">
            Measures continuous score ranking robustness under severe class imbalance without relying on true negatives.
          </div>
        </div>
        <div class="card-box">
          <span class="badge badge-coral" style="margin-bottom:.8cqh">Strict Localisation</span>
          <div style="font-size:2cqw;font-weight:800;margin-top:.4cqh">AUPIMO</div>
          <div class="def-rule" style="margin:.8cqh 0"></div>
          <div style="font-size:1.02cqw;line-height:1.4;color:var(--muted)">
            Area Under Per-Image Overlap over the ultra-low false-positive window $\text{FPR} \in [10^{-5}, 10^{-4}]$ at canonical 256×256.
          </div>
        </div>
      </div>
    </div>
    <div class="source">Protocol: fair-eval-v1 (seed=42) · Canonical 256×256 resolution · Bertoldo et al. (2024)</div>
    """
    slide_html("Evaluation without Leakage", body, eyebrow="04 · Evidence Protocol")
