"""Evaluation protocol slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Render the leakage-safe split and metric roles."""
    body = """
    <div style="display:grid;grid-template-rows:24cqh 1fr;gap:5cqh;height:100%">
      <div style="display:grid;grid-template-columns:1fr 6cqw 1fr 6cqw 1fr;align-items:stretch">
        <div style="background:#dceee9;border-radius:1.2cqw;padding:2.2cqh 1.6cqw;text-align:center">
          <div class="metric-number accent" style="font-size:3.2cqw">85%</div><b>normal fit</b><div class="metric-note">memory bank</div>
        </div>
        <div style="display:grid;place-items:center;font-size:2.5cqw;color:var(--coral)">→</div>
        <div style="background:#f8e5d8;border-radius:1.2cqw;padding:2.2cqh 1.6cqw;text-align:center">
          <div class="metric-number coral" style="font-size:3.2cqw">15%</div><b>normal validation</b><div class="metric-note">threshold calibration</div>
        </div>
        <div style="display:grid;place-items:center;font-size:2.5cqw;color:var(--coral)">→</div>
        <div style="background:#152d38;color:#edf6f4;border-radius:1.2cqw;padding:2.2cqh 1.6cqw;text-align:center">
          <div class="metric-number" style="font-size:3.2cqw">TEST</div><b>final evaluation</b><div class="metric-note" style="color:#a9bdc1">touched once</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3cqw">
        <div><div style="font-size:2.45cqw;font-weight:800">F1</div><div class="def-rule" style="margin:1cqh 0"></div><div class="def-sub" style="font-size:1.12cqw">Decision quality at the frozen threshold</div></div>
        <div><div style="font-size:2.45cqw;font-weight:800">PR-AUC</div><div class="def-rule" style="margin:1cqh 0"></div><div class="def-sub" style="font-size:1.12cqw">Ranking under class imbalance</div></div>
        <div><div style="font-size:2.45cqw;font-weight:800">AUPIMO</div><div class="def-rule" style="margin:1cqh 0"></div><div class="def-sub" style="font-size:1.12cqw">Localisation at ultra-low false-positive rates</div></div>
      </div>
    </div>
    """
    slide_html("Evaluation without leakage", body, eyebrow="03 · Evidence protocol")
