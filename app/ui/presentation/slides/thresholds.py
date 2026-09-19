"""Threshold trade-off slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import GENERATED_CHARTS, img, slide_html


def render() -> None:
    """Use evaluation trade-offs to frame operational threshold calibration and tiering."""
    curve = GENERATED_CHARTS / "patchcore_bottle_threshold_tradeoff.png"
    body = f"""
    <div style="display:grid;grid-template-columns:1.15fr .85fr;gap:3cqw;height:100%;align-items:center">
      <div style="display:grid;grid-template-rows:auto 1fr;gap:1cqh">
        <div class="tiny-label">Empirical Precision-Recall Trade-off Curve (Bottle)</div>
        <div class="image-frame" style="height:44cqh;background:white;padding:.8cqh .8cqw">
          {img(curve, "Precision and recall across thresholds for bottle", "photo contain")}
        </div>
      </div>
      <div>
        <div style="font-size:1.35cqw;font-weight:780;line-height:1.2;color:var(--ink)">
          Balancing Pseudo-Scrap vs. Defect Escape
        </div>
        <div class="def-rule" style="margin:1.2cqh 0"></div>
        <div style="display:grid;gap:1.2cqh">
          <div class="card-box" style="padding:1.2cqh 1.2cqw">
            <b class="coral" style="font-size:1.02cqw">Tier 1 · Safety-Critical (e.g. Pill)</b><br>
            <span style="font-size:.92cqw;color:var(--muted)">Lower threshold (Recall &gt; 99%). Near-zero escape tolerance; flagged borderline parts enter a manual review loop.</span>
          </div>
          <div class="card-box" style="padding:1.2cqh 1.2cqw">
            <b class="accent" style="font-size:1.02cqw">Tier 2 · Autonomous Baseline (e.g. Bottle)</b><br>
            <span style="font-size:.92cqw;color:var(--muted)">Calibrated at &alpha; = 0.05 on 15% normal validation pool. Maximizes F1 score autonomously on the line.</span>
          </div>
          <div class="card-box" style="padding:1.2cqh 1.2cqw">
            <b style="font-size:1.02cqw;color:var(--ink)">Tier 3 · High-Volume Fasteners (e.g. Screw)</b><br>
            <span style="font-size:.92cqw;color:var(--muted)">Conservative threshold to avoid rejecting conforming units with benign machining marks.</span>
          </div>
        </div>
      </div>
    </div>
    <div class="source">Source: Business Report Section 5 · Operational Decision Making & Economic Cost Model</div>
    """
    slide_html("Thresholds as an Economic Decision", body, eyebrow="09 · Business Economics")
