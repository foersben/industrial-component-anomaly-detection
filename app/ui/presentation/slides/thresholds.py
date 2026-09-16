"""Threshold trade-off slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import GENERATED_CHARTS, img, slide_html


def render() -> None:
    """Use an actual evaluation curve to explain threshold economics."""
    curve = GENERATED_CHARTS / "patchcore_bottle_threshold_tradeoff.png"
    body = f"""
    <div style="display:grid;grid-template-columns:1.3fr .7fr;gap:3.7cqw;height:100%;align-items:center">
      <div class="image-frame" style="height:48cqh;background:white;padding:1cqh 1cqw">
        {img(curve, "Precision and recall across thresholds for bottle", "photo contain")}
      </div>
      <div>
        <div style="font-size:1.85cqw;font-weight:790;line-height:1.2">The threshold sets the cost balance</div>
        <div class="def-rule"></div>
        <div style="display:grid;gap:2.2cqh;font-size:1.18cqw;line-height:1.35">
          <div><b class="coral">Lower threshold</b><br><span class="def-sub" style="font-size:1.02cqw">More defects caught, more false rejects</span></div>
          <div><b class="accent">Higher threshold</b><br><span class="def-sub" style="font-size:1.02cqw">Fewer false rejects, more missed defects</span></div>
          <div class="flat-note">Calibrate with plant-specific costs and accepted normal validation data.</div>
        </div>
      </div>
    </div>
    <div class="source">Actual PatchCore bottle evaluation curve</div>
    """
    slide_html("Threshold = business decision", body, eyebrow="07 · Operating point")
