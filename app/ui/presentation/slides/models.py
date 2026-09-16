"""Model evolution slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Explain the model progression and the reason for each move."""
    body = """
    <div style="display:grid;grid-template-columns:1fr 5cqw 1fr 5cqw 1.08fr;height:100%;align-items:center">
      <div>
        <div class="tiny-label">First signal</div>
        <div style="font-size:2.25cqw;font-weight:790;margin:1.1cqh 0">Simple baselines</div>
        <div class="def-sub" style="font-size:1.18cqw">Pixel statistics exposed the imbalance, but could not model appearance.</div>
      </div>
      <div style="text-align:center;font-size:2.6cqw;color:var(--muted)">→</div>
      <div>
        <div class="tiny-label">Reconstruction</div>
        <div style="font-size:2.25cqw;font-weight:790;margin:1.1cqh 0">AE / CAE</div>
        <div class="def-sub" style="font-size:1.18cqw">Residual maps found defects, but also punished valid texture and lighting.</div>
      </div>
      <div style="text-align:center;font-size:2.6cqw;color:var(--muted)">→</div>
      <div style="background:#0d2a33;color:#eef6f4;border-radius:1.4cqw;padding:4cqh 2.2cqw;box-shadow:0 1.2cqw 3cqw #10212b33">
        <div class="tiny-label" style="color:#8dd8c9">Selected approach</div>
        <div style="font-size:2.5cqw;font-weight:820;margin:1.1cqh 0">PatchCore</div>
        <div style="font-size:1.18cqw;line-height:1.4;color:#b8ccce">Deep patch features compare local appearance with a memory of normal examples.</div>
      </div>
    </div>
    """
    slide_html("Model evolution", body, eyebrow="04 · What changed and why")
