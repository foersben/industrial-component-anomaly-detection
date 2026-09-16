"""Deployment recommendation slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Render the recommended system and the remaining validation gates."""
    body = """
    <div style="display:grid;grid-template-columns:1.14fr .86fr;gap:5cqw;height:100%;align-items:center">
      <div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.8cqw;align-items:center">
          <div style="background:#dceee9;color:#10212b;border-radius:1cqw;padding:2.3cqh .8cqw;text-align:center;font-size:1.05cqw;font-weight:760">PatchCore</div>
          <div style="background:#dceee9;color:#10212b;border-radius:1cqw;padding:2.3cqh .8cqw;text-align:center;font-size:1.05cqw;font-weight:760">ResNet-18</div>
          <div style="background:#f7e2d5;color:#10212b;border-radius:1cqw;padding:2.3cqh .8cqw;text-align:center;font-size:1.05cqw;font-weight:760">Calibrated threshold</div>
          <div style="background:#0f3038;color:white;border-radius:1cqw;padding:2.3cqh .8cqw;text-align:center;font-size:1.05cqw;font-weight:760">Heatmap + contour</div>
        </div>
        <div style="font-size:2.1cqw;font-weight:790;line-height:1.18;margin-top:5cqh">A practical quality gate from normal-only examples</div>
        <div class="def-sub" style="font-size:1.18cqw;margin-top:1.4cqh">Local reference memory keeps onboarding simple and operator evidence visible.</div>
      </div>
      <div>
        <div class="tiny-label">Before production</div>
        <div style="display:grid;gap:2.1cqh;margin-top:2cqh;font-size:1.18cqw;line-height:1.35">
          <div><b>01&nbsp; Takt time</b><br><span class="def-sub" style="font-size:1cqw">Measure latency on the target line hardware.</span></div>
          <div><b>02&nbsp; Plant calibration</b><br><span class="def-sub" style="font-size:1cqw">Freeze thresholds on accepted local data.</span></div>
          <div><b>03&nbsp; Drift control</b><br><span class="def-sub" style="font-size:1cqw">Approve memory updates through QA.</span></div>
        </div>
      </div>
    </div>
    """
    slide_html("Deployment recommendation", body, eyebrow="09 · Decision")
