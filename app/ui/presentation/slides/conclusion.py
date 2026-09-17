"""Deployment recommendation slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Render the recommended production system, validation gates, and future prospects."""
    body = """
    <div style="display:grid;grid-template-columns:1.08fr .92fr;gap:3.5cqw;height:100%;align-items:center">
      <div>
        <div style="display:flex;gap:.8cqw;margin-bottom:1.4cqh">
          <span class="badge">Production Recommendation</span>
          <span class="badge">PatchCore + ResNet-18</span>
        </div>
        <div style="font-size:1.55cqw;font-weight:790;line-height:1.2;color:var(--ink)">
          Day-One Factory Quality Gate from Normal-Only Parts
        </div>
        <div class="def-rule" style="margin:1.4cqh 0"></div>
        <div style="display:grid;gap:1.2cqh">
          <div class="card-box" style="padding:1.2cqh 1.2cqw">
            <b>01 · Takt Time & Cycle Verification</b><br>
            <span style="font-size:.92cqw;color:var(--muted)">Benchmark forward-pass & coreset search latency directly on plant IPC hardware.</span>
          </div>
          <div class="card-box" style="padding:1.2cqh 1.2cqw">
            <b>02 · Plant-Specific Calibration</b><br>
            <span style="font-size:.92cqw;color:var(--muted)">Freeze operational decision thresholds on the 15% local normal validation pool.</span>
          </div>
          <div class="card-box" style="padding:1.2cqh 1.2cqw">
            <b>03 · Drift Control Without Retraining</b><br>
            <span style="font-size:.92cqw;color:var(--muted)">Absorb material and lighting shifts by updating memory banks via QA approval, zero weight retraining.</span>
          </div>
        </div>
      </div>

      <div>
        <div class="card-box-dark" style="padding:2.4cqh 1.8cqw">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="tiny-label" style="color:#8dd8c9">Research Horizon</span>
            <span class="badge" style="background:rgba(141,216,201,.2);color:#8dd8c9;border-color:#8dd8c9">Future Prospects</span>
          </div>
          <div style="font-size:1.3cqw;font-weight:780;color:#eef7f6;margin:1.1cqh 0">
            Self-Supervised Vision Transformers (DINOv2 / DINOv3)
          </div>
          <div style="font-size:.95cqw;line-height:1.4;color:#b8ccce">
            Exploratory tests show foundation models push the representation ceiling:
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1cqw;margin:1.4cqh 0">
            <div style="background:rgba(255,255,255,.06);border-radius:.8cqw;padding:1cqh .8cqw">
              <div class="metric-number" style="font-size:2cqw;color:#8dd8c9">0.680</div>
              <div style="font-size:.85cqw;color:#aec2c7">DINOv3 AUPIMO (+7.7 pts)</div>
            </div>
            <div style="background:rgba(255,255,255,.06);border-radius:.8cqw;padding:1cqh .8cqw">
              <div class="metric-number" style="font-size:2cqw;color:#8dd8c9">0.957</div>
              <div style="font-size:.85cqw;color:#aec2c7">DINOv3 Image F1</div>
            </div>
          </div>
          <div style="font-size:.9cqw;line-height:1.35;color:#8dd8c9">
            <b>Status:</b> Future roadmap. Industrial PC hardware memory limits and clean protocol freezing retain ResNet-18 as today's deployment standard.
          </div>
        </div>
      </div>
    </div>
    <div class="source">Source: Business Report Section 8 & Appendix A1 · Strategic Recommendations & Future Directions</div>
    """
    slide_html("Deployment Recommendation & Future Horizons", body, eyebrow="11 · Strategic Roadmap")
