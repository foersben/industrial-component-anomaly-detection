"""Paradigm Shift: The Transfer Learning Catalyst."""

# ruff: noqa: E501

from app.ui.presentation.theme import slide_html


def render() -> None:
    """Explain the decisive architectural shift from training from scratch to transfer learning."""
    body = """
    <div style="display:grid;grid-template-columns:1.05fr 4cqw 1.05fr;gap:1.5cqw;height:100%;align-items:center">
      <div class="card-box" style="padding:2.8cqh 2cqw">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="tiny-label">Paradigm 1</span>
          <span class="badge badge-coral">From Scratch</span>
        </div>
        <div style="font-size:1.6cqw;font-weight:780;margin:1cqh 0;color:var(--ink)">
          Generative Autoencoders
        </div>
        <div class="def-sub" style="font-size:1.05cqw;line-height:1.4">
          Attempted to learn the complete nominal image distribution from scratch on &lt;300 normal training images.
        </div>
        <div class="def-rule" style="margin:1.6cqh 0"></div>
        <div style="font-size:1cqw;line-height:1.4;color:var(--muted)">
          <b>Outcome:</b> Severe capacity dilemma. Optimization minimizes global variance, blurring fine textures and losing localized defect contrast (Best Bottle AUROC: <b>0.877</b>).
        </div>
      </div>

      <div style="display:grid;place-items:center;font-size:2.8cqw;color:var(--teal);font-weight:900">
        →
      </div>

      <div class="card-box-dark" style="padding:2.8cqh 2cqw">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span class="tiny-label" style="color:#8dd8c9">Paradigm 2 · The Catalyst</span>
          <span class="badge" style="background:rgba(141,216,201,.2);color:#8dd8c9;border-color:#8dd8c9">Transfer Learning</span>
        </div>
        <div style="font-size:1.6cqw;font-weight:780;margin:1cqh 0;color:#eef7f6">
          Pre-trained Feature Transfer
        </div>
        <div style="font-size:1.05cqw;line-height:1.4;color:#b8ccce">
          Exploratory frozen ResNet-18 (ImageNet) + 5-NN nearest-neighbour scoring on Bottle.
        </div>
        <div class="def-rule" style="background:#8dd8c9;margin:1.6cqh 0"></div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1cqw">
          <div>
            <div class="metric-number" style="font-size:2.4cqw;color:#8dd8c9">0.988</div>
            <div style="font-size:.92cqw;color:#aec2c7">AUROC on Bottle</div>
          </div>
          <div>
            <div class="metric-number" style="font-size:2.4cqw;color:#8dd8c9">0.959</div>
            <div style="font-size:.92cqw;color:#aec2c7">Image F1 Score</div>
          </div>
        </div>
        <div style="font-size:.92cqw;color:#8dd8c9;margin-top:1.2cqh;line-height:1.35">
          Caught 59/63 defects with only 1 false alarm. Proved that general visual priors decisively outperform from-scratch learning.
        </div>
      </div>
    </div>
    <div class="source">Source: Technical Report Section 5.3 · Bottle Pretrained 5-NN Baseline Experiment</div>
    """
    slide_html("The Transfer Learning Catalyst", body, eyebrow="05 · Technical Architecture")
