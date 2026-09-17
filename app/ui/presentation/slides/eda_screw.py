"""EDA Case Study: The Screw Discovery."""

# ruff: noqa: E501

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html


def render() -> None:
    """Render the exploratory engineering case study on the screw category."""
    visual = DEFECT_EXAMPLES / "screw_masking_bug_vs_fix.png"
    body = f"""
    <div class="slide-grid-2" style="grid-template-columns:1.05fr .95fr;gap:2.8cqw;height:100%;align-items:center">
      <div>
        <div style="display:flex;gap:.8cqw;margin-bottom:1.4cqh">
          <span class="badge">Edge Case Study</span>
          <span class="badge badge-coral">Rotational Variance</span>
          <span class="badge">Greyscale Sensor</span>
        </div>
        <div style="font-size:1.45cqw;font-weight:780;line-height:1.2;color:var(--ink)">
          The Masking Bug: Diagnosing Data Invariants
        </div>
        <div class="def-rule" style="margin:1.4cqh 0"></div>
        <div class="card-box" style="margin-bottom:1.4cqh">
          <div style="font-size:1.02cqw;line-height:1.4;color:var(--ink)">
            <b>The 62% Background Miss:</b> Sampling 500 random masked training patches revealed that <b>62.0%</b> fell entirely on empty background, starving the network of structural defect signal (AP plummeted to 0.0117).
          </div>
        </div>
        <div class="card-box" style="background:#dceee9;border-color:rgba(0,127,122,.25)">
          <div style="font-size:1.02cqw;line-height:1.4;color:var(--ink)">
            <b>Object-Centred Fix:</b> Restricting patch masks to the segmented foreground restored the training gradient directly to the screw body, validating domain-informed preprocessing.
          </div>
        </div>
        <div style="font-size:.92cqw;color:var(--muted);margin-top:1.2cqh">
          Lesson: Unsupervised algorithms fail silently without rigorous exploratory data diagnostics.
        </div>
      </div>
      <div style="display:grid;grid-template-rows:auto 1fr;gap:1cqh">
        <div class="tiny-label">Masking Diagnostic: Random Frame vs. Foreground Constrained</div>
        <div class="image-frame" style="height:44cqh;background:#071117">
          {img(visual, "Screw masking bug vs object-centred fix", "photo contain")}
        </div>
      </div>
    </div>
    <div class="source">Source: Technical Report Section 5.4 · Screw Exploratory Milestones S3 vs. S4</div>
    """
    slide_html("EDA Case Study: The Screw Discovery", body, eyebrow="03 · Data Exploration")
