"""Title slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html


def render() -> None:
    """Render the defense title and project objective."""
    visual = DEFECT_EXAMPLES / "title_capsule_scratch.png"
    body = f"""
    <div class="slide-grid-2" style="grid-template-columns:1.15fr .85fr;align-items:center">
      <div>
        <div style="font-size:1.85cqw;line-height:1.25;max-width:42cqw;color:var(--ink);font-weight:600">
          Unsupervised visual inspection from normal-only manufacturing parts — from generative reconstruction to deep transfer learning.
        </div>
        <div class="def-rule"></div>
        <div style="display:flex;gap:.8cqw;margin-bottom:2cqh">
          <span class="badge">Unsupervised AD</span>
          <span class="badge">Deep Transfer Learning</span>
          <span class="badge">Zero-Leakage Protocol</span>
        </div>
        <div class="tiny-label">Project defense team</div>
        <div style="margin-top:.8cqh;font-size:1.12cqw;line-height:1.45;color:var(--muted)">
          Benjamin Förster · Ali Abul Hawa · Mahboubeh Nadaf · Karim Khalifa
        </div>
      </div>
      <div class="image-frame" style="width:48cqh;height:48cqh;margin-left:auto;background:transparent">
        {img(visual, "Capsule with a subtle scratch defect", "photo contain")}
      </div>
    </div>
    """
    slide_html("Industrial Component<br><span class='accent'>Anomaly Detection</span>", body, eyebrow="Project defense")
