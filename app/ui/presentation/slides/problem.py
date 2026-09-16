"""Problem framing slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DATASET, img, slide_html


def render() -> None:
    """Show why pixel imbalance makes the inspection problem difficult."""
    defect = DATASET / "capsule" / "test" / "squeeze" / "013.png"
    mask = DATASET / "capsule" / "ground_truth" / "squeeze" / "013_mask.png"
    body = f"""
    <div style="display:grid;grid-template-columns:1.25fr .52fr .92fr;gap:2.6cqw;height:100%;align-items:center">
      <div>
        <div class="tiny-label" style="margin-bottom:1cqh">Defective image</div>
        <div class="image-frame" style="height:45cqh">{img(defect, "Defective capsule", "photo")}</div>
      </div>
      <div>
        <div class="tiny-label" style="margin-bottom:1cqh">Pixel mask</div>
        <div class="image-frame" style="height:24cqh;background:#050505">{img(mask, "Small ground-truth defect mask", "photo contain")}</div>
      </div>
      <div>
        <div class="metric-number coral">~1.5%</div>
        <div style="font-size:1.55cqw;font-weight:760;line-height:1.15;margin-top:1.4cqh">median anomalous pixels</div>
        <div class="def-rule"></div>
        <div class="flat-note">A model can label almost every pixel “normal” and still report impressive accuracy.</div>
      </div>
    </div>
    <div class="source">MVTec AD · 1,258 defective test masks</div>
    """
    slide_html("Why the problem is hard", body, eyebrow="01 · The inspection challenge")
