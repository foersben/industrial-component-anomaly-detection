"""Problem framing slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html


def render() -> None:
    """Show why pixel imbalance and the accuracy paradox dictate metric selection."""
    defect = DEFECT_EXAMPLES / "problem_capsule_squeeze.png"
    mask = DEFECT_EXAMPLES / "problem_capsule_squeeze_mask.png"
    body = f"""
    <div style="display:grid;grid-template-columns:1.05fr .48fr 1.12fr;gap:2.2cqw;height:100%;align-items:center">
      <div>
        <div class="tiny-label" style="margin-bottom:.8cqh">Industrial sample (Capsule)</div>
        <div class="image-frame" style="height:42cqh">{img(defect, "Defective capsule", "photo")}</div>
      </div>
      <div>
        <div class="tiny-label" style="margin-bottom:.8cqh">Defect ground truth</div>
        <div class="image-frame" style="height:22cqh;background:#050505">{img(mask, "Small ground-truth defect mask", "photo contain")}</div>
        <div style="font-size:.92cqw;color:var(--muted);margin-top:1.2cqh;line-height:1.3">
          1,258 defective masks evaluated across 15 categories.
        </div>
      </div>
      <div>
        <div style="display:flex;align-items:baseline;gap:.8cqw">
          <span class="metric-number coral">1.54%</span>
          <span style="font-size:1.15cqw;color:var(--muted);font-weight:600">median area</span>
        </div>
        <div style="font-size:1.3cqw;font-weight:750;line-height:1.2;margin-top:.6cqh">
          Extreme Spatial Imbalance & The Accuracy Paradox
        </div>
        <div class="def-rule" style="margin:1.4cqh 0"></div>
        <div class="card-box" style="margin-bottom:1.4cqh">
          <div style="font-size:1.05cqw;line-height:1.4;color:var(--ink)">
            <b>The Accuracy Paradox:</b> An empty model predicting 100% "normal" scores <b>98.46% pixel accuracy</b> while catching <b>0%</b> of manufacturing flaws.
          </div>
        </div>
        <div class="flat-note" style="font-size:1.02cqw">
          Defect scales range from 0.037% (hairline pinholes) to 49.3%. Standard AUROC is distorted by normal pixels — <b>PR-AUC</b> and <b>AUPIMO</b> are mathematically mandatory.
        </div>
      </div>
    </div>
    <div class="source">Source: Exploratory Data Analysis · MVTec AD defect mask distribution</div>
    """
    slide_html("The Inspection Challenge & The Accuracy Paradox", body, eyebrow="01 · Data Exploration")
