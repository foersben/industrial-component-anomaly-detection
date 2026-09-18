"""Problem framing slide."""

# ruff: noqa: E501

from app.ui.presentation.theme import GENERATED_CHARTS, img, slide_html


def render() -> None:
    """Show how pixel and dataset class imbalances create the accuracy paradox and dictate metrics."""
    chart_pixel = GENERATED_CHARTS / "imbalance_pixel_area.png"
    chart_split = GENERATED_CHARTS / "imbalance_test_split.png"

    body = f"""<div style="display:flex;flex-direction:column;gap:1.2cqh;margin-bottom:2.6cqh">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.6cqw">
        <div class="card-box" style="padding:1cqh 1.2cqw;background:#ffffff">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4cqh">
            <div class="tiny-label" style="margin:0;font-weight:700">1. Spatial / Pixel-Level Imbalance</div>
            <span class="badge badge-coral" style="font-size:0.68cqw;padding:0.2cqh 0.5cqw">Median: 1.54% Area</span>
          </div>
          <div class="image-frame" style="height:25cqh;background:#ffffff;border:none">
            {img(chart_pixel, "Spatial / Pixel Imbalance: Defect area distribution across 15 categories (log scale)", "photo contain")}
          </div>
          <div style="font-size:0.75cqw;color:var(--muted);margin-top:0.4cqh;line-height:1.35">
            Defect areas range from <b>0.037%</b> to <b>49.3%</b> (median <b>1.54%</b>). <b>98.46%</b> of pixels are conforming background/structure.
          </div>
        </div>
        <div class="card-box" style="padding:1cqh 1.2cqw;background:#ffffff">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4cqh">
            <div class="tiny-label" style="margin:0;font-weight:700">2. Dataset / Split-Level Imbalance</div>
            <span class="badge" style="font-size:0.68cqw;padding:0.2cqh 0.5cqw">3,629:0 One-Class Fit</span>
          </div>
          <div class="image-frame" style="height:25cqh;background:#ffffff;border:none">
            {img(chart_split, "Dataset / Split Imbalance: Test set composition across 15 categories", "photo contain")}
          </div>
          <div style="font-size:0.75cqw;color:var(--muted);margin-top:0.4cqh;line-height:1.35">
            Training fit is strictly <b>100% normal</b> (zero anomalies). Test sets are defect-enriched (<b>60&ndash;84%</b> anomalous) for evaluation.
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1.05fr 1fr 1fr;gap:1.2cqw;height:auto">
        <div class="card-box" style="padding:1cqh 1.1cqw;border-left:3px solid var(--coral)">
          <div class="tiny-label" style="color:var(--coral);margin-bottom:0.3cqh">The Accuracy Paradox</div>
          <div style="font-size:0.78cqw;line-height:1.35;color:var(--ink)">
            An empty model predicting 100% "conforming" achieves <b>98.46% pixel accuracy</b> while detecting <b>0.0%</b> of manufacturing defects.
          </div>
        </div>
        <div class="card-box" style="padding:1cqh 1.1cqw;border-left:3px solid var(--teal)">
          <div class="tiny-label" style="color:var(--teal);margin-bottom:0.3cqh">Image-Level Mandate</div>
          <div style="font-size:0.78cqw;line-height:1.35;color:var(--ink)">
            Because test sets are defect-dense, raw accuracy is misleading. <b>PR-AUC</b> and calibrated <b>F1 score</b> are mathematically mandatory.
          </div>
        </div>
        <div class="card-box" style="padding:1cqh 1.1cqw;border-left:3px solid var(--ink)">
          <div class="tiny-label" style="margin-bottom:0.3cqh">Pixel-Level Mandate</div>
          <div style="font-size:0.78cqw;line-height:1.35;color:var(--ink)">
            Normal pixels dominate TN, inflating AUROC. <b>AUPIMO</b> evaluates per-image overlap under strict false-positive bounds (FPR &le; 10<sup>&minus;4</sup>).
          </div>
        </div>
      </div>
    </div>
    <div class="source">Source: Exploratory Data Analysis &middot; MVTec AD 1,258 ground-truth defect masks &amp; dataset split distributions</div>"""
    slide_html(
        "The Inspection Challenge: Class Imbalances & The Accuracy Paradox", body, eyebrow="01 · Data Exploration"
    )
