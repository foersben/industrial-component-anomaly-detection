"""Streamlit renderers mirroring the 18-page LaTeX Beamer defense deck."""

# The HTML is intentionally formatted as presentation markup.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import streamlit as st

from app.ui.presentation.theme import (
    DEFECT_EXAMPLES,
    PDF_FIGURES,
    beamer_section_html,
    beamer_slide_html,
    img,
    math_html,
)


def render_title() -> None:
    """Render the PDF title page."""
    rows = "".join(
        f"""
        <div class="pdf-examples-row">
          <div class="pdf-example-name"><b>{name}</b><span>{defect}</span></div>
          <div class="pdf-example-image">{img(DEFECT_EXAMPLES / good, f'Conforming {name}', 'pdf-image')}<span class="pdf-example-badge">GOOD</span></div>
          <div class="pdf-example-image">{img(DEFECT_EXAMPLES / bad, f'Defective {name}', 'pdf-image')}<span class="pdf-example-badge defect">DEFECT</span></div>
        </div>
        """
        for name, defect, good, bad in (
            ("Bottle", "Broken rim", "title_good_bottle.png", "title_anomaly_bottle.png"),
            ("Capsule", "Scratch", "title_good_capsule.png", "title_anomaly_capsule.png"),
            ("Leather", "Surface cut", "title_good_leather.png", "title_anomaly_leather.png"),
            ("Screw", "Thread defect", "title_good_screw.png", "title_anomaly_screw.png"),
        )
    )
    st.markdown(
        f"""
        <div class="pdf-title-slide"><div class="pdf-title-grid">
          <div class="pdf-title-copy">
            <h1>Industrial Component<br>Anomaly Detection</h1>
            <div class="pdf-title-subhead">Project Presentation</div>
            <div class="title-rule"></div>
            <div class="subtitle">Unsupervised defect classification and pixel-level anomaly localisation from normal-only manufacturing parts.</div>
            <div class="pdf-title-pills"><span class="pdf-pill">Unsupervised AD</span><span class="pdf-pill">Deep Transfer Learning</span><span class="pdf-pill">Zero-Leakage Protocol</span></div>
            <div class="pdf-title-team"><div class="pdf-label">Team</div><div>B. Förster · A. Abul Hawa · M. Nadaf · K. Khalifa</div></div>
          </div>
          <div class="pdf-examples">
            <div class="pdf-examples-heading"><span>MVTec AD 2D Dataset</span><span>Selected Examples · Objects &amp; Textures</span></div>
            <div class="pdf-examples-columns"><span>Category</span><span>Conforming</span><span>Defective</span></div>
            {rows}
          </div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )


def render_data_section() -> None:
    """Render the data exploration divider."""
    beamer_section_html("Data Exploration")


def render_dataset_imbalance() -> None:
    """Render dataset composition and class imbalance."""
    charts = PDF_FIGURES / "charts"
    kruskal_stat = math_html(r"H(14)\approx 626.5,\ p<0.001")
    body = f"""
    <div class="pdf-grid-2" style="grid-template-columns:1.08fr .92fr">
      <div class="pdf-reading-column">
        <b>Dataset at a glance</b>
        <ul><li>15 manufacturing categories: 10 objects + 5 textures</li><li>&gt; 5000 images, ≈ 1200 anomalous images</li><li>Normal-only training; test set has pixel-level anomaly ground truth</li></ul>
        <div class="pdf-accent" style="font-weight:750;margin-top:2.2cqh">Accuracy paradox</div>
        <ul><li>Median anomalous-pixel area: <b>1.54%</b></li><li>Predicting <i>all-normal</i> yields &gt;98.5% pixel accuracy while catching <i>zero</i> defects.</li></ul>
        <b>Kruskal-Wallis: {kruskal_stat}</b>
        <ul><li>43× spread in defect size across categories.</li></ul>
      </div>
      <div style="display:grid;grid-template-rows:auto minmax(0,1fr) auto minmax(0,1fr);gap:.65cqh;min-height:0;height:100%">
        <div class="pdf-label">Test-set class imbalance (images)</div><div style="min-height:0;overflow:hidden">{img(charts / 'imbalance_test_split.png', 'Test-set class imbalance', 'pdf-image')}</div>
        <div class="pdf-label">Pixel-area imbalance (defect region %)</div><div style="min-height:0;overflow:hidden">{img(charts / 'imbalance_pixel_area.png', 'Pixel-area imbalance', 'pdf-image')}</div>
      </div>
    </div>"""
    beamer_slide_html("The MVTec AD Dataset - Class Imbalance", body, frame_number=2)


def render_taxonomy() -> None:
    """Render object/texture defect taxonomy and examples."""
    defects = PDF_FIGURES / "defect_examples"
    examples = [
        f"<figure class='pdf-taxonomy-figure'>{img(defects / filename, label, 'pdf-image')}<figcaption>{label}</figcaption></figure>"
        for filename, label in (
            ("slide_03_screw_thread_contour.png", "Screw thread"),
            ("slide_03_cable_missing_contour.png", "Cable missing"),
            ("title_anomaly_capsule.png", "Capsule scratch"),
            ("slide_03_grid_glue_contour.png", "Grid glue strip"),
            ("title_anomaly_leather.png", "Leather cut"),
            ("problem_capsule_squeeze_mask.png", "GT mask example"),
        )
    ]
    body = f"""
    <div class="pdf-taxonomy-layout">
      <div class="pdf-taxonomy-copy">
        <h2>Two fundamentally different defect regimes:</h2>
        <div class="pdf-taxonomy-card"><div><b>Structural Objects</b> (e.g. Screw, Cable)</div><ul><li>Defects cluster at specific locations</li><li>Strong positional bias (shortcut risk)</li><li>Foreground/background separation critical</li></ul></div>
        <div class="pdf-taxonomy-card coral"><div><b>Continuous Textures</b> (e.g. Grid, Leather)</div><ul><li>Defects broadly distributed across surface</li><li>Rich augmentation needed (rotation, scale)</li><li>No foreground mask required</li></ul></div>
        <p><b>Covariate shift:</b> Screw, Leather, Grid show measurable brightness/contrast drift between train and test normal images - motivates photometric augmentation.</p>
      </div>
      <div class="pdf-taxonomy-gallery">
        <div class="pdf-taxonomy-group"><div class="pdf-taxonomy-group-title">Objects - localised defects <span>(contour = defect boundary)</span></div><div class="pdf-taxonomy-grid">{''.join(examples[:3])}</div></div>
        <div class="pdf-taxonomy-group coral"><div class="pdf-taxonomy-group-title">Textures - distributed defects <span>(defect across full surface)</span></div><div class="pdf-taxonomy-grid">{''.join(examples[3:])}</div></div>
      </div>
    </div>"""
    beamer_slide_html("Defect Taxonomy &amp; Spatial Distribution", body, frame_number=3)


def render_metrics() -> None:
    """Render metric-selection rationale."""
    chart = PDF_FIGURES / "charts" / "binarization_threshold_aupimo.png"
    fpr_window = math_html(r"[10^{-5},10^{-4}]")
    body = f"""
    <div class="pdf-grid-2" style="grid-template-columns:1.04fr .96fr">
      <div>
        <h3 class="pdf-accent" style="font-size:1.15cqw">The problem with Pixel-AUROC</h3>
        <ul><li>AUROC relies on False Positive Rate<ul><li>With ≈98.5% normal pixels, TN dominates - FPR barely moves.</li><li>SOTA scores all exceed 99%, making differences indistinguishable.</li></ul></li></ul>
        <h3 class="pdf-kicker" style="font-size:1.15cqw;margin-top:2cqh">Our Evaluation Paradigm</h3>
        <ul><li><b>AUPIMO:</b> Pixel-level threshold over the ultra-low FPR window {fpr_window} on normal images.</li><li><b>Dual Thresholds:</b> Distinct image-level and pixel-level anomaly thresholds.</li><li><b>F1-Score Focus:</b> Additionally, we always evaluate the F1-Score.</li></ul>
      </div>
      <div style="display:grid;grid-template-rows:1fr auto;gap:1cqh"><div>{img(chart, 'Binarization threshold curve and AUPIMO range', 'pdf-image')}</div><div class="pdf-tiny pdf-muted" style="text-align:center">Binarization threshold curve &amp; AUPIMO range</div></div>
    </div>"""
    beamer_slide_html("Metric Selection: AUPIMO &amp; F1-Score", body, frame_number=4)


def render_models_section() -> None:
    """Render the models divider."""
    beamer_section_html("Models")


def render_model_overview() -> None:
    """Render the two-model architecture overview."""
    body = """
    <div class="pdf-grid-2">
      <div>
        <h3 style="font-size:1.2cqw">1. Convolutional Autoencoders</h3><div class="pdf-muted pdf-small">Paradigm: Generative Reconstruction · Built from scratch</div>
        <div class="pdf-flow" style="margin:3.2cqh 0"><div class="pdf-box">Input x</div><div class="pdf-arrow">→</div><div class="pdf-box">Encoder</div><div class="pdf-arrow">→</div><div class="pdf-box">z</div><div class="pdf-arrow">→</div><div class="pdf-box">Decoder</div><div class="pdf-arrow">→</div><div class="pdf-box">x̂</div></div>
        <ul><li><b>Complete control:</b> latent bottleneck, skip connections, and loss constraints.</li><li><b>Intuition:</b> trained on normal images; unable to reconstruct defects.</li><li><b>Trade-offs:</b> prone to blurriness or identity shortcuts; threshold sensitive.</li></ul>
      </div>
      <div>
        <h3 style="font-size:1.2cqw">2. PatchCore + ResNet-18 (Anomalib)</h3><div class="pdf-muted pdf-small">Paradigm: Deep Feature Memory Bank · High accessibility</div>
        <div class="pdf-flow" style="margin:3.2cqh 0"><div class="pdf-box coral">Input x</div><div class="pdf-arrow">→</div><div class="pdf-box">Frozen ResNet-18</div><div class="pdf-arrow">→</div><div class="pdf-box">Coreset M (10%)</div><div class="pdf-arrow">→</div><div class="pdf-box coral">Distance d*</div></div>
        <ul><li><b>High accessibility:</b> zero model training required; out-of-the-box integration via Intel's anomalib.</li><li><b>ResNet-18 backbone:</b> compact ImageNet representations; mid-level layers capture both texture and shape.</li><li><b>Coreset sampling:</b> compresses the memory bank by 90%+ without losing local anomaly detection accuracy.</li></ul>
      </div>
    </div>"""
    beamer_slide_html("Model Selection &amp; Architecture Overview", body, frame_number=5, extra_class="balanced")


def render_patchcore() -> None:
    """Render PatchCore transfer-learning architecture."""
    visual = PDF_FIGURES / "report_figures" / "fig_v2_localization_examples.png"
    score_math = math_html(r"s(p^*)=\min_{m\in\mathcal{M}}\lVert p^*-m\rVert_2")
    body = f"""
    <div class="pdf-grid-2" style="grid-template-columns:.78fr 1.22fr">
      <div><div class="pdf-kicker" style="text-align:center;margin-bottom:1.2cqh">Transfer Learning Paradigm</div>
        <div class="pdf-box" style="text-align:center">Normal Factory Images<br><span class="pdf-tiny">(0 defect labels)</span></div><div class="pdf-arrow" style="text-align:center">↓</div>
        <div class="pdf-box" style="text-align:center">Frozen Feature Extractor<br><span class="pdf-tiny">ResNet-18 (ImageNet)</span></div><div class="pdf-arrow" style="text-align:center">↓</div>
        <div class="pdf-box" style="text-align:center">Coreset Memory Bank<br><span class="pdf-tiny">Greedy subsampling (~10×)</span></div><div class="pdf-arrow" style="text-align:center">↓</div>
        <div class="pdf-box coral" style="text-align:center">Inference (Nearest-Neighbour)<br><span class="pdf-tiny">{score_math}</span></div>
      </div>
      <div style="display:grid;grid-template-rows:22cqh auto 1fr;gap:1cqh"><div>{img(visual, 'PatchCore anomaly localisations', 'pdf-image')}</div><div class="pdf-tiny pdf-muted" style="text-align:center">PatchCore anomaly localisations - defect contours outlining identified anomaly regions.</div><div><b>Key Features</b><ul><li><b>Cold-start eliminated:</b> deployment without defect collection.</li><li><b>IP Protection:</b> memory bank contains embeddings, not raw model weights.</li></ul><b class="pdf-kicker">Why it excels</b><ul><li>Mid-level layers capture local textures.</li><li>No catastrophic forgetting (frozen backbone).</li><li>ResNet-18 provides predictable GPU memory bounds.</li></ul></div></div>
    </div>"""
    beamer_slide_html("PatchCore: Transfer Learning &amp; Architecture", body, frame_number=6)


def render_cae_learning() -> None:
    """Render CAE learning process."""
    loss_math = math_html(r"\mathcal{L}=\alpha(1-\mathrm{SSIM})+(1-\alpha)\mathrm{MSE}")
    body = f"""
    <div class="pdf-grid-2">
      <div><h3 class="pdf-kicker" style="font-size:1.18cqw">The Concept</h3><p style="margin-top:1.5cqh"><b>Training Phase:</b></p><ul><li>Trained exclusively on defect-free images.</li><li>Learns the structural grammar of "normal" surfaces.</li></ul><p style="margin-top:2cqh"><b>Inference Phase:</b></p><ul><li>On defective images, anomalies are reconstructed as "normal" parts.</li><li>Reconstruction fails at defect sites ⇒ high reconstruction error.</li><li>High reconstruction error ⇒ anomaly signal.</li></ul></div>
      <div><h3 class="pdf-kicker" style="font-size:1.18cqw">Key Design Decisions</h3><ul style="margin-top:1.5cqh"><li><b>Masked Image Modelling (MIM):</b><br><span class="pdf-small">25% of patches randomly masked before encoding. This forces global context and prevents identity mapping.</span></li><li><b>SSIM + MSE Loss:</b><br><span class="pdf-small">{loss_math} MSE captures global variance, while SSIM enforces structural and perceptual fidelity.</span></li><li><b>Architectural Choices:</b><br><span class="pdf-small">ELU avoids dying neurons and provides a zero-centered mean. BatchNorm stabilizes covariate shifts. AdamW decouples weight decay.</span></li></ul></div>
    </div>"""
    beamer_slide_html("Keras CAE - Autoencoder Learning Process", body, frame_number=7, extra_class="balanced")


def render_cae_pipeline() -> None:
    """Render the end-to-end CAE pipeline."""
    body = """
    <div class="pdf-pipeline-layout">
      <div class="pdf-pipeline-map" role="img" aria-label="Nine-step CAE pipeline: data, preprocessing, category-dependent augmentation, masking, model construction, training, scoring, thresholding, evaluation">
        <svg class="pdf-pipeline-lines" viewBox="0 0 1000 381.5" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
          <defs><marker id="pdf-pipeline-arrow" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 Z" /></marker></defs>
          <path class="arrow" d="M210 89.65 H250" /><path class="arrow" d="M480 89.65 H520" />
          <path class="arrow" d="M690 89.65 H710 V32.43 H730" /><path class="arrow" d="M710 89.65 V143.06 H730" />
          <path d="M960 32.43 H985 V234.62 M960 143.06 H985" /><path class="arrow" d="M985 234.62 H960" />
          <path class="arrow" d="M730 234.62 H610" /><path class="arrow" d="M370 234.62 H250" />
          <path class="arrow" d="M135 267.05 V312.83" /><path class="arrow" d="M250 345.26 H370" /><path class="arrow" d="M610 345.26 H730" />
        </svg>
        <div class="pdf-pipeline-node data"><strong>1. MVTec Data</strong></div>
        <div class="pdf-pipeline-node prep"><strong>2. Preprocessing</strong><span>Otsu, CLAHE, Blur</span></div>
        <div class="pdf-pipeline-node category accent"><strong>3. Category?</strong></div>
        <div class="pdf-pipeline-node texture tint"><strong>Texture</strong><span>Rotate + Flip</span></div>
        <div class="pdf-pipeline-node object accent-tint"><strong>Object</strong><span>Colour + Noise</span></div>
        <div class="pdf-pipeline-node masking"><strong>4. Masking</strong><span>MIM Input Drop</span></div>
        <div class="pdf-pipeline-node build tint"><strong>5. Build CAE</strong><span>ELU + BN + AdamW</span></div>
        <div class="pdf-pipeline-node train strong"><strong>6. Train Model</strong><span>Loss: SSIM + MSE</span></div>
        <div class="pdf-pipeline-node score"><strong>7. Score Images</strong><span>Top-K Pooling</span></div>
        <div class="pdf-pipeline-node threshold"><strong>8. Adapt. Thresh</strong><span>Quantile / Mahal.</span></div>
        <div class="pdf-pipeline-node evaluation accent-tint"><strong>9. Evaluation</strong><span>AUROC + AUPIMO</span></div>
      </div>
      <div class="pdf-pipeline-note"><b>Automated Pipeline &amp; Optuna Integration</b><p>The pipeline is dynamically orchestrated. Hyperparameters (Preprocessing/Model), such as augmentation strategy and latent dimensions, are optimized using Optuna targeting <b>Pixel AUPIMO</b>.</p></div>
    </div>"""
    beamer_slide_html("Keras CAE - End-to-End Pipeline", body, frame_number=8)


def render_evaluation_section() -> None:
    """Render the evaluation divider."""
    beamer_section_html("Evaluation Protocol &amp;<br>Benchmarks")


def render_protocol() -> None:
    """Render the zero-leakage evaluation protocol."""
    alpha_math = math_html(r"\alpha=0.05")
    fpr_math = math_html(r"\mathrm{FPR}\in[10^{-5},10^{-4}]")
    body = f"""
    <div class="pdf-grid-2">
      <div><b>The Core Principle</b><p style="margin-top:1.6cqh">Every evaluation result in this presentation is produced under a deterministic, audit-compliant split that keeps the test set completely unseen until final scoring ⇒ <b>zero-test-leakage</b>.</p>
      <div style="margin-top:4cqh"><b>Peak Performance (PatchCore vs CAE)</b><table class="pdf-table" style="margin-top:1cqh"><tr><th>Category</th><th>Metric</th><th>PatchCore</th><th>Keras CAE</th></tr><tr><td rowspan="2">Bottle</td><td>Image F1</td><td>0.962</td><td><b>0.992</b></td></tr><tr><td>AUPIMO</td><td><b>0.982</b></td><td>0.507</td></tr><tr><td rowspan="2">Macro Avg.</td><td>Image F1</td><td><b>0.929</b></td><td>0.495</td></tr><tr><td>AUPIMO</td><td><b>0.603</b></td><td>0.058</td></tr></table></div></div>
      <div><b>Canonical Evaluation Settings</b><ul style="margin-top:1.6cqh"><li><b>Split Size:</b> 85% fitting / 15% validation</li><li><b>Test Set:</b> held out - normal + anomalies</li><li><b>Resolution:</b> 256 × 256 pixels<ul><li>Bilinear interpolation for heatmaps.</li><li>Nearest-neighbor + binarization for GT masks.</li></ul></li><li><b>Threshold:</b> 95th percentile ({alpha_math} false-alarm budget).</li><li><b>AUPIMO:</b> {fpr_math}, 50k thresholds.</li></ul></div>
    </div>"""
    beamer_slide_html("Zero-Leakage Evaluation Protocol", body, frame_number=9, extra_class="balanced")


def _comparison_row(category: str) -> str:
    """Build one PDF-equivalent localisation comparison row."""
    root = PDF_FIGURES / "extracted" / category
    return f"""
    <div class="pdf-comparison-row" style="display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.65cqw;height:24cqh;min-height:0">
      <div>{img(root / 'patchcore' / 'original.png', f'{category} input', 'pdf-image')}</div><div>{img(root / 'patchcore' / 'gt_mask.png', f'{category} ground truth', 'pdf-image')}</div>
      <div>{img(root / 'patchcore' / 'anomaly_map.png', f'{category} PatchCore map', 'pdf-image')}</div><div>{img(root / 'patchcore' / 'predicted_mask.png', f'{category} PatchCore mask', 'pdf-image')}</div>
      <div>{img(root / 'keras_cae' / 'anomaly_map.png', f'{category} CAE map', 'pdf-image')}</div><div>{img(root / 'keras_cae' / 'gt_overlay.png', f'{category} CAE overlay', 'pdf-image')}</div>
    </div>"""


def render_qualitative() -> None:
    """Render the qualitative localisation comparison."""
    body = f"""
    <div style="display:grid;grid-template-rows:auto 24cqh 24cqh auto;align-content:start;gap:1.4cqh;height:100%;min-height:0">
      <div class="pdf-label" style="display:grid;grid-template-columns:.16fr .16fr .32fr .32fr;gap:.65cqw;text-align:center"><span>Input</span><span>GT Mask</span><span class="pdf-kicker">PatchCore (Map | Mask)</span><span class="pdf-kicker">Keras CAE (Map | GT Overlay)</span></div>
      {_comparison_row('bottle')}{_comparison_row('cable')}
      <div class="pdf-small"><b>Key Observations:</b><ul><li><b>PatchCore:</b> Heatmaps sharply bound the defect, leading to precise predicted masks.</li><li><b>Keras CAE:</b> Reconstruction errors diffuse and fail at fine-grained pixel localization.</li></ul></div>
    </div>"""
    beamer_slide_html("Qualitative Results - Localisation Comparison", body, frame_number=10)


def render_business_section() -> None:
    """Render the business interpretation divider."""
    beamer_section_html("Business Interpretation")


def render_cost_model() -> None:
    """Render economic error asymmetry and the cost model."""
    fp_dominates = math_html(r"c_{\mathrm{FP}}\gg c_{\mathrm{FN}}")
    fn_dominates = math_html(r"c_{\mathrm{FN}}\gg c_{\mathrm{FP}}")
    threshold_math = math_html(r"t^*=\frac{c_{\mathrm{FP}}}{c_{\mathrm{FP}}+c_{\mathrm{FN}}}")
    f_half = math_html(r"F_{0.5}")
    f_one = math_html(r"F_1")
    f_two = math_html(r"F_2")
    four_exp = math_html(r"\approx4\times10^{-6}")
    ten_exp = math_html(r"\approx10^{-8}")
    body = f"""
    <div class="pdf-grid-2" style="grid-template-columns:.95fr 1.05fr">
      <div><b>Two types of error, very different costs</b><div class="pdf-box coral" style="margin-top:1.5cqh"><b class="pdf-accent">False Alarm (FP):</b> good part rejected.<br><span class="pdf-small">Cost: line stoppage, lost throughput, scrapped unit<br><i>e.g. Screw: {fp_dominates}</i></span></div><div class="pdf-box" style="margin-top:1.2cqh"><b class="pdf-kicker">Missed Defect (FN):</b> bad part ships<br><span class="pdf-small">Cost: recall, liability, safety incident<br><i>e.g. Pill: {fn_dominates}</i></span></div><p style="margin-top:2.2cqh"><b>Optimal threshold: {threshold_math}</b></p><p class="pdf-small">High false-negative cost pushes the threshold toward zero; high false-positive cost pushes it toward one.</p></div>
      <div><table class="pdf-table"><tr><th>Category</th><th>Objective</th><th>{math_html(r't^*')}</th></tr><tr><td>Fasteners (Screw)</td><td>{f_half}</td><td>≈ 0.9999</td></tr><tr><td>Packaging (Bottle)</td><td>{f_one}</td><td>≈ 0.0003</td></tr><tr><td>Automotive (M. Nut)</td><td>{f_one}</td><td>{four_exp}</td></tr><tr><td>Life Sci. (Pill)</td><td>{f_two}</td><td>{ten_exp}</td></tr></table><p class="pdf-small pdf-muted" style="margin-top:2.4cqh">Cost profiles are illustrative estimates from composite industry case studies. They demonstrate the spectrum of liability, not precise financial figures.</p></div>
    </div>"""
    beamer_slide_html("Economic Error Asymmetry - The Cost Model", body, frame_number=11, extra_class="balanced")


def render_thresholds() -> None:
    """Render threshold calibration and risk tiers."""
    chart = PDF_FIGURES / "business" / "business_threshold_tiers.png"
    calibration_math = math_html(
        r"\tau_\alpha=\mathrm{Quantile}_{1-\alpha}\left(\{s_i\mid x_i\in\mathcal{D}_{\mathrm{val,norm}}\}\right)",
        display=True,
    )
    life_safety_math = math_html(r"\alpha\approx0.10,\ F_2")
    balanced_math = math_html(r"\alpha\approx0.05,\ F_1")
    throughput_math = math_html(r"\alpha\approx0.01,\ F_{0.5}")
    body = f"""
    <div class="pdf-grid-2" style="grid-template-columns:.9fr 1.1fr"><div><b>Calibration formula</b>{calibration_math}<p class="pdf-small">{math_html(r'\alpha')}: plant's tolerable false-alarm budget. Three pre-declared operating profiles:</p><div class="pdf-box coral" style="margin-top:1.2cqh"><b>T1 - Life-Safety</b> ({life_safety_math})<br><span class="pdf-small">Recall weighted 4× over precision. Every borderline part is manually audited.</span></div><div class="pdf-box" style="margin-top:1cqh"><b>T2 - Prec. Industrial</b> ({balanced_math})<br><span class="pdf-small">Balanced Type I / II errors. Reported benchmark.</span></div><div class="pdf-box neutral" style="margin-top:1cqh"><b>T3 - High-Throughput</b> ({throughput_math})<br><span class="pdf-small">Precision weighted 4×; prevents pseudo-scrap.</span></div></div>
      <div style="display:grid;grid-template-rows:30cqh 1fr;gap:1.4cqh"><div>{img(chart, 'Risk-tiered threshold profiles', 'pdf-image')}</div><ul><li><b>Schematic:</b> profiles select different quantiles from the same normal-validation distribution.</li><li><b>Key:</b> decoupling features from calibration makes the decision gate dynamic and auditable.</li></ul></div></div>"""
    beamer_slide_html("Threshold Calibration &amp; Risk-Tiered Operating Profiles", body, frame_number=12, extra_class="balanced")


def render_sla() -> None:
    """Render SLA interpretation and operational value."""
    visual = PDF_FIGURES / "business" / "business_explainability.png"
    alpha_math = math_html(r"\alpha")
    body = f"""
    <div class="pdf-grid-2" style="grid-template-columns:1.18fr .82fr"><div><b>From academic metric to operational SLA</b><table class="pdf-table" style="margin-top:1cqh"><tr><th>Metric</th><th>Operational meaning</th></tr><tr><td>Image F1 = 0.929</td><td>&lt;8% of decisions are wrong</td></tr><tr><td>Recall = 0.932</td><td>≤6.8% of real defects missed</td></tr><tr><td>Precision = 0.936</td><td>≤6.4% of rejected parts are good</td></tr><tr><td>PR-AUC = 0.988</td><td>Model quality across all thresholds</td></tr><tr><td>AUPIMO = 0.603</td><td>Strong pixel-localisation</td></tr></table><div class="pdf-kicker" style="font-weight:750;margin-top:2.4cqh">Day-1 Deployment &amp; Cold-Start SLAs</div><p class="pdf-small" style="margin-top:.8cqh">No defect samples required at commissioning; onboarding uses only normal parts.</p><p class="pdf-small"><b>The {alpha_math} mechanism:</b> without defect data, we cannot optimize F-scores. Day-1 SLA guarantees a maximum false-alarm rate on normal parts - a reliable proxy for target F-score.</p></div>
      <div style="display:grid;grid-template-rows:25cqh auto 1fr;gap:1cqh"><div>{img(visual, 'Explainable anomaly heatmaps', 'pdf-image')}</div><div class="pdf-tiny pdf-muted">Anomaly heatmaps provide pixel-level attribution, enabling rapid quality-engineer review.</div><div><b class="pdf-accent">Data sovereignty</b><p class="pdf-small" style="margin-top:.7cqh">Coreset memory bank stores patterns locally. CAD toolpaths and material formulations never leave the plant.</p></div></div></div>"""
    beamer_slide_html("SLA Interpretation &amp; Operational Value", body, frame_number=13, extra_class="balanced")


def render_summary() -> None:
    """Render summary and future prospects."""
    body = """
    <div class="pdf-summary-layout"><div class="pdf-grid-2"><div><b class="pdf-kicker">Results</b><ul><li>Unsupervised anomaly detection on MVTec AD 2D (15 categories)</li><li>Two architectures: Keras CAE (generative) &amp; PatchCore (transfer learning)</li><li>Zero-leakage protocol; AUPIMO as primary pixel metric</li><li>Business-grade threshold framework</li></ul><b class="pdf-kicker">Key design principles</b><ul><li>Cold-start: normal-only training, no defect labels</li><li>IP protection: local coreset, no cloud upload</li><li>Calibrated decision gate as financial parameter</li></ul></div>
      <div><b class="pdf-muted">Future Prospects &amp; Experimental Work</b><ul><li><b>Advanced CAE Architectures:</b><ul><li><b>Targeted MIM:</b> masked image modeling for object geometry, dual-weight masked loss computation.</li><li><b>Perceptual Loss:</b> ResNet-18 feature-space penalization.</li><li><b>Multi-scale SSIM:</b> enhanced structural defect isolation.</li></ul></li><li><b>Self-supervised ViTs (DINOv2/v3):</b> foundation-model dense tokens breaking the convolutional representation ceiling.</li></ul></div></div>
      <div class="pdf-summary-close"><b>Thank you</b><span>Questions welcome</span><small>B. Förster · A. Abul Hawa · M. Nadaf · K. Khalifa</small></div>
    </div>"""
    beamer_slide_html("Summary &amp; Future Prospects", body, frame_number=14)
