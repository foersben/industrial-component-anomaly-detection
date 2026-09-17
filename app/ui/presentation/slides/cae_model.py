"""Generative Reconstruction: Keras Convolutional Autoencoder."""

# ruff: noqa: E501, RUF001

from app.ui.presentation.theme import DEFECT_EXAMPLES, img, slide_html


def render() -> None:
    """Explain the Keras CAE architecture and why generative reconstruction fails on textures."""
    reconstruction_fig = DEFECT_EXAMPLES / "cae_localization_examples.png"
    body = f"""
    <div class="slide-grid-2" style="grid-template-columns:1.05fr .95fr;gap:2.6cqw;height:100%;align-items:center">
      <div>
        <div style="display:flex;gap:.8cqw;margin-bottom:1.2cqh">
          <span class="badge badge-coral">From-Scratch Baseline</span>
          <span class="badge">MIM + SSIM/MSE</span>
          <span class="badge">24x Compression</span>
        </div>
        <div style="font-size:1.45cqw;font-weight:780;line-height:1.2;color:var(--ink)">
          Keras CAE & The Texture Reconstruction Dilemma
        </div>
        <div class="def-rule" style="margin:1.4cqh 0"></div>
        <div class="card-box" style="margin-bottom:1.2cqh">
          <div style="font-size:1cqw;line-height:1.4;color:var(--ink)">
            <b>Architecture:</b> 4-stage Conv/Deconv on overlapping 64×64 crops (stride 32), bottleneck 32×4×4. Trained with Masked Image Modeling and composite SSIM+MSE loss (Bergmann et al.).
          </div>
        </div>
        <div class="card-box" style="background:#fdf2ee;border-color:rgba(240,108,84,.3)">
          <div style="font-size:1cqw;line-height:1.4;color:var(--ink)">
            <b>The Fatal Failure Mode:</b> Decoders minimize global pixel variance. On stochastic textures (Wood, Carpet, Leather), subtle reconstruction blur inflates residuals on <i>normal</i> surfaces &rarr; severe <b>pseudo-scrap</b> (Macro Image F1: <b>0.495</b>, AUPIMO: <b>0.058</b>).
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-rows:auto 1fr;gap:1cqh">
        <div class="tiny-label">Input vs. Ground Truth vs. CAE Heatmap vs. Predicted Mask</div>
        <div class="image-frame" style="height:44cqh;background:#071117">
          {img(reconstruction_fig, "CAE localization examples showing residual errors", "photo contain")}
        </div>
      </div>
    </div>
    <div class="source">Source: Technical Report Section 6 · Keras CAE Formulation & Failure Mode Analysis</div>
    """
    slide_html("Generative Reconstruction: Keras CAE", body, eyebrow="06 · Technical Architecture")
