"""Reconstruction Error Heatmap explainability for the Keras CAE.

Why not Grad-CAM?
=================
Grad-CAM works by tracing the gradient of the anomaly score back to the last spatial
convolutional layer. This works perfectly for models with Global Average Pooling.
However, our Keras CAE uses a `Flatten()` followed by a `Dense(128)` bottleneck layer.
The Dense layer completely destroys spatial locality — every pixel in the reconstructed
image depends on every feature in the encoder's output. When we backpropagate through
it, the gradients pool indiscriminately, resulting in a giant, useless blob in the
center of the image regardless of where the actual defect is.

The Right Tool: Pixel Reconstruction Error
==========================================
For an Autoencoder, we don't need to guess which features caused the anomaly using
gradients. The Autoencoder *directly outputs* the pixel-wise reconstruction.
The anomaly score is exactly derived from the (MSE) difference between the input image
and the reconstruction.

Therefore, the exact, mathematically faithful "heatmap" of the anomaly is simply the
squared error map itself. We just apply a slight Gaussian blur to make it visually
interpretable (smooth like Grad-CAM) and blend it over the original image.

Module Contents
---------------
- ``compute_error_heatmap``: Main function — returns heatmap for one image.
- ``overlay_heatmap``: Blends a heatmap onto an original image with a colourmap.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.ndimage

logger = logging.getLogger(__name__)


from typing import Any


def compute_error_heatmap(
    model: Any,
    image: np.ndarray,
    sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """Compute a smoothed reconstruction error heatmap for a single image.

    Args:
        model: A compiled ``tf.keras.Model`` produced by ``build_cae()``.
        image: Single normalised image, shape (H, W, 3), float32 values in [0, 1].
        sigma: Standard deviation for the Gaussian blur (smoothness).

    Returns:
        Dictionary containing:
        - ``"heatmap"``: Normalised error heatmap, shape (H, W), float32 in [0, 1].
          Higher values = regions with greater reconstruction error.
    """
    from app.pipelines.multi_stage_ae.scoring import compute_pixel_error_map

    image_batch = np.expand_dims(image, 0)  # (1, H, W, 3)

    # 1. Forward pass to get reconstruction
    reconstruction = model.predict(image_batch, verbose=0)[0]

    # 2. Pixel-wise MAE error (matching the exact scoring logic)
    pixel_error = compute_pixel_error_map(image, reconstruction)

    # 3. Smooth with Gaussian filter for visual appeal
    heatmap = scipy.ndimage.gaussian_filter(pixel_error, sigma=sigma)

    # 4. Normalise robustly (1st-99th percentile) to [0, 1]
    p_low = float(np.percentile(heatmap, 1))
    p_high = float(np.percentile(heatmap, 99))

    if abs(p_high - p_low) > 1e-8:
        heatmap_norm = np.clip((heatmap - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    logger.info(
        "Heatmap complete. Range: [%.4f, %.4f], Quantiles: [%.4f, %.4f]",
        heatmap.min(),
        heatmap.max(),
        p_low,
        p_high,
    )
    return {"heatmap": heatmap_norm.astype(np.float32)}


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,  # Reduced from 0.55 for more transparency (better visibility of the original part)
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap onto the original image using a perceptual colourmap.

    The heatmap is converted from greyscale → RGB via a colourmap (jet by default),
    then composited over the original image. Opacity is scaled per-pixel by the
    heatmap magnitude so regions with near-zero activation show the original image
    unchanged, while highly activated regions show a vivid colour tint.

    Args:
        original_image: RGB image, shape (H, W, 3), uint8 values in [0, 255].
        heatmap: Normalised heatmap, shape (H, W), float32 in [0, 1].
        alpha: Maximum overlay opacity for the highest-activation pixels.
            Default 0.35 keeps the original image clearly visible beneath the anomaly.
        colormap: Matplotlib colourmap name applied to the heatmap.

    Returns:
        RGB overlay image, shape (H, W, 3), uint8.
    """
    import matplotlib  # Lazy import

    cmap = matplotlib.colormaps[colormap]
    heatmap_rgb = cmap(heatmap)[..., :3]  # (H, W, 3), float64 in [0, 1]
    heatmap_rgb = heatmap_rgb.astype(np.float32)

    # Per-pixel alpha: proportional to heatmap magnitude
    pixel_alpha = (heatmap * alpha)[..., np.newaxis]  # (H, W, 1)

    orig_norm = original_image.astype(np.float32) / 255.0
    blended = pixel_alpha * heatmap_rgb + (1.0 - pixel_alpha) * orig_norm
    blended = np.clip(blended, 0.0, 1.0)

    return (blended * 255).astype(np.uint8)
