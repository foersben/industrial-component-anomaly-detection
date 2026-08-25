"""Anomaly scoring algorithms for the Keras CAE pipeline.

This module implements two key improvements over naive anomaly scoring that make
the system significantly more robust for real-world industrial inspection:

1. Top-K Pooling (replaces Max-Pooling for image-level scoring)
2. Adaptive Thresholding (replaces a fixed hard-coded cut-off)

Why Image-Level Anomaly Scoring Matters
========================================
The autoencoder produces a 2D error map (pixel-wise reconstruction error). To decide
"is this image anomalous?" we need to collapse this map into a single score.

The Naive Approach: Max-Pooling
    Image score = max(error_map)
    Problem: A single noisy pixel (camera sensor spike, dust particle, JPEG artefact)
    can drive the max value very high, producing a false positive on a perfectly good part.

The Better Approach: Top-K Pooling
    Image score = mean(top K highest pixels)
    Rationale: Real industrial defects (cracks, scratches, contamination patches)
    always appear as *clusters* of elevated error pixels, not isolated spikes.
    By averaging the K highest values, isolated single-pixel noise is diluted,
    while genuine defect clusters (which affect many pixels together) still produce
    reliably high scores.

    A commonly effective value is K = 0.2% of total pixels.
    For a 128x128 image = 16,384 pixels -> K ~= 33 pixels.

Why Adaptive Thresholds Are Essential
======================================
A fixed threshold (e.g., "score > 0.05 = anomalous") will fail when:
    - Different cameras / lighting conditions shift the absolute score range.
    - Different MVTec categories (leather vs. metal) have vastly different texture complexity.
    - Batch-to-batch variation in normal samples changes the baseline reconstruction quality.

Adaptive approaches calibrate the threshold on the normal training/validation data:

``quantile`` method:
    threshold = np.percentile(normal_scores, 95)
    Interpretation: "The model is trained; 95% of normal images score below this value.
    Anything higher is likely anomalous."

``mahalanobis`` method:
    Models the normal score distribution as a Gaussian. The threshold is set at
    mean + n_sigma * std. This is more principled than a percentile and is closer to
    a proper statistical test (rejecting the null hypothesis that the image is normal).

Module Contents
---------------
- ``compute_pixel_error_map``: Computes per-pixel reconstruction error.
- ``top_k_pooling``: Aggregates error map to a single image-level score robustly.
- ``compute_image_scores``: Scores an entire dataset using Top-K pooling.
- ``compute_adaptive_threshold``: Derives a decision boundary from normal score statistics.
"""

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def compute_pixel_error_map(
    original: np.ndarray, reconstruction: np.ndarray, alpha: float = 0.84, sigma: float = 2.0
) -> np.ndarray:
    """Compute the per-pixel absolute reconstruction error map.

    The error map is a weighted blend of Structural Similarity (SSIM) error
    and channel-wise Mean Absolute Error (MAE), aligned with the training loss.
    A Gaussian blur is applied to smooth noise and cluster anomaly predictions.

    Args:
        original: Original normalised image, shape (H, W, 3), values in [0, 1].
        reconstruction: Reconstructed image from the CAE, same shape as original.
        alpha: Weight for the SSIM component (default 0.84, matches loss).
        sigma: Standard deviation for Gaussian kernel (default 2.0).

    Returns:
        2D error map, shape (H, W), values ≥ 0. Higher values = more likely anomalous.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.metrics import structural_similarity as ssim

    # 1. Structural Error Map (1 - SSIM)
    _, ssim_map = ssim(original, reconstruction, data_range=1.0, channel_axis=-1, full=True)  # type: ignore[no-untyped-call]
    ssim_error = 1.0 - np.mean(ssim_map, axis=-1)  # (H, W)

    # 2. Pixel Error Map (MAE)
    per_channel_error = np.abs(original - reconstruction)  # (H, W, 3)
    mae_error = np.mean(per_channel_error, axis=-1)  # (H, W)

    # 3. Blend them together
    error_map = alpha * ssim_error + (1.0 - alpha) * mae_error

    # 4. Smooth the result to cluster defect pixels
    if sigma > 0:
        error_map = gaussian_filter(error_map, sigma=sigma)

    return error_map


def top_k_pooling(error_map: np.ndarray, k: int | None = None, k_fraction: float = 0.002) -> float:
    """Compute the image-level anomaly score using Top-K pooling.

    Top-K pooling is significantly more robust than max-pooling because:
    - Single noisy pixels (sensor spikes, JPEG artefacts) produce 1 high pixel.
    - Real defects (scratches, cracks) produce a *cluster* of many high pixels.
    Averaging the top-K pixels dilutes isolated spikes while keeping defect clusters high.

    Args:
        error_map: 2D pixel error map, shape (H, W), values ≥ 0.
        k: Explicit number of top pixels to average. If None, derived from ``k_fraction``.
        k_fraction: Fraction of total pixels to use as K when ``k`` is not specified.
            Default 0.002 = 0.2% of pixels. For 128x128 → K ~= 33 pixels.

    Returns:
        Single float representing the image-level anomaly score. Higher = more anomalous.
    """
    flat = error_map.flatten()
    total_pixels = len(flat)

    if k is None:
        k = max(1, int(total_pixels * k_fraction))

    # Sort descending and take the top-k
    top_k_values = np.partition(flat, -k)[-k:]  # np.partition is O(n), faster than full sort
    return float(np.mean(top_k_values))


def compute_image_scores(
    model: Any,
    images: np.ndarray,
    k_fraction: float = 0.002,
    reconstructions: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute image-level anomaly scores and pixel error maps for a dataset.

    This function runs the trained CAE on every test image, computes the per-pixel
    error map, and aggregates to an image-level score using Top-K pooling.

    Args:
        model: Trained Keras CAE model. Must have a ``predict`` method. Can be None if reconstructions are provided.
        images: Array of normalised test images, shape (N, H, W, 3), values in [0, 1].
        k_fraction: Top-K pooling fraction. See ``top_k_pooling`` for details.
        reconstructions: Optional pre-computed reconstructions. If None, uses model.predict.

    Returns:
        Tuple of:
            - scores: 1D numpy array of image-level anomaly scores, shape (N,).
            - error_maps: List of N 2D error maps, each shape (H, W).
    """
    # Run all images through the CAE in a single batched predict call if not provided
    if reconstructions is None:
        reconstructions = model.predict(images, verbose=0)

    scores: list[float] = []
    error_maps: list[np.ndarray] = []

    for orig, recon in zip(images, reconstructions, strict=True):
        emap = compute_pixel_error_map(orig, recon)
        score = top_k_pooling(emap, k_fraction=k_fraction)
        error_maps.append(emap)
        scores.append(score)

    return np.array(scores, dtype=np.float32), error_maps


def compute_adaptive_threshold(
    normal_scores: np.ndarray,
    method: Literal["quantile", "mahalanobis"] = "quantile",
    quantile: float = 0.95,
    n_sigma: float = 3.0,
) -> float:
    """Compute an adaptive anomaly decision threshold from normal image scores.

    Rather than using a hand-tuned fixed threshold, this function calibrates the
    decision boundary using the statistical distribution of normal image scores.

    Args:
        normal_scores: 1D array of anomaly scores computed on known-good (normal) images.
            These are used as the calibration reference.
        method: Threshold derivation method:
            - ``"quantile"``: Set threshold at the given percentile of normal scores.
              Intuitive and non-parametric. Works well when the score distribution is
              non-Gaussian or has outliers.
            - ``"mahalanobis"``: Fit a Gaussian to normal scores (mean + std), then set
              threshold at ``mean + n_sigma * std``. More statistically principled.
              Assumes the normal score distribution is approximately Gaussian.
        quantile: Percentile to use for the quantile method (0 < quantile < 1).
            Default 0.95 → 95th percentile of normal scores becomes the threshold.
        n_sigma: Number of standard deviations above the mean for Mahalanobis method.
            Default 3.0 → corresponds to a false positive rate of ≈0.13% under Gaussian.

    Returns:
        Threshold float value. Images scoring above this are classified as anomalous.

    Raises:
        ValueError: If ``normal_scores`` is empty or if ``method`` is not recognised.
    """
    if len(normal_scores) == 0:
        raise ValueError("normal_scores must not be empty for threshold calibration.")

    if method == "quantile":
        threshold = float(np.percentile(normal_scores, quantile * 100))
        logger.info("Quantile threshold (%.0f%%): %.6f", quantile * 100, threshold)

    elif method == "mahalanobis":
        mu = float(np.mean(normal_scores))
        sigma = float(np.std(normal_scores))
        if sigma < 1e-8:
            logger.warning("Normal scores have near-zero variance. Using mean as threshold.")
            threshold = mu
        else:
            threshold = mu + n_sigma * sigma
        logger.info(
            "Mahalanobis threshold (mu + %.1f*sigma): %.6f  (mu=%.6f, sigma=%.6f)",
            n_sigma,
            threshold,
            mu,
            sigma,
        )

    else:
        raise ValueError(f"Unknown threshold method '{method}'. Choose 'quantile' or 'mahalanobis'.")

    return threshold
