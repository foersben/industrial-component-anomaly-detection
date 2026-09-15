"""Evaluation metrics and heatmap generation for the Keras CAE pipeline.

Why Evaluation Methodology Matters
=====================================
Getting the right evaluation metric is as important as the model itself.
Using the wrong metric can make a terrible detector look great on paper,
and vice versa.

Image-Level: AUROC (Area Under ROC Curve)
------------------------------------------
AUROC measures how well the model ranks anomalous images above normal ones across
ALL possible thresholds simultaneously. An AUROC of 1.0 means perfect ranking;
0.5 means the model is no better than random guessing.

Advantages over plain accuracy:
- Threshold-independent: does not require choosing a specific cut-off.
- Handles class imbalance well (MVTec test sets are typically imbalanced).

Pixel-Level: AUPIMO vs. PRO-Score
------------------------------------
For pixel-level evaluation (localising *where* the defect is), two metrics exist:

``PRO-Score (Per-Region Overlap / AUPRO)``
    Integrates overlap between predicted anomaly maps and ground-truth masks across
    thresholds up to a fixed FPR on normal images. In practice it over-weights
    tiny label annotation errors, which are common in real industrial datasets.

``AUPIMO (Area Under Per-Image Overlap)`` ← Used here
    AUPIMO introduces two critical improvements:

    1. **Normal-Only Validation**: Thresholds are calibrated exclusively on images
       with zero defects. This prevents the metric from being "gamed" by correctly
       identifying easy normal regions.

    2. **Logarithmic FPR Bounds**: Integration happens only between FPR = 10⁻⁵ and
       FPR = 10⁻⁴. This extremely tight range corresponds to real industrial reject
       rates (maximum 1 false alarm per 10,000-100,000 inspected parts).

    The result is a metric that honestly reflects real industrial performance, not
    laboratory performance under lenient conditions.

Heatmap Generation
------------------
Raw pixel error maps need normalisation before visualisation, because:
- Absolute error values depend on the model's training quality.
- Different images have different baseline error levels.

We use **quantile normalisation**: clamp to the 1st and 99th percentile of the error
distribution, then rescale to [0, 255]. This prevents a few outlier pixels from
washing out the rest of the heatmap.

Module Contents
---------------
- ``compute_image_auroc``: Image-level ROC AUC from scores and binary labels.
- ``compute_aupimo``: Pixel-level AUPIMO using anomalib's implementation.
- ``generate_heatmap_overlay``: Creates an RGB overlay of error on the original image.
- ``evaluate_cae``: Full evaluation pipeline returning all metrics.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Canonical cross-model evaluation metrics are maintained in metrics.py
from app.pipelines.evaluation.metrics import compute_aupimo as compute_aupimo
from app.pipelines.evaluation.metrics import compute_image_auroc

logger = logging.getLogger(__name__)


def generate_heatmap_overlay(
    original_image: np.ndarray,
    error_map: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Generate a colour heatmap overlay of the reconstruction error on the original image.

    The error map is normalised using robust quantile clamping (1st-99th percentile)
    to prevent outlier pixels from dominating the colour scale. The heatmap is then
    blended with the original image.

    Colour scheme:
    - Blue (cool) → low error → likely normal region.
    - Red (warm) → high error → likely anomalous region.

    Args:
        original_image: RGB image as numpy array, shape (H, W, 3), values in [0, 255] uint8.
        error_map: 2D pixel error map, shape (H, W), values ≥ 0.
        alpha: Blend weight for the heatmap overlay (0=original only, 1=heatmap only).
            Default 0.6 → 60% heatmap, 40% original.

    Returns:
        RGB overlay image as numpy array, shape (H, W, 3), uint8.
    """
    import cv2

    # Quantile-normalise: clamp to 1st-99th percentile range to suppress outliers
    p_low = float(np.percentile(error_map, 1))
    p_high = float(np.percentile(error_map, 99))

    if abs(p_high - p_low) < 1e-8:
        # Flat map (no variation) → return original image unchanged
        return original_image.copy()

    normalised = np.clip((error_map - p_low) / (p_high - p_low), 0.0, 1.0)
    heatmap_uint8 = (normalised * 255).astype(np.uint8)

    # Apply OpenCV's JET colormap (blue=low, red=high error)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original image
    original_float = original_image.astype(np.float32)
    heatmap_float = heatmap_rgb.astype(np.float32)
    blended = (alpha * heatmap_float + (1.0 - alpha) * original_float).clip(0, 255).astype(np.uint8)

    return blended


def evaluate_cae(
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    gt_masks: list[np.ndarray | None],
    threshold: float,
    k_fraction: float = 0.002,
    output_dir: Path | None = None,
    reconstructions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run full evaluation of the trained CAE on the test set.

    Computes:
    - Image-level anomaly scores using Top-K pooling.
    - AUROC across all test images.
    - Accuracy, precision, recall using the calibrated adaptive threshold.
    - AUPIMO for pixel-level localisation on images with ground truth masks.

    Args:
        model: Trained Keras CAE model.
        test_images: Normalised test images, shape (N, H, W, 3), values in [0, 1].
        test_labels: Binary labels, shape (N,). 0 = normal, 1 = anomalous.
        gt_masks: List of ground truth defect masks (or None for normal images).
        threshold: Decision threshold from ``compute_adaptive_threshold``.
        k_fraction: Top-K pooling fraction for image-level scoring.
        output_dir: Directory to save detailed PR metrics (.npz files) for UI rendering.
        reconstructions: Optional pre-computed full-image reconstructions.

    Returns:
        Dictionary containing all evaluation results:
        - ``"auroc"``: Image-level AUROC.
        - ``"aupimo"``: Pixel-level AUPIMO.
        - ``"accuracy"``: Classification accuracy at the given threshold.
        - ``"precision"``: Classification precision at the given threshold.
        - ``"recall"``: Classification recall at the given threshold.
        - ``"f1_score"``: Classification F1 score at the given threshold.
        - ``"scores"``: Raw image anomaly scores (numpy array).
        - ``"error_maps"``: List of 2D pixel error maps.
        - ``"threshold"``: The decision threshold used.
    """
    from sklearn.metrics import accuracy_score

    from app.pipelines.evaluation.metrics import (
        AUPIMO_FPR_BOUNDS,
        compute_and_save_pr_metrics,
        compute_image_confusion_metrics,
        compute_shared_pixel_metrics,
    )
    from app.pipelines.evaluation.scoring import compute_image_scores

    logger.info("Running CAE evaluation on %d test images...", len(test_images))

    scores, error_maps = compute_image_scores(
        model, test_images, k_fraction=k_fraction, reconstructions=reconstructions
    )
    binary_labels = test_labels.astype(int)

    auroc = compute_image_auroc(scores, binary_labels)
    pixel_metrics, canonical_maps, canonical_masks = compute_shared_pixel_metrics(error_maps, gt_masks, binary_labels)
    aupimo = float(pixel_metrics["pixel_aupimo"])

    predictions = (scores > threshold).astype(int)
    acc = float(accuracy_score(binary_labels, predictions))
    confusion = compute_image_confusion_metrics(binary_labels, scores, threshold)
    prec = float(confusion["precision"])
    rec = float(confusion["recall"])
    f1 = float(confusion["f1_score"])

    logger.info(
        "Evaluation complete — AUROC: %.4f | AUPIMO: %.4f | Acc: %.2f%% | Prec: %.2f | Rec: %.2f",
        auroc,
        aupimo,
        acc * 100,
        prec,
        rec,
    )

    pixel_auroc = float(pixel_metrics["pixel_auroc"])
    pixel_f1 = 0.0
    if output_dir:
        compute_and_save_pr_metrics(binary_labels, scores, output_dir / "image_metrics.npz", level="image")

        y_true_pixel = canonical_masks.reshape(-1)
        y_score_pixel = canonical_maps.reshape(-1)

        compute_and_save_pr_metrics(
            y_true_pixel,
            y_score_pixel,
            output_dir / "pixel_metrics.npz",
            level="pixel",
            aupimo=aupimo,
            fpr_bounds=AUPIMO_FPR_BOUNDS,
        )

        from sklearn.metrics import precision_score, recall_score

        pixel_pred = (y_score_pixel > threshold).astype(int)
        pixel_prec = float(precision_score(y_true_pixel, pixel_pred, zero_division=0))
        pixel_rec = float(recall_score(y_true_pixel, pixel_pred, zero_division=0))
        pixel_f1 = 2 * (pixel_prec * pixel_rec) / (pixel_prec + pixel_rec) if (pixel_prec + pixel_rec) > 0 else 0.0

    return {
        "auroc": auroc,
        "aupimo": aupimo,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "pixel_auroc": pixel_auroc,
        "pixel_f1": pixel_f1,
        "scores": scores,
        "error_maps": error_maps,
        "threshold": threshold,
        **confusion,
        **pixel_metrics,
    }
