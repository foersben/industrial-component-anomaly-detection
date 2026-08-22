"""Unit tests for CAE evaluation metrics (AUROC, AUPIMO) and explainability visual overlays.

This module validates image-level classification metrics, pixel-level localization scores,
error heatmap synthesis from autoencoder residual deviations, and blended contour overlays.
"""

from pathlib import Path
from typing import Any

import numpy as np

from app.pipelines.multi_stage_ae.error_heatmap import compute_error_heatmap, overlay_ground_truth, overlay_heatmap
from app.pipelines.multi_stage_ae.evaluation import (
    compute_aupimo,
    compute_image_auroc,
    evaluate_cae,
)


def test_evaluate_cae_perfect_separation(mock_keras_cae: Any, tmp_path: Path) -> None:
    """Verify that evaluate_cae computes AUROC, Precision, Recall, and F1 on separated distributions.

    Args:
        mock_keras_cae: Pre-compiled lightweight CAE model fixture.
        tmp_path: Pytest temporary directory fixture.
    """
    test_images = np.random.rand(4, 32, 32, 3).astype(np.float32)
    test_reconstructed = test_images.copy()
    # Introduce high error into anomalous images
    test_reconstructed[2:, ...] = 1.0 - test_images[2:, ...]

    labels = np.array([0, 0, 1, 1], dtype=int)
    mask_anomaly = np.ones((32, 32), dtype=np.uint8) * 255
    gt_masks: list[np.ndarray | None] = [None, None, mask_anomaly, mask_anomaly]
    threshold = 0.1

    eval_res = evaluate_cae(
        model=mock_keras_cae,
        test_images=test_images,
        test_labels=labels,
        gt_masks=gt_masks,
        threshold=threshold,
        output_dir=tmp_path,
        reconstructions=test_reconstructed,
    )

    assert "auroc" in eval_res
    assert "precision" in eval_res
    assert "recall" in eval_res
    assert "f1_score" in eval_res
    assert isinstance(eval_res["auroc"], float)


def test_compute_image_auroc() -> None:
    """Verify AUROC computation for perfect and inverted anomaly score separation."""
    # Perfect separation
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    auroc = compute_image_auroc(scores, labels)
    assert np.isclose(auroc, 1.0)

    # Inverted separation
    auroc_inv = compute_image_auroc(scores, 1 - labels)
    assert np.isclose(auroc_inv, 0.0)


def test_compute_aupimo_no_defects() -> None:
    """Verify that compute_aupimo gracefully returns 0.0 when test samples contain no defects."""
    gt_masks: list[np.ndarray | None] = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
    anomaly_maps = [np.random.rand(32, 32).astype(np.float32) for _ in range(3)]

    results = compute_aupimo(anomaly_maps, gt_masks)

    assert isinstance(results, float)
    assert results == 0.0


def test_compute_error_heatmap(mock_keras_cae: Any) -> None:
    """Verify that compute_error_heatmap produces 2D normalized error heatmaps bounded in [0, 1].

    Args:
        mock_keras_cae: Pre-compiled lightweight CAE model fixture.
    """
    img = np.random.rand(32, 32, 3).astype(np.float32)
    res = compute_error_heatmap(mock_keras_cae, img, sigma=2.0)

    assert "heatmap" in res
    hm = res["heatmap"]
    assert hm.shape == (32, 32)
    assert hm.min() >= 0.0
    assert hm.max() <= 1.0


def test_overlay_heatmap() -> None:
    """Verify that overlay_heatmap blends an RGB image with an anomaly heatmap without shape distortion."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[10:20, 10:20] = 1.0

    overlay = overlay_heatmap(img, heatmap, alpha=0.5)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
    assert np.any(overlay[15, 15] > 0)


def test_overlay_ground_truth() -> None:
    """Verify that overlay_ground_truth renders visible green contour outlines for defect regions."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    gt_mask = np.zeros((32, 32), dtype=np.uint8)
    gt_mask[10:20, 10:20] = 255

    overlay = overlay_ground_truth(img, gt_mask)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
    assert np.sum(overlay[..., 1]) > 0
