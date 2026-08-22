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
    """Test full CAE evaluation metrics with clear normal vs anomalous separation."""
    test_images = np.random.rand(4, 32, 32, 3).astype(np.float32)
    test_reconstructed = test_images.copy()
    # Introduce error into anomalous images
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
    """Test AUROC computation for image-level anomaly scores."""
    # Perfect separation
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    auroc = compute_image_auroc(scores, labels)
    assert np.isclose(auroc, 1.0)

    # Inverted separation
    auroc_inv = compute_image_auroc(scores, 1 - labels)
    assert np.isclose(auroc_inv, 0.0)


def test_compute_aupimo_no_defects() -> None:
    """Test AUPIMO gracefully handles when no ground truth defects are present."""
    # Create empty GT masks
    gt_masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
    anomaly_maps = [np.random.rand(32, 32).astype(np.float32) for _ in range(3)]

    # Should warn and return 0.0, but not crash
    results = compute_aupimo(anomaly_maps, gt_masks)

    assert isinstance(results, float)
    assert results == 0.0


def test_compute_error_heatmap(mock_keras_cae: Any) -> None:
    """Test error heatmap generation."""
    img = np.random.rand(32, 32, 3).astype(np.float32)
    res = compute_error_heatmap(mock_keras_cae, img, sigma=2.0)

    assert "heatmap" in res
    hm = res["heatmap"]
    assert hm.shape == (32, 32)
    assert hm.min() >= 0.0
    assert hm.max() <= 1.0


def test_overlay_heatmap() -> None:
    """Test overlaying a heatmap on an image."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[10:20, 10:20] = 1.0  # High error region

    overlay = overlay_heatmap(img, heatmap, alpha=0.5)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8

    # The high error region should have a different color than the background
    assert np.any(overlay[15, 15] > 0)


def test_overlay_ground_truth() -> None:
    """Test overlaying ground truth contours."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    gt_mask = np.zeros((32, 32), dtype=np.uint8)
    gt_mask[10:20, 10:20] = 255

    overlay = overlay_ground_truth(img, gt_mask)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8

    # There should be non-zero green pixels where the contour was drawn
    # The contour is drawn on the boundary of the 10:20 region
    assert np.sum(overlay[..., 1]) > 0
