"""Unit tests for CAE evaluation metrics (AUROC, AUPIMO) and explainability visual overlays.

This module validates image-level classification metrics, pixel-level localization scores,
error heatmap synthesis from autoencoder residual deviations, and blended contour overlays.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.pipelines.evaluation.cae_metrics import (
    compute_aupimo,
    compute_image_auroc,
    evaluate_cae,
)
from app.pipelines.evaluation.heatmaps import compute_error_heatmap, overlay_ground_truth, overlay_heatmap


def test_evaluate_cae_perfect_separation(mock_keras_cae: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that evaluate_cae computes AUROC, Precision, Recall, and F1 on separated distributions.

    Args:
        mock_keras_cae: Pre-compiled lightweight CAE model fixture.
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest fixture used to isolate image-level classification behavior.
    """
    test_images = np.random.rand(4, 32, 32, 3).astype(np.float32)
    test_reconstructed = test_images.copy()
    # Introduce high error into anomalous images
    test_reconstructed[2:, ...] = 1.0 - test_images[2:, ...]

    labels = np.array([0, 0, 1, 1], dtype=int)
    mask_anomaly = np.ones((32, 32), dtype=np.uint8) * 255
    gt_masks: list[np.ndarray | None] = [None, None, mask_anomaly, mask_anomaly]
    threshold = 0.1
    monkeypatch.setattr("app.pipelines.evaluation.cae_metrics.compute_aupimo", lambda *_args, **_kwargs: 0.5)

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
    with np.load(tmp_path / "pixel_metrics.npz") as pixel_metrics:
        assert float(pixel_metrics["aupimo"]) == 0.5
        assert np.array_equal(pixel_metrics["aupimo_fpr_bounds"], np.array([1e-5, 1e-4]))
        assert "t_aupimo_min" not in pixel_metrics


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
    """Verify that compute_aupimo rejects inputs for which the metric is undefined."""
    gt_masks: list[np.ndarray | None] = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
    anomaly_maps = [np.random.rand(32, 32).astype(np.float32) for _ in range(3)]

    with pytest.raises(ValueError, match="at least one anomalous image"):
        compute_aupimo(anomaly_maps, gt_masks)


def test_compute_aupimo_uses_50k_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the shared AUPIMO path uses the documented threshold-grid resolution."""
    import anomalib.metrics

    captured: dict[str, Any] = {}

    class FakeAUPIMO:
        def __init__(self, num_thresholds: int, fpr_bounds: tuple[float, float]) -> None:
            captured["num_thresholds"] = num_thresholds
            captured["fpr_bounds"] = fpr_bounds

        def update(self, _batch: Any) -> None:
            return None

        def compute(self) -> dict[str, float]:
            return {"aupimo": 0.75}

    monkeypatch.setattr(anomalib.metrics, "AUPIMO", FakeAUPIMO)
    anomaly_maps = [np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32)]
    gt_masks = [np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]

    score = compute_aupimo(anomaly_maps, gt_masks, fpr_bounds=(1e-5, 1e-4))

    assert score == 0.75
    assert captured == {"num_thresholds": 50_000, "fpr_bounds": (1e-5, 1e-4)}


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
