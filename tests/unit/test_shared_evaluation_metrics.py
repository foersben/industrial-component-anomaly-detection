"""Tests for model-independent fair-evaluation metrics."""

from typing import Any

import numpy as np
import pytest

from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    CANONICAL_MAP_SIZE,
    canonicalize_pixel_inputs,
    compute_image_confusion_metrics,
    compute_shared_pixel_metrics,
)


def test_canonical_inputs_have_shared_shape_and_dtypes() -> None:
    """Maps and masks use the shared shape, interpolation rules, and dtypes."""
    maps = [np.zeros((8, 12)), np.ones((16, 10))]
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255

    canonical_maps, canonical_masks = canonicalize_pixel_inputs(maps, [None, mask], [0, 1])

    assert canonical_maps.shape == (2, *CANONICAL_MAP_SIZE)
    assert canonical_maps.dtype == np.float32
    assert canonical_masks.shape == (2, *CANONICAL_MAP_SIZE)
    assert canonical_masks.dtype == np.uint8
    assert set(np.unique(canonical_masks)) == {0, 1}


def test_shared_pixel_metrics_uses_canonical_full_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pixel AUROC and AUPIMO receive the same canonical full-map inputs."""
    captured: dict[str, Any] = {}

    def fake_aupimo(maps: list[np.ndarray], masks: list[np.ndarray], fpr_bounds: tuple[float, float]) -> float:
        captured.update(maps=maps, masks=masks, bounds=fpr_bounds)
        return 0.75

    monkeypatch.setattr("app.pipelines.evaluation.cae_metrics.compute_aupimo", fake_aupimo)
    anomaly = np.zeros((5, 7), dtype=np.uint8)
    anomaly[2:, 3:] = 1
    metrics, maps, masks = compute_shared_pixel_metrics(
        [np.zeros((5, 7)), anomaly.astype(np.float32)],
        [None, anomaly],
        [0, 1],
    )

    assert maps.shape == masks.shape == (2, 256, 256)
    assert [item.shape for item in captured["maps"]] == [(256, 256), (256, 256)]
    assert captured["bounds"] == AUPIMO_FPR_BOUNDS
    assert metrics["pixel_aupimo"] == 0.75
    assert metrics["pixel_auroc"] > 0.999


def test_missing_anomalous_mask_is_an_error() -> None:
    """Missing anomalous ground truth is an error, never a numeric fallback."""
    with pytest.raises(ValueError, match="missing"):
        canonicalize_pixel_inputs([np.zeros((4, 4))], [None], [1])


def test_image_confusion_metrics_match_hand_calculation() -> None:
    """Image confusion counts and derived metrics match a small manual example."""
    metrics = compute_image_confusion_metrics(
        labels=[0, 0, 1, 1, 1],
        scores=[0.1, 0.8, 0.9, 0.7, 0.2],
        threshold=0.5,
    )

    assert metrics == {
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "f1_score": pytest.approx(2 / 3),
    }
