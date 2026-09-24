"""Tests for persistent evaluation-chart caching."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from app.pipelines.evaluation.metrics import MAX_PERSISTED_PIXEL_CURVE_POINTS, save_evaluation_metrics
from app.pipelines.evaluation.visualization import (
    MAX_PLOT_POINTS,
    _downsample_curve,
    _resolve_or_create_curve_images,
    load_and_prepare_evaluation_data,
)


def _write_metrics(path: Path) -> None:
    """Write the smallest representative PR metrics artifact."""
    np.savez(
        path,
        precision=np.array([0.5, 0.8, 1.0]),
        recall=np.array([1.0, 0.6, 0.2]),
        thresholds=np.array([0.2, 0.5, 0.8]),
        level=np.array("image"),
    )


def test_curve_images_are_reused_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing chart files must be reused without invoking plot generation."""
    metrics_path = tmp_path / "image_metrics.npz"
    _write_metrics(metrics_path)
    data = load_and_prepare_evaluation_data(str(metrics_path))

    first_paths = _resolve_or_create_curve_images(str(metrics_path), data)
    assert all(path.is_file() for path in first_paths)

    def fail_if_regenerated(_data: object) -> None:
        raise AssertionError("cached chart was regenerated")

    monkeypatch.setattr("app.pipelines.evaluation.visualization.plot_tradeoff_curve", fail_if_regenerated)
    monkeypatch.setattr("app.pipelines.evaluation.visualization.plot_pr_curve", fail_if_regenerated)
    assert _resolve_or_create_curve_images(str(metrics_path), data) == first_paths

    plt.close("all")


def test_curve_display_is_bounded_without_losing_endpoints() -> None:
    """Large metric arrays should use a bounded display sample only."""
    x = np.arange(MAX_PLOT_POINTS * 3)
    y = x * 2

    sampled_x, sampled_y = _downsample_curve(x, y)

    assert len(sampled_x) == MAX_PLOT_POINTS
    assert sampled_x[0] == x[0]
    assert sampled_x[-1] == x[-1]
    np.testing.assert_array_equal(sampled_y, sampled_x * 2)


def test_pixel_curve_is_compact_but_keeps_exact_summaries(tmp_path: Path) -> None:
    """Pixel caches retain exact summaries while bounding persisted curve points."""
    thresholds = np.linspace(0.0, 1.0, MAX_PERSISTED_PIXEL_CURVE_POINTS * 3)
    precision = np.linspace(0.2, 1.0, len(thresholds) + 1)
    recall = np.linspace(1.0, 0.0, len(thresholds) + 1)
    metrics_path = tmp_path / "pixel_metrics.json"

    save_evaluation_metrics(metrics_path, precision, recall, thresholds, aupimo=0.73, level="pixel")

    import json

    archive = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(archive["thresholds"]) == MAX_PERSISTED_PIXEL_CURVE_POINTS
    assert int(archive["curve_points_full"]) == len(thresholds)
    assert bool(archive["curve_compacted"])
    assert float(archive["aupimo"]) == pytest.approx(0.73)
    exact_aupr = float(archive["integrated_aupr"])
    exact_crossover = float(archive["t_crossover"])

    loaded = load_and_prepare_evaluation_data(str(metrics_path))
    assert loaded.integrated_aupr == pytest.approx(exact_aupr)
    assert loaded.t_crossover == pytest.approx(exact_crossover)
