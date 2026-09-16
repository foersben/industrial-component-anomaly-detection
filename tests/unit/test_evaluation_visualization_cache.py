"""Tests for persistent evaluation-chart caching."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
