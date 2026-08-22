"""Unit tests for reconstruction error mapping, Top-K pooling, and adaptive threshold calibration.

This module validates that pixel error maps properly highlight anomalous deviations, that Top-K
percentile pooling correctly aggregates localized error energy, and that quantile and Mahalanobis
statistical thresholds calculate valid decision boundaries.
"""

import numpy as np

from app.pipelines.evaluation.scoring import compute_adaptive_threshold, compute_pixel_error_map, top_k_pooling


def test_compute_pixel_error_map() -> None:
    """Verify that compute_pixel_error_map produces smoothed 2D error maps reflecting localized defects."""
    h, w = 32, 32
    original = np.random.rand(h, w, 3).astype(np.float32)
    reconstruction = original.copy()

    # Add a strong defect in the top-left corner
    reconstruction[0:5, 0:5, :] = 1.0
    original[0:5, 0:5, :] = 0.0

    error_map = compute_pixel_error_map(original, reconstruction, alpha=0.84, sigma=2.0)

    assert error_map.shape == (h, w)
    assert np.all(error_map >= 0.0)
    assert error_map[2, 2] > error_map[20, 20]


def test_top_k_pooling() -> None:
    """Verify that Top-K pooling correctly handles max, mean, zero, and constant error distributions."""
    h, w = 32, 32
    error_map = np.zeros((h, w), dtype=np.float32)
    error_map[10, 10] = 5.0
    error_map[20, 20] = 2.0

    # Max pooling (K=1)
    score_max = top_k_pooling(error_map, k=1)
    assert np.isclose(score_max, 5.0)

    # Mean pooling (K=H*W)
    score_mean = top_k_pooling(error_map, k=h * w)
    expected_mean = (5.0 + 2.0) / (h * w)
    assert np.isclose(score_mean, expected_mean)

    # Uniform zero map
    zero_map = np.zeros((h, w), dtype=np.float32)
    score_zero = top_k_pooling(zero_map, k=10)
    assert np.isclose(score_zero, 0.0)

    # Constant non-zero map
    const_map = np.ones((h, w), dtype=np.float32) * 3.0
    score_const = top_k_pooling(const_map, k=5)
    assert np.isclose(score_const, 3.0)


def test_compute_adaptive_threshold() -> None:
    """Verify adaptive threshold calibration for quantile and Mahalanobis parametric methods."""
    normal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Quantile method (95th percentile)
    thresh_quantile = compute_adaptive_threshold(normal_scores, method="quantile")
    assert np.isclose(thresh_quantile, np.percentile(normal_scores, 95))

    # Mahalanobis method (mu + 3 * sigma)
    thresh_mahala = compute_adaptive_threshold(normal_scores, method="mahalanobis")
    mu = np.mean(normal_scores)
    sigma = np.std(normal_scores)
    assert np.isclose(thresh_mahala, mu + 3.0 * sigma)

    # Degenerate inputs (zero variance)
    degenerate_scores = np.array([0.5, 0.5, 0.5])
    thresh_degen = compute_adaptive_threshold(degenerate_scores, method="mahalanobis")
    assert not np.isnan(thresh_degen)
    assert not np.isinf(thresh_degen)
