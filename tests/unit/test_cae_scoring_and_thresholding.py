import numpy as np

from app.pipelines.multi_stage_ae.scoring import compute_adaptive_threshold, compute_pixel_error_map, top_k_pooling


def test_compute_pixel_error_map() -> None:
    """Test generating and smoothing a pixel error map."""
    h, w = 32, 32
    original = np.random.rand(h, w, 3).astype(np.float32)
    reconstruction = original.copy()

    # Add a strong defect in one corner
    reconstruction[0:5, 0:5, :] = 1.0
    original[0:5, 0:5, :] = 0.0

    error_map = compute_pixel_error_map(original, reconstruction, alpha=0.84, sigma=2.0)

    # Check shape
    assert error_map.shape == (h, w)

    # Check bounds
    assert np.all(error_map >= 0.0)

    # Since there's a strong defect in the top-left corner, those pixels should have higher error
    assert error_map[2, 2] > error_map[20, 20]


def test_top_k_pooling() -> None:
    """Test top-k score pooling across different K values."""
    h, w = 32, 32
    error_map = np.zeros((h, w), dtype=np.float32)
    error_map[10, 10] = 5.0  # Localized spike
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
    """Test decision boundary calibration methods."""
    normal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    # Quantile method
    thresh_quantile = compute_adaptive_threshold(normal_scores, method="quantile")
    assert np.isclose(thresh_quantile, np.percentile(normal_scores, 95))

    # Mahalanobis method
    thresh_mahala = compute_adaptive_threshold(normal_scores, method="mahalanobis")
    mu = np.mean(normal_scores)
    sigma = np.std(normal_scores)
    assert np.isclose(thresh_mahala, mu + 3.0 * sigma)

    # Degenerate inputs (zero variance)
    degenerate_scores = np.array([0.5, 0.5, 0.5])
    thresh_degen = compute_adaptive_threshold(degenerate_scores, method="mahalanobis")
    assert not np.isnan(thresh_degen)
    assert not np.isinf(thresh_degen)
