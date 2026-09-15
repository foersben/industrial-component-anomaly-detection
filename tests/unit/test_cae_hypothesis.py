"""Property-based tests for CAE pipeline mathematical invariants using Hypothesis.

This module validates tensor transformation invariants, loss symmetries, pooling monotonicity,
adaptive threshold stability, and visual segmentation robustness across generative parameter spaces.
"""

from typing import Literal

import hypothesis.strategies as st
import numpy as np
import tensorflow as tf
from hypothesis import given, settings

from app.pipelines.evaluation.scoring import compute_adaptive_threshold, top_k_pooling
from app.pipelines.modelling.keras_cae.cae_keras import ssim_mse_loss
from app.pipelines.modelling.keras_cae.cae_pipeline import extract_crops, stitch_crops
from app.pipelines.preprocessing.segmentation import OtsuCannySegmentor


@settings(max_examples=20, deadline=None)
@given(
    h=st.integers(min_value=32, max_value=64),
    w=st.integers(min_value=32, max_value=64),
    crop_size=st.integers(min_value=16, max_value=32),
    stride=st.integers(min_value=8, max_value=16),
    n=st.integers(min_value=1, max_value=3),
)
def test_extract_stitch_roundtrip(h: int, w: int, crop_size: int, stride: int, n: int) -> None:
    """Verify that extract_crops followed by stitch_crops perfectly reconstructs input tensors.

    Args:
        h: Synthetic image height.
        w: Synthetic image width.
        crop_size: Spatial dimensions of extracted square patches.
        stride: Step size between consecutive patch origins.
        n: Number of images in the synthetic batch.
    """
    if h < crop_size or w < crop_size:
        return
    if (h - crop_size) % stride != 0 or (w - crop_size) % stride != 0:
        return

    images = np.random.rand(n, h, w, 3).astype(np.float32)

    crops = extract_crops(images, crop_size, stride)
    stitched = stitch_crops(crops, n, h, w, crop_size, stride)

    np.testing.assert_allclose(images, stitched, atol=1e-5)


@settings(max_examples=20, deadline=None)
@given(
    h=st.integers(min_value=16, max_value=48),
    w=st.integers(min_value=16, max_value=48),
    c=st.floats(min_value=0.0, max_value=10.0),
)
def test_top_k_pooling_invariants(h: int, w: int, c: float) -> None:
    """Verify bounds, monotonicity, and scale equivariance of Top-K pooling.

    Args:
        h: Error map height.
        w: Error map width.
        c: Non-negative scalar scaling factor.
    """
    k_val = np.random.randint(1, h * w + 1)
    map_a = np.random.uniform(0.0, 10.0, size=(h, w)).astype(np.float32)

    # Invariant 1: Top-K pooling is bounded between min and max map values
    score_a = top_k_pooling(map_a, k=k_val)
    assert np.min(map_a) <= score_a <= np.max(map_a) + 1e-6

    # Invariant 2: Monotonicity with respect to elementwise additive noise
    map_b = map_a + np.random.uniform(0.0, 2.0, size=(h, w)).astype(np.float32)
    score_b = top_k_pooling(map_b, k=k_val)
    assert score_a <= score_b + 1e-5

    # Invariant 3: Scale equivariance for non-negative multipliers
    score_c = top_k_pooling(map_a * c, k=k_val)
    assert abs(score_c - c * score_a) < 1e-4


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1000))
def test_ssim_mse_loss_invariants(seed: int) -> None:
    """Verify identity of indiscernibles, symmetry, and non-negativity of SSIM+MSE loss.

    Args:
        seed: Random seed for generating synthetic image tensors.
    """
    np.random.seed(seed)

    map_a = np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)
    map_b = np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)

    a_tf = tf.convert_to_tensor(map_a)
    b_tf = tf.convert_to_tensor(map_b)

    loss_fn = ssim_mse_loss()

    # Invariant 1: Identity loss is zero
    loss_aa = loss_fn(a_tf, a_tf).numpy()
    assert float(loss_aa) < 1e-5

    # Invariant 2: Loss is symmetric between inputs
    loss_ab = loss_fn(a_tf, b_tf).numpy()
    loss_ba = loss_fn(b_tf, a_tf).numpy()
    assert abs(float(loss_ab) - float(loss_ba)) < 1e-5

    # Invariant 3: Non-negativity
    assert float(loss_ab) >= 0.0


@settings(max_examples=20, deadline=None)
@given(
    scores=st.lists(st.floats(min_value=0.0, max_value=5.0), min_size=3, max_size=50),
    method=st.sampled_from(["quantile", "mahalanobis"]),
    degenerate=st.booleans(),
)
def test_adaptive_threshold_stability(
    scores: list[float], method: Literal["quantile", "mahalanobis"], degenerate: bool
) -> None:
    """Verify numerical stability and finiteness of calibrated adaptive decision thresholds.

    Args:
        scores: List of simulated normal anomaly scores.
        method: Thresholding method name ('quantile' or 'mahalanobis').
        degenerate: Whether to test a zero-variance distribution.
    """
    if degenerate:
        score_arr = np.full(len(scores), fill_value=scores[0], dtype=np.float32)
    else:
        score_arr = np.array(scores, dtype=np.float32)

    threshold = compute_adaptive_threshold(score_arr, method=method)

    assert np.isfinite(threshold)
    assert threshold >= np.min(score_arr) - 1e-6


@settings(max_examples=20, deadline=None)
@given(image_type=st.sampled_from(["zeros", "ones", "random"]))
def test_segmentation_robustness(image_type: str) -> None:
    """Verify Otsu+Canny foreground segmentation robustness against degenerate visual inputs.

    Args:
        image_type: Type of synthetic image to test ('zeros', 'ones', or 'random').
    """
    if image_type == "zeros":
        image = np.zeros((64, 64, 3), dtype=np.uint8)
    elif image_type == "ones":
        image = np.full((64, 64, 3), fill_value=255, dtype=np.uint8)
    else:
        image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

    segmentor = OtsuCannySegmentor(morph_kernel_size=3)
    masked_image, mask = segmentor.apply(image)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, [0, 255]))
    assert masked_image.shape == (64, 64, 3)
