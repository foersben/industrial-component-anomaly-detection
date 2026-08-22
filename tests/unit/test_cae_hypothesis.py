import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from app.pipelines.multi_stage_ae.cae_keras import ssim_mse_loss
from app.pipelines.multi_stage_ae.cae_pipeline import extract_crops, stitch_crops
from app.pipelines.multi_stage_ae.scoring import compute_adaptive_threshold, top_k_pooling
from app.pipelines.multi_stage_ae.segmentation import OtsuCannySegmentor


# 1. Crop Extraction & Stitching Roundtrip Invariant
@settings(max_examples=20, deadline=None)
@given(
    h=st.integers(min_value=32, max_value=64),
    w=st.integers(min_value=32, max_value=64),
    crop_size=st.integers(min_value=16, max_value=32),
    stride=st.integers(min_value=8, max_value=16),
    n=st.integers(min_value=1, max_value=3),
)
def test_extract_stitch_roundtrip(h: int, w: int, crop_size: int, stride: int, n: int) -> None:
    """Property test for crop extraction and stitching roundtrip."""
    if h < crop_size or w < crop_size:
        return
    if (h - crop_size) % stride != 0 or (w - crop_size) % stride != 0:
        return

    images = np.random.rand(n, h, w, 3).astype(np.float32)

    crops = extract_crops(images, crop_size, stride)
    stitched = stitch_crops(crops, n, h, w, crop_size, stride)

    np.testing.assert_allclose(images, stitched, atol=1e-5)


# 2. Top-K Pooling Monotonicity, Scale & Bounds
@settings(max_examples=20, deadline=None)
@given(
    h=st.integers(min_value=16, max_value=48),
    w=st.integers(min_value=16, max_value=48),
    c=st.floats(min_value=0.0, max_value=10.0),
)
def test_top_k_pooling_invariants(h: int, w: int, c: float) -> None:
    """Test bounds, monotonicity, and scale equivariance of top_k_pooling."""
    # Generate random map inline because we need h, w for k
    k_val = np.random.randint(1, h * w + 1)

    # Map values in [0.0, 10.0]
    map_a = np.random.uniform(0.0, 10.0, size=(h, w)).astype(np.float32)

    # Invariant 1 (Bounds)
    score_a = top_k_pooling(map_a, k=k_val)
    assert np.min(map_a) <= score_a <= np.max(map_a) + 1e-6

    # Invariant 2 (Monotonicity)
    map_b = map_a + np.random.uniform(0.0, 2.0, size=(h, w)).astype(np.float32)
    score_b = top_k_pooling(map_b, k=k_val)
    assert score_a <= score_b + 1e-5

    # Invariant 3 (Scale Equivariance)
    score_c = top_k_pooling(map_a * c, k=k_val)
    assert abs(score_c - c * score_a) < 1e-4


# 3. SSIM + MSE Combined Loss Mathematical Invariants
@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1000))
def test_ssim_mse_loss_invariants(seed: int) -> None:
    """Test identity, symmetry, and non-negativity of SSIM+MSE loss."""
    import tensorflow as tf

    np.random.seed(seed)

    # 1, 32, 32, 3 arrays in [0.0, 1.0]
    map_a = np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)
    map_b = np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)

    a_tf = tf.convert_to_tensor(map_a)
    b_tf = tf.convert_to_tensor(map_b)

    loss_fn = ssim_mse_loss()

    # Invariant 1 (Identity)
    loss_aa = loss_fn(a_tf, a_tf).numpy()
    assert float(loss_aa) < 1e-5

    # Invariant 2 (Symmetry)
    loss_ab = loss_fn(a_tf, b_tf).numpy()
    loss_ba = loss_fn(b_tf, a_tf).numpy()
    assert abs(float(loss_ab) - float(loss_ba)) < 1e-5

    # Invariant 3 (Non-negativity)
    assert float(loss_ab) >= 0.0


# 4. Adaptive Threshold Numerical Stability
@settings(max_examples=20, deadline=None)
@given(
    scores=st.lists(st.floats(min_value=0.0, max_value=5.0), min_size=3, max_size=50),
    method=st.sampled_from(["quantile", "mahalanobis"]),
    degenerate=st.booleans(),
)
def test_adaptive_threshold_stability(scores: list[float], method: str, degenerate: bool) -> None:
    """Test numerical stability of adaptive threshold."""
    if degenerate:
        # Create a zero-variance array
        score_arr = np.full(len(scores), fill_value=scores[0], dtype=np.float32)
    else:
        score_arr = np.array(scores, dtype=np.float32)

    threshold = compute_adaptive_threshold(score_arr, method=method)

    # Invariant
    assert np.isfinite(threshold)
    assert threshold >= np.min(score_arr) - 1e-6


# 5. Segmentation Robustness on Edge-Case Visual Inputs
@settings(max_examples=20, deadline=None)
@given(image_type=st.sampled_from(["zeros", "ones", "random"]))
def test_segmentation_robustness(image_type: str) -> None:
    """Test segmentation robustness on edge cases."""
    if image_type == "zeros":
        image = np.zeros((64, 64, 3), dtype=np.uint8)
    elif image_type == "ones":
        image = np.full((64, 64, 3), fill_value=255, dtype=np.uint8)
    else:
        image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

    segmentor = OtsuCannySegmentor(morph_kernel_size=3)
    masked_image, mask = segmentor.apply(image)

    # Invariant checks
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, [0, 255]))
    assert masked_image.shape == (64, 64, 3)
