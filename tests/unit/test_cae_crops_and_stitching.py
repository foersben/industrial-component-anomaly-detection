"""Unit tests for sliding-window patch extraction and blending reconstruction in the CAE pipeline.

This module validates that overlapping sub-image patches extracted from multi-channel image batches
match the mathematical grid dimensions and can be losslessly blended back into the original image space.
"""

import numpy as np

from app.pipelines.modelling.keras_cae.cae_pipeline import extract_crops, stitch_crops


def test_extract_crops_dimension_matching() -> None:
    """Verify that patch extraction produces the exact expected tensor shapes and grid counts."""
    n_val = 2
    h, w = 64, 64
    c_val = 3
    crop_size = 32
    crop_stride = 16

    images = np.random.rand(n_val, h, w, c_val).astype(np.float32)

    crops = extract_crops(images, crop_size=crop_size, crop_stride=crop_stride)

    # Expected crops per dimension: ((64 - 32) // 16) + 1 = 3
    # Total crops per image: 3 * 3 = 9
    # Total crops for N images: N * 9
    expected_crops = n_val * 9

    assert crops.shape == (expected_crops, crop_size, crop_size, c_val)


def test_perfect_identity_reconstruction() -> None:
    """Verify that extracting patches and stitching them without degradation yields the original image."""
    n_val = 1
    h, w = 64, 64
    c_val = 3
    crop_size = 32
    crop_stride = 16

    # Create a synthetic gradient image for better testing than uniform random
    gradient_x = np.linspace(0, 1, w)
    gradient_y = np.linspace(0, 1, h)
    x_val, y_val = np.meshgrid(gradient_x, gradient_y)

    base_image = np.zeros((h, w, c_val), dtype=np.float32)
    base_image[..., 0] = x_val
    base_image[..., 1] = y_val
    base_image[..., 2] = (x_val + y_val) / 2

    images = np.expand_dims(base_image, axis=0)

    crops = extract_crops(images, crop_size=crop_size, crop_stride=crop_stride)

    # In a perfect autoencoder, reconstructed_crops = crops
    reconstructed_crops = crops.copy()

    stitched_images = stitch_crops(
        reconstructed_crops, n_images=n_val, img_h=h, img_w=w, crop_size=crop_size, crop_stride=crop_stride
    )

    assert stitched_images.shape == images.shape
    assert np.allclose(images, stitched_images, atol=1e-5)
