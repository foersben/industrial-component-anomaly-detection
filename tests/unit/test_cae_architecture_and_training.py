"""Unit tests for Keras Convolutional Autoencoder architecture, loss functions, and training loops.

This module validates CAE layer topologies across multiple patch sizes, SSIM+MSE combined loss
monotonicity with respect to noise perturbations, Masked Image Modeling (MIM) spatial grid masking,
and convergence behavior in the minimal training loop.
"""

from typing import Any

import numpy as np
import tensorflow as tf

from app.pipelines.modelling.keras_cae.cae_keras import apply_patch_masking, build_cae, ssim_mse_loss, train_cae


def test_build_cae_spatial_invariants() -> None:
    """Verify that build_cae constructs models preserving input/output spatial shape invariants."""
    # Test crop size 32
    model_32 = build_cae(crop_size=32, latent_channels=16)
    assert model_32.input_shape == (None, 32, 32, 3)
    assert model_32.output_shape == (None, 32, 32, 3)

    # Test crop size 64
    model_64 = build_cae(crop_size=64, latent_channels=8)
    assert model_64.input_shape == (None, 64, 64, 3)
    assert model_64.output_shape == (None, 64, 64, 3)


def test_ssim_mse_loss() -> None:
    """Verify combined SSIM + MSE loss monotonicity and zero error on identical inputs."""
    loss_fn = ssim_mse_loss(alpha=0.84)

    # Identical tensors
    y_true = tf.constant(np.random.rand(4, 32, 32, 3), dtype=tf.float32)
    y_pred = y_true
    loss_zero = loss_fn(y_true, y_pred).numpy()
    assert np.isclose(loss_zero, 0.0, atol=1e-5)

    # Low noise tensor
    noise = tf.random.normal(shape=(4, 32, 32, 3), mean=0.5, stddev=0.1)
    loss_noisy = loss_fn(y_true, y_true + noise).numpy()
    assert loss_noisy > 0.0

    # High noise tensor
    more_noise = tf.random.normal(shape=(4, 32, 32, 3), mean=1.0, stddev=0.2)
    loss_more_noisy = loss_fn(y_true, y_true + more_noise).numpy()
    assert loss_more_noisy > loss_noisy


def test_apply_patch_masking() -> None:
    """Verify that Masked Image Modeling (MIM) zeros out the expected proportion of image patches."""
    batch_size = 4
    crop_size = 32
    images = np.ones((batch_size, crop_size, crop_size, 3), dtype=np.float32)

    mask_ratio = 0.25
    patch_size = 8

    masked_images = apply_patch_masking(images, mask_ratio=mask_ratio, patch_size=patch_size)

    assert masked_images.shape == (batch_size, crop_size, crop_size, 3)

    zeros_count = np.sum(masked_images == 0.0)
    total_elements = masked_images.size
    actual_ratio = zeros_count / total_elements

    assert np.isclose(actual_ratio, mask_ratio, atol=0.1)


def test_train_cae_minimal(mock_keras_cae: Any) -> None:
    """Verify that train_cae executes epochs and populates train and validation loss histories.

    Args:
        mock_keras_cae: Pre-compiled lightweight CAE model fixture.
    """
    train_images = np.random.rand(8, 32, 32, 3).astype(np.float32)
    val_good_images = np.random.rand(4, 32, 32, 3).astype(np.float32)

    history = train_cae(
        model=mock_keras_cae,
        train_images=train_images,
        val_good_images=val_good_images,
        epochs=1,
        batch_size=4,
        mask_ratio=0.0,
    )

    assert "train" in history
    assert "val_good" in history
    assert len(history["train"]) == 1
    assert len(history["val_good"]) == 1
    assert isinstance(history["train"][0], float)
    assert history["train"][0] > 0.0
