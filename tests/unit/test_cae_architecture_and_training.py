import numpy as np
import tensorflow as tf

from app.pipelines.multi_stage_ae.cae_keras import apply_patch_masking, build_cae, ssim_mse_loss, train_cae


def test_build_cae_spatial_invariants() -> None:
    """Test the spatial invariants and shape of the built CAE."""
    # Test crop size 32
    model_32 = build_cae(crop_size=32, latent_channels=16)
    assert model_32.input_shape == (None, 32, 32, 3)
    assert model_32.output_shape == (None, 32, 32, 3)
    # Output activation should be sigmoid, so values will be in [0, 1] - tested implicitly by architecture setup

    # Test crop size 64
    model_64 = build_cae(crop_size=64, latent_channels=8)
    assert model_64.input_shape == (None, 64, 64, 3)
    assert model_64.output_shape == (None, 64, 64, 3)


def test_ssim_mse_loss() -> None:
    """Test the SSIM + MSE combined loss function on various tensors."""
    loss_fn = ssim_mse_loss(alpha=0.84)

    # Identical tensors
    y_true = tf.constant(np.random.rand(4, 32, 32, 3), dtype=tf.float32)
    y_pred = y_true
    loss_zero = loss_fn(y_true, y_pred).numpy()
    assert np.isclose(loss_zero, 0.0, atol=1e-5)

    # Noisy tensor
    noise = tf.random.normal(shape=(4, 32, 32, 3), mean=0.5, stddev=0.1)
    loss_noisy = loss_fn(y_true, y_true + noise).numpy()
    assert loss_noisy > 0.0

    # More noise
    more_noise = tf.random.normal(shape=(4, 32, 32, 3), mean=1.0, stddev=0.2)
    loss_more_noisy = loss_fn(y_true, y_true + more_noise).numpy()
    assert loss_more_noisy > loss_noisy


def test_apply_patch_masking() -> None:
    """Test applying patch masking with random proportions."""
    batch_size = 4
    crop_size = 32
    images = np.ones((batch_size, crop_size, crop_size, 3), dtype=np.float32)

    mask_ratio = 0.25
    patch_size = 8

    masked_images = apply_patch_masking(images, mask_ratio=mask_ratio, patch_size=patch_size)

    # Verify shape
    assert masked_images.shape == (batch_size, crop_size, crop_size, 3)

    # Verify roughly the correct proportion of the image is masked
    # In each batch, we have (32/8)^2 = 16 patches.
    # 25% of 16 = 4 patches should be masked (0 values).
    # This might have minor variations based on implementation, so we check the approximate ratio.
    zeros_count = np.sum(masked_images == 0.0)
    total_elements = masked_images.size
    actual_ratio = zeros_count / total_elements

    assert np.isclose(actual_ratio, mask_ratio, atol=0.1)


def test_train_cae_minimal(mock_keras_cae) -> None:
    """Test minimal training loop and verify loss values."""
    # Setup minimal data
    train_images = np.random.rand(8, 32, 32, 3).astype(np.float32)
    val_good_images = np.random.rand(4, 32, 32, 3).astype(np.float32)

    model = mock_keras_cae

    # Train for 1 epoch for speed
    history = train_cae(
        model=model,
        train_images=train_images,
        val_good_images=val_good_images,
        epochs=1,
        batch_size=4,
        mask_ratio=0.0,  # simple training
    )

    assert "train" in history
    assert "val_good" in history

    # Verify values are populated
    assert len(history["train"]) == 1
    assert len(history["val_good"]) == 1

    # Basic numeric checks (loss should be float)
    assert isinstance(history["train"][0], float)
    assert history["train"][0] > 0.0
