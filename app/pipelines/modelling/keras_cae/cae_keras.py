"""TF/Keras Convolutional Autoencoder (CAE) for industrial anomaly detection.

This module implements a from-scratch Convolutional Autoencoder using TensorFlow/Keras
incorporating the following state-of-the-art design decisions:

Why TF/Keras?
    TF/Keras provides a high-level, declarative API that makes the model architecture
    easy to read, modify, and debug. The ``tf.keras.losses`` and ``tf.image`` modules
    include SSIM natively, making combined loss functions straightforward to implement.
    Note: This module coexists alongside the existing PyTorch baseline autoencoder in
    ``app/pipelines/modelling/autoencoder.py`` as an independent improvement experiment.

Key Design Decisions (and Why)
================================

1. ELU instead of ReLU (and Alternatives)
-----------------------------------------
The standard Rectified Linear Unit (ReLU) has a well-known failure mode: the
"Dying ReLU" problem. If a neuron receives consistently negative inputs during
training, its gradient becomes permanently zero, and the neuron stops contributing
to learning entirely. This is especially problematic in autoencoders where the
bottleneck constrains information flow.

The Exponential Linear Unit (ELU) smoothly saturates for large negative inputs
instead of zeroing them:
    - For x > 0: ELU(x) = x (same as ReLU)
    - For x <= 0: ELU(x) = alpha * (exp(x) - 1) where alpha is typically 1.0

Why not Leaky ReLU?
While Leaky ReLU also prevents dying neurons by adding a small linear slope for
x < 0, it has a sharp kink at x = 0 (it is not continuously differentiable).
This sharp non-linearity can sometimes destabilize the fine-grained reconstruction
gradients in autoencoders. ELU is smooth everywhere, producing more predictable
gradients. Furthermore, ELU's saturation curve naturally pushes mean activations
closer to zero (a "self-normalizing" property), which speeds up learning and acts
like internal batch normalization-a benefit Leaky ReLU lacks.

Benefits of ELU:
    - Neurons never completely die -> stable gradient flow throughout training.
    - Mean activations closer to zero -> network acts like batch normalisation internally.
    - Smooth gradients everywhere -> better fine-grained structural reconstruction.

2. Masked Image Modeling (MIM / MAE-style)
------------------------------------------
A naive autoencoder trained to reconstruct its input can learn a trivial
"identity mapping" - just copying the input directly to the output. This defeats
the entire purpose: such a model would reconstruct anomalies just as well as normal
images, yielding zero anomaly detection capability.

Masked Image Modeling (inspired by Masked Autoencoders, MAE) prevents this:
    - Before training, random square patches of the input image are zeroed (masked).
    - The model must reconstruct the **original clean image** from the **corrupted input**.
    - This forces the model to learn context and structure (filling in missing patches
      from surrounding information) rather than copying pixels.

The result: the model learns deep structural representations of "what normal looks like",
and fails to reconstruct unusual or anomalous regions at test time.

3. SSIM + MSE Combined Loss
---------------------------
Mean Squared Error (MSE) is the most common reconstruction loss, but it has a critical
weakness for anomaly detection: it computes error independently per pixel, ignoring the
spatial structure of the image. A 2-pixel horizontal shift of a normal texture pattern
would register as a massive MSE error, even though the image looks completely normal.

The Structural Similarity Index Measure (SSIM) addresses this by evaluating:
    - Luminance: Are the local mean intensities similar?
    - Contrast: Are the local standard deviations similar?
    - Structure: Are the local spatial correlations similar?

SSIM is computed over a sliding window, capturing neighbourhood context. It is bounded
between -1 and 1 (1 = identical), so we use (1 - SSIM) as the loss term.

Combined loss: L = alpha * (1 - SSIM) + (1 - alpha) * MSE
    - alpha = 0.84 is the recommended value from literature for structural emphasis.
    - This penalises structural differences more strongly than pixel-wise noise.

4. AdamW Optimizer
------------------
Adam (Adaptive Moment Estimation) is the standard deep learning optimizer, adapting
the learning rate per parameter based on gradient history. However, standard Adam
conflates weight decay with the adaptive learning rate step, which can lead to
insufficient regularisation and overfitting on small datasets (like MVTec training sets).

AdamW (Adam with decoupled Weight Decay) separates these two mechanisms:
    - The adaptive learning rate handles the gradient-based update.
    - Weight decay is applied directly to the weights AFTER the gradient step.

This produces stronger and more effective regularisation, which is crucial when training
on a small set of normal images (MVTec training split = 60-400 images per category).

Module Contents
---------------
- ``TF_AVAILABLE``: Boolean flag for whether TensorFlow is importable.
- ``build_cae``: Factory function to build and compile the Keras CAE model.
- ``ssim_mse_loss``: Factory for the combined SSIM+MSE loss function.
- ``apply_patch_masking``: Mask random patches for Masked Image Modeling.
- ``train_cae``: Full training loop with MIM and logging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

logger = logging.getLogger(__name__)

# Lazy TensorFlow import guard.
# TF is large (~500MB+) and may not be installed in all environments.
# By importing lazily, the rest of the app still works without TF.
try:
    import tensorflow as _tf

    TF_AVAILABLE: bool = True
    logger.debug("TensorFlow %s detected.", _tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not found. Install it with: pip install tensorflow")


def _require_tf() -> Any:
    """Check that TensorFlow is available, run device configuration, and return the module.

    Device configuration (GPU memory growth or CPU AVX2 + oneDNN) is applied
    exactly once via ``configure_tensorflow()`` - subsequent calls are no-ops
    thanks to the module-level cache in ``app.core.tf_device``.

    Returns:
        The tensorflow module.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for the Keras CAE pipeline but is not installed. "
            "Install it with: pip install tensorflow  (or tensorflow-cpu for CPU-only)"
        )

    # Configure device selection: GPU if CUDA + sufficient VRAM, else CPU-AVX2.
    # This is safe to call multiple times - it caches after the first call.
    from app.core.tf_device import configure_tensorflow

    configure_tensorflow()

    import tensorflow as tf

    return tf


def ssim_mse_loss(alpha: float = 0.84) -> Any:
    """Create a combined SSIM + MSE reconstruction loss function.

    Rationale for alpha = 0.84:
        This value is the empirically recommended default from the SSIM anomaly detection
        literature (e.g. Bergmann et al., 2019 "Improving Unsupervised Defect Segmentation
        by Applying Structural Similarity to Autoencoders"). It weights structural fidelity
        (SSIM) more heavily than per-pixel accuracy (MSE).

    Mathematical formulation:
        L(y_true, y_pred) = alpha * (1 - SSIM(y_true, y_pred)) + (1 - alpha) * MSE(y_true, y_pred)

    Args:
        alpha: Weight of the SSIM term (default: 0.84). Must be in [0, 1].
            - alpha = 1.0 -> pure SSIM loss.
            - alpha = 0.0 -> pure MSE loss.

    Returns:
        A Keras-compatible loss function ``loss(y_true, y_pred) -> scalar tensor``.
    """
    tf = _require_tf()

    def _loss(y_true: Any, y_pred: Any) -> Any:
        """Compute the combined SSIM + MSE loss.

        Args:
            y_true: Ground truth images, shape (B, H, W, 3), values in [0, 1].
            y_pred: Reconstructed images, shape (B, H, W, 3), values in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        # SSIM returns per-image similarity in [-1, 1]. max_val=1.0 for normalised images.
        ssim_per_image = tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim_loss = 1.0 - tf.reduce_mean(ssim_per_image)

        # MSE: mean squared error across all pixels and channels
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        return alpha * ssim_loss + (1.0 - alpha) * mse_loss

    return _loss


def build_cae(crop_size: int = 64, latent_channels: int = 32) -> Any:
    """Build and compile a Convolutional Autoencoder (CAE) using TF/Keras.

    Architecture Overview:
        The model follows an encoder-bottleneck-decoder design:

        Encoder (compression):
            Input(H, W, 3)
            -> Conv2D(32,  4x4, stride 2) + BatchNorm + ELU   -> H/2  x W/2  x 32
            -> Conv2D(64,  4x4, stride 2) + BatchNorm + ELU   -> H/4  x W/4  x 64
            -> Conv2D(128, 4x4, stride 2) + BatchNorm + ELU   -> H/8  x W/8  x 128
            -> Conv2D(256, 4x4, stride 2) + BatchNorm + ELU   -> H/16 x W/16 x 256
            -> Conv2D(latent_channels, 3x3, stride 1)         -> H/16 x W/16 x latent_channels

        Decoder (reconstruction):
            -> Conv2DTranspose(256, 3x3, stride 1) + BatchNorm + ELU -> H/16 x W/16 x 256
            -> Conv2DTranspose(128, 4x4, stride 2) + BatchNorm + ELU -> H/8  x W/8  x 128
            -> Conv2DTranspose(64,  4x4, stride 2) + BatchNorm + ELU -> H/4  x W/4  x 64
            -> Conv2DTranspose(32,  4x4, stride 2) + BatchNorm + ELU -> H/2  x W/2  x 32
            -> Conv2DTranspose(3,   4x4, stride 2) + Sigmoid         -> H    x W    x 3

    Choices explained:
        - Stride-2 convolutions for downsampling (no separate MaxPool layers).
        - BatchNormalisation after each convolution for training stability.
        - ELU activations everywhere except the final decoder layer.
        - Sigmoid on the final layer to produce outputs in [0, 1] (for SSIM compatibility).
        - AdamW optimiser (weight_decay=1e-4) for regularised training.

    Args:
        crop_size: Spatial size of input image crops (both width and height). Must be divisible by 16.
        latent_channels: Number of channels in the convolutional bottleneck.

    Returns:
        Compiled ``tf.keras.Model`` ready for training.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    tf = _require_tf()

    if crop_size % 16 != 0:
        raise ValueError(f"crop_size must be divisible by 16 (for 4 stride-2 layers). Got: {crop_size}")

    # Spatial dimensions at the bottleneck (after 4 stride-2 downsampling layers)
    bottleneck_spatial = crop_size // 16  # e.g. 64 -> 4

    layers = tf.keras.layers
    sequential = tf.keras.Sequential

    model = sequential(
        [
            # ENCODER
            tf.keras.Input(shape=(crop_size, crop_size, 3), name="encoder_input"),
            layers.Conv2D(32, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # Fully Convolutional Bottleneck
            layers.Conv2D(
                latent_channels, kernel_size=3, strides=1, padding="same", use_bias=False, name="latent_conv"
            ),
            # DECODER
            layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding="same", use_bias=False, name="decoder_conv"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ELU(),
            # Final layer: sigmoid to keep outputs in [0, 1] for SSIM computation
            layers.Conv2DTranspose(
                3, kernel_size=4, strides=2, padding="same", activation="sigmoid", name="reconstruction"
            ),
        ],
        name="ConvAutoencoder_ELU",
    )

    # AdamW with decoupled weight decay for regularisation
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.compile(optimizer=optimizer, loss=ssim_mse_loss(alpha=0.84))

    logger.info(
        "Built Keras CAE: crop_size=%d, latent_channels=%d, bottleneck_spatial=%d",
        crop_size,
        latent_channels,
        bottleneck_spatial,
    )
    return model


def apply_patch_masking(
    images: Any,
    mask_ratio: float = 0.25,
    patch_size: int = 16,
) -> Any:
    """Randomly mask square patches in a batch of images (Masked Image Modeling).

    This is the core of the MAE (Masked Autoencoder) training strategy. The model
    must learn to fill in the blanks from context - which forces it to understand
    the structural grammar of normal industrial surfaces rather than copying pixels.

    How it works:
        1. Divide the image into a grid of (img_size // patch_size) x (img_size // patch_size) patches.
        2. Randomly select ``mask_ratio`` fraction of patches to zero out.
        3. Return the corrupted images (model input) alongside the originals (training target).

    Args:
        images: Batch of normalised images, shape (B, H, W, 3), values in [0, 1].
            This should be a numpy array, not a tf.Tensor, for simple indexing.
        mask_ratio: Fraction of patches to mask. 0.25 means 25% of patches are zeroed.
            Higher ratios force more aggressive in-painting, but can destabilise training.
        patch_size: Side length of each square patch in pixels.

    Returns:
        Masked copy of the input batch as a numpy array of the same shape.
        The original (unmasked) images serve as the reconstruction target during training.

    Example:
        For a 128x128 image with patch_size=16: 8x8 = 64 total patches.
        With mask_ratio=0.25: 16 patches are zeroed (selected randomly each batch).
    """
    images_np = np.array(images)  # Ensure numpy for easy indexing
    batch_size, img_h, img_w, _ = images_np.shape

    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked = max(1, int(total_patches * mask_ratio))  # Always mask at least 1 patch

    masked = images_np.copy()

    for b in range(batch_size):
        # Randomly sample which patches to mask for this image
        patch_indices = np.random.choice(total_patches, size=num_masked, replace=False)
        for patch_idx in patch_indices:
            row = patch_idx // num_patches_w
            col = patch_idx % num_patches_w
            # Zero out the patch region
            r_start = row * patch_size
            c_start = col * patch_size
            masked[b, r_start : r_start + patch_size, c_start : c_start + patch_size, :] = 0.0

    return masked


def train_cae(
    model: Any,
    train_images: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16,
    mask_ratio: float = 0.25,
    patch_size: int = 16,
    val_good_images: np.ndarray | None = None,
    val_anomalous_images: np.ndarray | None = None,
    early_stopping_patience: int = 20,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    min_delta: float = 1e-6,
    trial: optuna.Trial | None = None,
) -> dict[str, list[float]]:
    """Train the CAE model using Masked Image Modeling (MIM).

    For each training batch:
        1. Apply random patch masking to create a corrupted version of the images.
        2. Feed the corrupted images to the model (encoder input).
        3. Compute the combined SSIM+MSE loss against the ORIGINAL clean images.
        4. Backpropagate and update weights.

    This teaches the model to reconstruct clean normal images from partial corrupted views,
    building a rich internal representation of "what is normal".

    Why manual callbacks (Early Stopping, Checkpoint, ReduceLROnPlateau)?
        Because we apply random patch masking dynamically *per batch* using a custom
        training loop (`model.train_on_batch`), we cannot directly pass standard
        `tf.keras.callbacks` to a `model.fit()` call. Instead, we manually track
        validation loss and implement these optimization strategies natively within
        the loop. This provides full control over the MIM augmentation process
        while retaining state-of-the-art training optimizations.

    Args:
        model: Compiled Keras CAE model from ``build_cae()``.
        train_images: Normalised training images, shape (N, H, W, 3), values in [0, 1].
        epochs: Number of full passes over the training data.
        batch_size: Number of images per gradient update step.
        mask_ratio: Fraction of patches to mask per image per step.
        patch_size: Side length of each masked patch in pixels.
        val_good_images: Optional normal images for validation loss tracking.
        val_anomalous_images: Optional anomalous images for validation loss tracking.
        early_stopping_patience: Epochs without improvement before stopping.
        lr_patience: Epochs without improvement before reducing learning rate.
        lr_factor: Factor by which to reduce the learning rate (e.g. 0.5).
        min_delta: Minimum change required to qualify as an improvement.
        trial: Optional Optuna trial for early pruning.

    Returns:
        Dictionary containing lists of epoch-average loss values:
        {'train': [...], 'val_good': [...], 'val_anomalous': [...]}.
    """
    tf = _require_tf()
    
    class NumpyBatchGenerator(tf.keras.utils.Sequence):
        def __init__(self, x, y, batch_size):
            self.x = x
            self.y = y
            self.batch_size = batch_size
            
        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))
            
        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

    n_samples = len(train_images)
    history: dict[str, list[float]] = {"train": [], "val_good": [], "val_anomalous": []}

    best_loss = float("inf")
    best_weights = None
    patience_counter = 0
    lr_patience_counter = 0

    for epoch in range(epochs):
        # Shuffle training data at the start of each epoch
        indices = np.random.permutation(n_samples)
        shuffled_clean = train_images[indices]
        shuffled_masked = apply_patch_masking(shuffled_clean, mask_ratio, patch_size)

        # Use a Sequence generator to avoid allocating huge CPU tensors and OOMing during copies
        gen = NumpyBatchGenerator(shuffled_masked, shuffled_clean, batch_size)
        
        fit_hist = model.fit(
            gen,
            epochs=1,
            verbose=0,
            shuffle=False,  # We already shuffled manually
        )
        avg_loss = fit_hist.history["loss"][0]
        history["train"].append(avg_loss)

        log_msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.6f}"

        # Evaluate validation losses without masking (simulating test-time reconstruction)
        if val_good_images is not None and len(val_good_images) > 0:
            val_good_loss = float(model.evaluate(val_good_images, val_good_images, batch_size=batch_size, verbose=0))
            history["val_good"].append(val_good_loss)
            log_msg += f" | Val Good Loss: {val_good_loss:.6f}"
            monitor_loss = val_good_loss
        else:
            monitor_loss = avg_loss

        if val_anomalous_images is not None and len(val_anomalous_images) > 0:
            val_an_loss = float(
                model.evaluate(val_anomalous_images, val_anomalous_images, batch_size=batch_size, verbose=0)
            )
            history["val_anomalous"].append(val_an_loss)
            log_msg += f" | Val Anomaly Loss: {val_an_loss:.6f}"

        logger.info(log_msg)

        # Optuna Pruning Integration
        if trial is not None:
            trial.report(float(monitor_loss), step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Callbacks Logic (Early Stopping, Checkpoint, ReduceLR)
        if monitor_loss < best_loss - min_delta:
            best_loss = monitor_loss
            best_weights = model.get_weights()
            patience_counter = 0
            lr_patience_counter = 0
        else:
            patience_counter += 1
            lr_patience_counter += 1

        if lr_patience_counter >= lr_patience:
            old_lr = float(model.optimizer.learning_rate.numpy())
            new_lr = old_lr * lr_factor
            model.optimizer.learning_rate.assign(new_lr)
            logger.info("ReduceLROnPlateau triggered: reducing learning rate from %.6f to %.6f", old_lr, new_lr)
            lr_patience_counter = 0  # reset LR counter

        if patience_counter >= early_stopping_patience:
            logger.info("Early Stopping triggered after %d epochs without improvement.", early_stopping_patience)
            break

    # Restore best weights
    if best_weights is not None:
        logger.info("Restoring best model weights (monitor_loss = %.6f)", best_loss)
        model.set_weights(best_weights)

    return history
