"""End-to-end orchestrator for the Keras CAE anomaly detection pipeline.

This module ties together all the modular components developed in this package
into a single, configurable pipeline that can be triggered from the API.

Data Flow
=========
See the ASCII diagram below for the complete pipeline flow:

    [MVTec Dataset]
          │
          ▼
    ┌────────────────────────────────┐
    │ 1. Data Loading                │  build_mvtec_manifest + filter by category
    │    (framework-agnostic, numpy) │
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 2. Foreground Extraction       │  OtsuCannySegmentor → BGRP-G masking
    │    (segmentation.py)           │  Eliminates background noise from scoring
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 3. Category-Aware Augmentation │  get_augmenter(category) → applied to TRAIN only
    │    (augmentation.py)           │  Texture: heavy spatial; Object: light photometric
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 4. Normalise to [0, 1]         │  Divide by 255
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 5. Build & Train Keras CAE     │  ELU activations, AdamW, SSIM+MSE loss
    │    (cae_keras.py)              │  With Masked Image Modeling (patch masking)
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 6. Score Test Images           │  Top-K pooling (image-level) +
    │    (scoring.py)                │  Pixel error maps (pixel-level)
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 7. Adaptive Threshold          │  Calibrated from normal image scores
    │    (scoring.py)                │  quantile or Mahalanobis method
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 8. Full Evaluation             │  Image AUROC + Pixel AUPIMO (FPR 1e-5..1e-4)
    │    (evaluation.py)             │  + Accuracy / Precision / Recall
    └──────────────┬─────────────────┘
                   │
                   ▼
    ┌────────────────────────────────┐
    │ 9. [Optional] SHAP XAI         │  SLIC superpixels + KernelExplainer
    │    (explainability.py)         │  Attribution: red=anomaly, blue=normal
    └──────────────┬─────────────────┘
                   │
                   ▼
    [Results Dictionary → API → Streamlit UI]
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from app.domain.data import build_mvtec_manifest
from app.pipelines.multi_stage_ae.augmentation import augment_batch, get_augmenter
from app.pipelines.multi_stage_ae.cae_keras import build_cae, train_cae
from app.pipelines.multi_stage_ae.evaluation import evaluate_cae
from app.pipelines.multi_stage_ae.scoring import compute_adaptive_threshold, compute_image_scores
from app.pipelines.preprocessing import PreprocessingPipeline, build_pipeline_from_configs

logger = logging.getLogger(__name__)


def _load_images_as_numpy(paths: list[Any], img_size: int, pipeline: PreprocessingPipeline | None = None) -> np.ndarray:
    """Load, resize, and convert a list of image paths to a batched numpy array.

    Args:
        paths: List of absolute file paths to image files.
        img_size: Target size for resizing (both width and height, square images assumed).
        pipeline: Optional preprocessing pipeline to apply to each image.

    Returns:
        Numpy array of uint8 RGB images, shape (N, img_size, img_size, 3), values in [0, 255].
    """
    images: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as pil_img:
            resized = pil_img.convert("RGB").resize((img_size, img_size), Image.Resampling.LANCZOS)
            img_array = np.array(resized, dtype=np.uint8)

        if pipeline is not None and len(pipeline) > 0:
            img_array = pipeline(img_array)

        images.append(img_array)

    return np.stack(images, axis=0)  # (N, H, W, 3)


def _load_masks_as_numpy(mask_paths: list[Any], img_size: int) -> list[np.ndarray | None]:
    """Load ground truth defect masks for pixel-level evaluation.

    Masks are resized and binarised (thresholded at 127) to produce clean binary arrays.
    Handles None, pandas NaN (float), and invalid paths gracefully.

    Args:
        mask_paths: List of mask file paths. May contain None or NaN for images with no defect mask.
        img_size: Target size for mask resizing.

    Returns:
        List of 2D binary numpy arrays (0 = normal, 1 = defect) or None for no-mask images.
    """
    masks: list[np.ndarray | None] = []
    for mask_path in mask_paths:
        if (
            mask_path is None
            or not isinstance(mask_path, (str, Path))
            or not str(mask_path).strip()
            or str(mask_path).lower() in ("nan", "none")
            or not Path(mask_path).exists()
        ):
            masks.append(None)
        else:
            with Image.open(mask_path) as pil_mask:
                resized = pil_mask.resize((img_size, img_size), Image.Resampling.NEAREST).convert("L")
                mask_array = (np.array(resized, dtype=np.uint8) > 127).astype(np.uint8)
            masks.append(mask_array)
    return masks


def extract_crops(images: np.ndarray, crop_size: int, crop_stride: int) -> np.ndarray:
    """Extract overlapping crops from a batch of images.

    Args:
        images: Array of shape (N, H, W, C).
        crop_size: Size of the square crop.
        crop_stride: Stride between crops.

    Returns:
        Array of shape (N * num_crops, crop_size, crop_size, C).
    """
    _, h, w, c = images.shape
    crops = []
    for i in range(0, h - crop_size + 1, crop_stride):
        for j in range(0, w - crop_size + 1, crop_stride):
            crops.append(images[:, i : i + crop_size, j : j + crop_size, :])

    crops_stack = np.stack(crops, axis=1)
    return crops_stack.reshape(-1, crop_size, crop_size, c)


def stitch_crops(
    crops: np.ndarray, n_images: int, img_h: int, img_w: int, crop_size: int, crop_stride: int
) -> np.ndarray:
    """Stitch overlapping crops back into full images by averaging overlapping regions."""
    c = crops.shape[-1]
    reconstructed = np.zeros((n_images, img_h, img_w, c), dtype=np.float32)
    counts = np.zeros((n_images, img_h, img_w, 1), dtype=np.float32)

    num_crops = crops.shape[0] // n_images
    crops_reshaped = crops.reshape(n_images, num_crops, crop_size, crop_size, c)

    idx = 0
    for i in range(0, img_h - crop_size + 1, crop_stride):
        for j in range(0, img_w - crop_size + 1, crop_stride):
            reconstructed[:, i : i + crop_size, j : j + crop_size, :] += crops_reshaped[:, idx]
            counts[:, i : i + crop_size, j : j + crop_size, :] += 1.0
            idx += 1

    counts = np.maximum(counts, 1.0)
    return reconstructed / counts


def run_keras_cae_pipeline(
    data_root: str = "data/raw/mvtec_ad",
    category: str = "bottle",
    img_size: int = 256,
    crop_size: int = 64,
    crop_stride: int = 32,
    latent_channels: int = 32,
    epochs: int = 20,
    batch_size: int = 16,
    mask_ratio: float = 0.25,
    mask_patch_size: int = 8,
    threshold_method: str = "quantile",
    k_fraction: float = 0.002,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    run_heatmap: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Run the complete Keras CAE anomaly detection pipeline for one MVTec category.

    This is the main entry point called by the FastAPI endpoint. It:
    1. Loads train (normal only) and test images as numpy arrays.
    2. Optionally applies modular preprocessing transforms.
    3. Applies category-aware augmentation to training data.
    4. Normalises images to [0, 1].
    5. Builds and trains the Keras CAE with MIM + SSIM+MSE + AdamW.
    6. Scores all test images using Top-K pooling.
    7. Computes an adaptive threshold from normal test scores.
    8. Evaluates with image-level AUROC and pixel-level AUPIMO.
    9. Optionally computes Reconstruction Error Heatmap overlays for every anomalous test image.

    Args:
        data_root: Path to the MVTec AD dataset root directory.
        category: MVTec category to train and evaluate on (e.g., 'bottle', 'wood').
        img_size: Size (height and width) to resize base images to.
        crop_size: Size of sliding window crops extracted from the base image.
        crop_stride: Stride of the sliding window.
        latent_channels: Number of channels in the convolutional bottleneck.
        epochs: Number of training epochs.
        batch_size: Training batch size (number of crops, not full images).
        mask_ratio: Fraction of patches to mask during Masked Image Modeling training.
        mask_patch_size: Side length of each masked region within a crop.
        threshold_method: ``"quantile"`` or ``"mahalanobis"`` for adaptive threshold.
        k_fraction: Top-K fraction for image-level anomaly score pooling.
        preprocessing_steps: Optional configuration list for preprocessing transforms.
        run_heatmap: Whether to compute Reconstruction Error heatmap overlays for anomalous images.
        force_retrain: If True, bypass the cache and force training of a new model.

    Returns:
        Dictionary with all results (metrics, scores, heatmap, optional anomaly heatmaps).
    """
    logger.info("=== Keras CAE Pipeline: category='%s', img_size=%d ===", category, img_size)

    # ── 1. Load Dataset Manifest ────────────────────────────────────────────────
    manifest = build_mvtec_manifest(data_root)
    cat = manifest[manifest["product"] == category].copy()

    if cat.empty:
        raise ValueError(f"No images found for category '{category}' in '{data_root}'")

    from sklearn.model_selection import train_test_split

    full_train_df = cat[(cat["split"] == "train") & (~cat["is_anomaly"])].copy()
    test_df = cat[cat["split"] == "test"].copy()

    # Reserve 15% of the normal training data for validation to prevent test set leakage
    train_df, val_df = train_test_split(full_train_df, test_size=0.15, random_state=42)

    train_paths = train_df["path"].tolist()
    val_paths = val_df["path"].tolist()
    test_paths = test_df["path"].tolist()
    test_labels = test_df["is_anomaly"].astype(int).to_numpy()
    mask_paths = test_df["mask_path"].tolist()

    logger.info("Train (normal): %d | Val (normal): %d | Test: %d", len(train_paths), len(val_paths), len(test_paths))

    # ── 2. Preprocessing ────────────────────────────────────────────────────────
    pipeline = build_pipeline_from_configs(preprocessing_steps)
    if len(pipeline) > 0:
        logger.info("Applying %d preprocessing steps.", len(pipeline))

    train_images_uint8 = _load_images_as_numpy(train_paths, img_size, pipeline)
    val_images_uint8 = _load_images_as_numpy(val_paths, img_size, pipeline)
    test_images_uint8 = _load_images_as_numpy(test_paths, img_size, pipeline)

    # ── 3. Category-Aware Augmentation (Training Only) ─────────────────────────
    augmenter = get_augmenter(category)
    # Augment training data: double the training set with one augmented copy
    augmented = augment_batch(train_images_uint8, augmenter)
    train_images_uint8 = np.concatenate([train_images_uint8, augmented], axis=0)
    logger.info("After augmentation: %d training images.", len(train_images_uint8))

    # ── 4. Normalise to [0, 1] ──────────────────────────────────────────────────
    train_images = train_images_uint8.astype(np.float32) / 255.0
    test_images = test_images_uint8.astype(np.float32) / 255.0

    # ── 5. Build & Train Keras CAE (with Caching) ───────────────────────────────
    logger.info("Extracting overlapping %dx%d crops for training...", crop_size, crop_size)
    train_crops = extract_crops(train_images, crop_size, crop_stride)

    val_good_images = val_images_uint8.astype(np.float32) / 255.0
    val_good_crops = extract_crops(val_good_images, crop_size, crop_stride) if len(val_good_images) > 0 else None
    val_an_crops = None  # No anomalous images used during validation tuning

    # Create a unique hash for these hyperparameters
    hp_string = (
        f"{category}_{img_size}_{crop_size}_{crop_stride}_{latent_channels}_"
        f"{epochs}_{batch_size}_{mask_ratio}_{mask_patch_size}"
    )
    model_hash = hashlib.sha256(hp_string.encode()).hexdigest()[:12]
    registry_dir = Path("data/models/keras_cae") / model_hash
    model_path = registry_dir / "model.keras"
    meta_path = registry_dir / "metadata.json"

    loss_history: dict[str, list[float]] = {"train": [], "val_good": [], "val_anomalous": []}

    if not force_retrain and model_path.exists():
        logger.info("Found cached model with identical hyperparameters (Hash: %s). Loading from disk...", model_hash)
        model = tf.keras.models.load_model(model_path, compile=False)
        # Load cached loss history if available
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
                loss_history = meta.get("loss_history", loss_history)
    else:
        logger.info("No cache found (or force_retrain=True). Training new model (Hash: %s)...", model_hash)
        model = build_cae(crop_size=crop_size, latent_channels=latent_channels)
        loss_history = train_cae(
            model=model,
            train_images=train_crops,
            epochs=epochs,
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            patch_size=mask_patch_size,
            val_good_images=val_good_crops,
            val_anomalous_images=val_an_crops,
        )

        # Save model and metadata to registry
        registry_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        metadata = {
            "hash": model_hash,
            "category": category,
            "img_size": img_size,
            "crop_size": crop_size,
            "crop_stride": crop_stride,
            "latent_channels": latent_channels,
            "epochs": epochs,
            "batch_size": batch_size,
            "mask_ratio": mask_ratio,
            "loss_history": loss_history,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved model and metadata to %s", registry_dir)

    # ── 6. Compute Adaptive Threshold on Normal Test Images ─────────────────────
    logger.info("Extracting crops for val_good images and predicting...")
    val_good_reconstructed_crops = model.predict(val_good_crops, batch_size=batch_size, verbose=0)
    val_good_reconstructed = stitch_crops(
        val_good_reconstructed_crops, len(val_good_images), img_size, img_size, crop_size, crop_stride
    )

    normal_scores, _ = compute_image_scores(
        model, val_good_images, k_fraction=k_fraction, reconstructions=val_good_reconstructed
    )
    threshold = compute_adaptive_threshold(normal_scores, method=threshold_method)  # type: ignore[arg-type]

    # ── 7. Full Evaluation ──────────────────────────────────────────────────────
    logger.info("Extracting crops for all test images and predicting...")
    test_crops = extract_crops(test_images, crop_size, crop_stride)
    test_reconstructed_crops = model.predict(test_crops, batch_size=batch_size, verbose=0)
    test_reconstructed = stitch_crops(
        test_reconstructed_crops, len(test_images), img_size, img_size, crop_size, crop_stride
    )

    gt_masks = _load_masks_as_numpy(mask_paths, img_size)
    results = evaluate_cae(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        gt_masks=gt_masks,
        threshold=threshold,
        k_fraction=k_fraction,
        output_dir=registry_dir,
        reconstructions=test_reconstructed,
    )
    t_aupimo_min = 0.0
    aupimo_recall = 0.0
    pixel_file = registry_dir / "pixel_metrics.npz"
    if pixel_file.exists():
        try:
            data = np.load(pixel_file)
            if "t_aupimo_min" in data:
                t_aupimo_min = float(data["t_aupimo_min"])
            if "aupimo" in data:
                aupimo_recall = float(data["aupimo"])
        except Exception:
            pass

    results["image_level"] = {
        "auroc": results.get("auroc", 0.0),
        "f1_score": results.get("f1_score", 0.0),
        "precision": results.get("precision", 0.0),
        "recall": results.get("recall", 0.0),
        "metrics_path": str(registry_dir / "image_metrics.npz"),
    }
    results["pixel_level"] = {
        "auroc": results.get("pixel_auroc", results.get("auroc", 0.0)),
        "f1_score": results.get("pixel_f1", results.get("f1_score", 0.0)),
        "t_aupimo_min": t_aupimo_min,
        "aupimo": aupimo_recall,
        "metrics_path": str(pixel_file),
    }
    results["final_train_loss"] = loss_history["train"][-1] if loss_history["train"] else 0.0
    results["category"] = category
    results["epochs"] = epochs
    results["loss_history"] = loss_history

    # Include anomalous image indices so the UI can offer a SHAP image selector
    anomalous_indices = [int(i) for i, label in enumerate(test_labels) if label == 1]
    results["anomalous_indices"] = anomalous_indices
    results["total_test_images"] = len(test_images)

    # Serialise numpy arrays for JSON transport via FastAPI
    results["scores"] = results["scores"].tolist()
    results.pop("error_maps", None)  # Large arrays: exclude from API response

    # ── 8. [Optional] Reconstruction Error Heatmap XAI ─────────────────────────
    # Computes a smoothed error heatmap for every anomalous test image.
    if run_heatmap and anomalous_indices:
        logger.info("Computing Reconstruction Error Heatmap for %d anomalous images...", len(anomalous_indices))
        from app.pipelines.multi_stage_ae.error_heatmap import (
            compute_error_heatmap,
            overlay_ground_truth,
            overlay_heatmap,
        )

        heatmap_overlays: dict[int, dict[str, list[Any]]] = {}
        for idx in anomalous_indices:
            img_float = test_images[idx]
            img_uint8 = (img_float * 255).astype(np.uint8)
            try:
                recon_float = test_reconstructed[idx] if test_reconstructed is not None else None
                result = compute_error_heatmap(model, img_float, sigma=3.0, reconstruction=recon_float)
                hm_overlay = overlay_heatmap(img_uint8, result["heatmap"], alpha=0.4)
                gt_and_heatmap = overlay_ground_truth(hm_overlay, gt_masks[idx])

                heatmap_overlays[idx] = {"heatmap": hm_overlay.tolist(), "gt_and_heatmap": gt_and_heatmap.tolist()}
            except Exception as exc:
                logger.warning("Error Heatmap failed for image %d: %s", idx, exc)

        results["heatmap_overlays"] = heatmap_overlays

    return results
