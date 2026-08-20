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
from app.pipelines.multi_stage_ae.segmentation import OtsuCannySegmentor

logger = logging.getLogger(__name__)


def _load_images_as_numpy(
    paths: list[str],
    img_size: int,
    segmentor: OtsuCannySegmentor | None = None,
) -> np.ndarray:
    """Load a list of image file paths into a numpy array, optionally applying segmentation.

    This function is framework-agnostic (returns numpy, not PyTorch tensors or TF tensors).
    It serves as the data ingestion layer for the Keras pipeline.

    Args:
        paths: List of absolute file paths to image files.
        img_size: Target size for resizing (both width and height, square images assumed).
        segmentor: Optional ``OtsuCannySegmentor`` to apply foreground extraction.
            If None, images are loaded without background removal.

    Returns:
        Numpy array of uint8 RGB images, shape (N, img_size, img_size, 3), values in [0, 255].
    """
    images: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as pil_img:
            resized = pil_img.convert("RGB").resize((img_size, img_size), Image.Resampling.LANCZOS)
            img_array = np.array(resized, dtype=np.uint8)

        if segmentor is not None:
            img_array, _ = segmentor.apply(img_array)

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


def run_keras_cae_pipeline(
    data_root: str = "data/raw/mvtec_ad",
    category: str = "bottle",
    img_size: int = 128,
    latent_dim: int = 128,
    epochs: int = 20,
    batch_size: int = 16,
    mask_ratio: float = 0.25,
    patch_size: int = 16,
    threshold_method: str = "quantile",
    k_fraction: float = 0.002,
    use_segmentation: bool = True,
    run_heatmap: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Run the complete Keras CAE anomaly detection pipeline for one MVTec category.

    This is the main entry point called by the FastAPI endpoint. It:
    1. Loads train (normal only) and test images as numpy arrays.
    2. Optionally applies Otsu+Canny foreground extraction.
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
        img_size: Size (height and width) to resize all images to. Must be divisible by 16.
        latent_dim: Bottleneck dimension of the CAE.
        epochs: Number of training epochs (more = better, but slower).
        batch_size: Training batch size.
        mask_ratio: Fraction of patches to mask during Masked Image Modeling training.
        patch_size: Side length of each masked patch in pixels.
        threshold_method: ``"quantile"`` or ``"mahalanobis"`` for adaptive threshold.
        k_fraction: Top-K fraction for image-level anomaly score pooling.
        use_segmentation: Whether to apply Otsu+Canny foreground extraction.
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

    train_df = cat[(cat["split"] == "train") & (~cat["is_anomaly"])].copy()
    test_df = cat[cat["split"] == "test"].copy()

    train_paths = train_df["path"].tolist()
    test_paths = test_df["path"].tolist()
    test_labels = test_df["is_anomaly"].astype(int).to_numpy()
    mask_paths = test_df["mask_path"].tolist()

    logger.info("Train samples (normal): %d | Test samples: %d", len(train_paths), len(test_paths))

    # ── 2. Foreground Extraction ────────────────────────────────────────────────
    segmentor = OtsuCannySegmentor() if use_segmentation else None
    if use_segmentation:
        logger.info("Applying Otsu+Canny foreground extraction (BGRP-G strategy).")

    train_images_uint8 = _load_images_as_numpy(train_paths, img_size, segmentor)
    test_images_uint8 = _load_images_as_numpy(test_paths, img_size, segmentor)

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
    val_good_images = test_images[test_labels == 0]
    val_anomalous_images = test_images[test_labels == 1]

    # Create a unique hash for these hyperparameters
    hp_string = f"{category}_{img_size}_{latent_dim}_{epochs}_{batch_size}_{mask_ratio}"
    model_hash = hashlib.sha256(hp_string.encode()).hexdigest()[:12]
    registry_dir = Path("data/models/keras_cae") / model_hash
    model_path = registry_dir / "model.keras"
    meta_path = registry_dir / "metadata.json"

    loss_history = {"train": [], "val_good": [], "val_anomalous": []}

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
        model = build_cae(img_size=img_size, latent_dim=latent_dim)
        loss_history = train_cae(
            model=model,
            train_images=train_images,
            epochs=epochs,
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            val_good_images=val_good_images,
            val_anomalous_images=val_anomalous_images,
        )

        # Save model and metadata to registry
        registry_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        metadata = {
            "hash": model_hash,
            "category": category,
            "img_size": img_size,
            "latent_dim": latent_dim,
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
    normal_scores, _ = compute_image_scores(model, val_good_images, k_fraction=k_fraction)
    threshold = compute_adaptive_threshold(normal_scores, method=threshold_method)  # type: ignore[arg-type]

    # ── 7. Full Evaluation ──────────────────────────────────────────────────────
    gt_masks = _load_masks_as_numpy(mask_paths, img_size)
    results = evaluate_cae(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        gt_masks=gt_masks,
        threshold=threshold,
        k_fraction=k_fraction,
        output_dir=registry_dir,
    )
    results["image_level"] = {
        "auroc": results.get("auroc", 0.0),
        "f1_score": results.get("f1_score", 0.0),
        "metrics_path": str(registry_dir / "image_metrics.npz"),
    }
    results["pixel_level"] = {
        "aupimo": results.get("aupimo", 0.0),
        "metrics_path": str(registry_dir / "pixel_metrics.npz"),
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
        from app.pipelines.multi_stage_ae.error_heatmap import compute_error_heatmap, overlay_heatmap

        heatmap_overlays: dict[int, list] = {}
        for idx in anomalous_indices:
            img_float = test_images[idx]
            img_uint8 = (img_float * 255).astype(np.uint8)
            try:
                result = compute_error_heatmap(model, img_float, sigma=3.0)
                overlay = overlay_heatmap(img_uint8, result["heatmap"], alpha=0.4)
                heatmap_overlays[idx] = overlay.tolist()
            except Exception as exc:
                logger.warning("Error Heatmap failed for image %d: %s", idx, exc)

        results["heatmap_overlays"] = heatmap_overlays

    return results
