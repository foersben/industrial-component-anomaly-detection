"""End-to-end orchestrator for the Keras Convolutional Autoencoder (CAE) pipeline.

Coordinates the complete anomaly detection workflow for MVTec AD categories under
the deterministic `fair-eval-v1` evaluation protocol:
    1. Data Loading & Partitioning: Loads dataset manifests and partitions normal
       samples into 85% fit and 15% validation subsets with zero test leakage.
    2. Preprocessing & Patching: Applies optional filters (CLAHE, blur, foreground
       masks) and extracts sliding-window crops.
    3. CAE Modeling: Builds and trains a convolutional autoencoder using Masked
       Image Modeling (MIM) with joint SSIM and MSE reconstruction loss.
    4. Scoring & Calibration: Generates pixel error maps, aggregates image scores
       via Top-K spatial pooling, and calibrates decision thresholds strictly on
       normal validation data.
    5. Evaluation & Persistence: Computes canonical image AUROC, strict AUPIMO,
       confusion matrices, and heatmaps, with deterministic caching and soft-delete
       trash management.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.domain.data import (
    FAIR_EVALUATION_PROTOCOL,
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.pipelines.evaluation.cae_metrics import evaluate_cae
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    PIXEL_METRICS_VERSION,
    fair_metric_evidence,
)
from app.pipelines.evaluation.scoring import compute_adaptive_threshold, compute_image_scores
from app.pipelines.modelling.keras_cae.cae_keras import _require_tf, build_cae, train_cae
from app.pipelines.modelling.keras_cae.crops import extract_crops, stitch_crops
from app.pipelines.modelling.keras_cae.registry import (
    _normalize_preprocessing_steps,
    find_cached_model,
)
from app.pipelines.preprocessing import PreprocessingPipeline, build_pipeline_from_configs
from app.pipelines.preprocessing.augmentation import augment_batch, get_augmenter

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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


def _resolve_cae_cache_and_config(
    category: str,
    img_size: int,
    crop_size: int,
    crop_stride: int,
    latent_channels: int,
    epochs: int,
    batch_size: int,
    mask_ratio: float,
    mask_patch_size: int,
    threshold_method: str,
    k_fraction: float,
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None,
    force_retrain: bool,
    model_hash: str | None,
    cache_evidence: dict[str, Any],
) -> tuple[tuple[Path, dict[str, Any]] | None, dict[str, Any], PreprocessingPipeline, list[dict[str, Any]]]:
    """Resolve model cache or compute unique hash identifiers and parameters for Keras CAE.

    Args:
        category: Component category name.
        img_size: Spatial image dimension.
        crop_size: Crop size.
        crop_stride: Crop stride.
        latent_channels: Latent bottleneck channels.
        epochs: Training epochs.
        batch_size: Batch size.
        mask_ratio: Mask ratio for MIM.
        mask_patch_size: Patch size for MIM.
        threshold_method: Thresholding method.
        k_fraction: Top-K pooling fraction.
        pipeline: Pipeline or list of step dicts.
        force_retrain: If True, bypass cache.
        model_hash: Specific target hash.
        cache_evidence: Fair evaluation split and protocol evidence dict.

    Returns:
        Tuple of (cached_tuple_or_None, resolved_config_dict, proc_pipeline, norm_prep).
    """
    cached = (
        find_cached_model(
            category=category,
            img_size=img_size,
            crop_size=crop_size,
            crop_stride=crop_stride,
            latent_channels=latent_channels,
            epochs=epochs,
            batch_size=batch_size,
            mask_ratio=mask_ratio,
            mask_patch_size=mask_patch_size,
            pipeline=pipeline,
            target_hash=model_hash,
            expected_split_evidence=cache_evidence,
        )
        if not force_retrain
        else None
    )

    cfg: dict[str, Any] = {
        "category": category,
        "img_size": img_size,
        "crop_size": crop_size,
        "crop_stride": crop_stride,
        "latent_channels": latent_channels,
        "epochs": epochs,
        "batch_size": batch_size,
        "mask_ratio": mask_ratio,
        "mask_patch_size": mask_patch_size,
        "threshold_method": threshold_method,
        "k_fraction": k_fraction,
        "model_hash": model_hash,
    }

    if cached is not None:
        registry_dir, meta = cached
        resolved_hash = meta.get("hash", registry_dir.name)
        cfg.update(
            {
                "category": str(meta.get("category", category)),
                "img_size": int(meta.get("img_size", img_size)),
                "crop_size": int(meta.get("crop_size", crop_size)),
                "crop_stride": int(meta.get("crop_stride", crop_stride)),
                "latent_channels": int(meta.get("latent_channels", meta.get("latent_dim", latent_channels))),
                "epochs": int(meta.get("epochs", epochs)),
                "batch_size": int(meta.get("batch_size", batch_size)),
                "mask_ratio": float(meta.get("mask_ratio", mask_ratio)),
                "mask_patch_size": int(meta.get("mask_patch_size", mask_patch_size)),
                "threshold_method": str(meta.get("threshold_method", threshold_method)),
                "k_fraction": float(meta.get("k_fraction", k_fraction)),
                "model_hash": resolved_hash,
                "registry_dir": registry_dir,
                "model_path": registry_dir / "model.keras",
                "meta_path": registry_dir / "metadata.json",
            }
        )
        if "preprocessing_steps" in meta and meta.get("preprocessing_steps") is not None:
            pipeline = meta.get("preprocessing_steps")

    if isinstance(pipeline, PreprocessingPipeline):
        proc_pipeline = pipeline
        norm_prep: list[dict[str, Any]] = []
    else:
        proc_pipeline = build_pipeline_from_configs(pipeline)
        norm_prep = _normalize_preprocessing_steps(pipeline)

    if cached is None:
        prep_str = json.dumps(norm_prep, sort_keys=True)
        hp_string = (
            f"{cfg['category']}_{cfg['img_size']}_{cfg['crop_size']}_{cfg['crop_stride']}_{cfg['latent_channels']}_"
            f"{cfg['epochs']}_{cfg['batch_size']}_{cfg['mask_ratio']}_{cfg['mask_patch_size']}_{prep_str}_"
            f"{json.dumps(cache_evidence, sort_keys=True)}_{CANONICAL_MAP_SIZE}_"
            f"{AUPIMO_FPR_BOUNDS}_{AUPIMO_NUM_THRESHOLDS}_{PIXEL_METRICS_VERSION}"
        )
        computed_hash = hashlib.sha256(hp_string.encode()).hexdigest()[:12]
        reg_dir = Path("data/models/keras_cae") / computed_hash
        cfg.update(
            {
                "model_hash": computed_hash,
                "registry_dir": reg_dir,
                "model_path": reg_dir / "model.keras",
                "meta_path": reg_dir / "metadata.json",
            }
        )

    return cached, cfg, proc_pipeline, norm_prep


def _train_and_save_cae_model(
    cfg: dict[str, Any],
    norm_prep: list[dict[str, Any]],
    dataset_split: dict[str, Any],
    train_crops: np.ndarray,
    val_good_crops: np.ndarray | None,
    val_an_crops: np.ndarray | None,
    trial: Any | None,
) -> tuple[Any, dict[str, list[float]], dict[str, Any]]:
    """Build, train, and persist a new Keras CAE model and its metadata.

    Args:
        cfg: Resolved model configuration dictionary.
        norm_prep: Normalized preprocessing configuration list.
        dataset_split: Fair evaluation partition information.
        train_crops: Array of training image crops.
        val_good_crops: Optional normal validation crops.
        val_an_crops: Optional anomalous validation crops.
        trial: Optional Optuna trial.

    Returns:
        Tuple of (model, loss_history, metadata).
    """
    logger.info("No cache found (or force_retrain=True). Training new model (Hash: %s)...", cfg["model_hash"])
    model = build_cae(crop_size=cfg["crop_size"], latent_channels=cfg["latent_channels"])
    loss_history = train_cae(
        model=model,
        train_images=train_crops,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        mask_ratio=cfg["mask_ratio"],
        patch_size=cfg["mask_patch_size"],
        val_good_images=val_good_crops,
        val_anomalous_images=val_an_crops,
        trial=trial,
    )

    registry_dir: Path = cfg["registry_dir"]
    registry_dir.mkdir(parents=True, exist_ok=True)
    model.save(cfg["model_path"])
    metadata = {
        "hash": cfg["model_hash"],
        "category": cfg["category"],
        "img_size": cfg["img_size"],
        "crop_size": cfg["crop_size"],
        "crop_stride": cfg["crop_stride"],
        "latent_channels": cfg["latent_channels"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "mask_ratio": cfg["mask_ratio"],
        "mask_patch_size": cfg["mask_patch_size"],
        "threshold_method": cfg["threshold_method"],
        "k_fraction": cfg["k_fraction"],
        "preprocessing_steps": norm_prep,
        "dataset_split": dataset_split,
        "protocol": FAIR_EVALUATION_PROTOCOL,
        "threshold_source": "normal_validation",
        "pixel_metrics_version": PIXEL_METRICS_VERSION,
        "loss_history": loss_history,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(cfg["meta_path"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved model and metadata to %s", registry_dir)
    return model, loss_history, metadata


def _compute_cae_heatmaps(
    model: Any,
    test_images: np.ndarray,
    test_reconstructed: np.ndarray,
    gt_masks: list[np.ndarray | None],
    anomalous_indices: list[int],
) -> dict[int, dict[str, list[Any]]]:
    """Compute smoothed reconstruction error heatmap overlays for anomalous test images.

    Args:
        model: Trained Keras CAE model.
        test_images: Normalized test images.
        test_reconstructed: Reconstructed test images.
        gt_masks: Ground truth masks.
        anomalous_indices: Indices of anomalous test images.

    Returns:
        Mapping of image index to heatmap overlay dictionaries.
    """
    logger.info("Computing Reconstruction Error Heatmap for %d anomalous images...", len(anomalous_indices))
    from app.pipelines.evaluation.heatmaps import (
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

    return heatmap_overlays


def _build_cae_result_dict(
    results: dict[str, Any],
    cfg: dict[str, Any],
    loss_history: dict[str, list[float]],
    active_meta: dict[str, Any],
    dataset_split: dict[str, Any],
    threshold: float,
    norm_prep: list[dict[str, Any]],
    test_images: np.ndarray,
    test_labels: Any,
) -> dict[str, Any]:
    """Format and persist evaluation metrics dictionary for Keras CAE pipeline.

    Args:
        results: Evaluated metrics dictionary from evaluate_cae.
        cfg: Configuration dictionary.
        loss_history: Training loss history.
        active_meta: Metadata dictionary from cache or training.
        dataset_split: Split partition evidence.
        threshold: Decision threshold.
        norm_prep: Normalized preprocessing steps.
        test_images: Normalized test images.
        test_labels: Binary test labels.

    Returns:
        Standardized baseline result dictionary.
    """
    registry_dir: Path = cfg["registry_dir"]
    pixel_file = registry_dir / "pixel_metrics.npz"
    results["image_level"] = {
        "auroc": results.get("auroc", 0.0),
        "f1_score": results.get("f1_score", 0.0),
        "precision": results.get("precision", 0.0),
        "recall": results.get("recall", 0.0),
        "threshold": threshold,
        "true_positives": results["true_positives"],
        "false_positives": results["false_positives"],
        "false_negatives": results["false_negatives"],
        "true_negatives": results["true_negatives"],
        "metrics_path": str(registry_dir / "image_metrics.npz"),
    }
    results["pixel_level"] = {
        "auroc": results.get("pixel_auroc", results.get("auroc", 0.0)),
        "f1_score": results.get("pixel_f1", results.get("f1_score", 0.0)),
        "aupimo_score": results.get("aupimo", 0.0),
        "fpr_lower_bound": AUPIMO_FPR_BOUNDS[0],
        "fpr_upper_bound": AUPIMO_FPR_BOUNDS[1],
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "aupimo": results.get("aupimo", 0.0),
        "metrics_path": str(pixel_file),
    }
    results["final_train_loss"] = loss_history["train"][-1] if loss_history["train"] else 0.0
    results["category"] = cfg["category"]
    results["epochs"] = cfg["epochs"]
    results["model_hash"] = cfg["model_hash"]
    results["loss_history"] = loss_history

    update_dict: dict[str, Any] = {
        "protocol": FAIR_EVALUATION_PROTOCOL,
        "dataset_split": dataset_split,
        "threshold_source": "normal_validation",
        "img_threshold": threshold,
        "true_positives": results["true_positives"],
        "false_positives": results["false_positives"],
        "false_negatives": results["false_negatives"],
        "true_negatives": results["true_negatives"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "pixel_auroc": results["pixel_auroc"],
        "pixel_aupimo": results["pixel_aupimo"],
        "aupimo_fpr_bounds": list(AUPIMO_FPR_BOUNDS),
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "pixel_metrics_version": PIXEL_METRICS_VERSION,
    }
    active_meta.update(update_dict)
    with open(registry_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(active_meta, metadata_file, indent=4)
    meta_src = active_meta if active_meta else cfg
    results["metadata"] = active_meta
    results["preprocessing_steps"] = meta_src.get("preprocessing_steps", norm_prep)
    results["hyperparameters"] = {
        "crop_size": meta_src.get("crop_size", cfg["crop_size"]),
        "crop_stride": meta_src.get("crop_stride", cfg["crop_stride"]),
        "latent_channels": meta_src.get("latent_channels", meta_src.get("latent_dim", cfg["latent_channels"])),
        "epochs": meta_src.get("epochs", cfg["epochs"]),
        "batch_size": meta_src.get("batch_size", cfg["batch_size"]),
        "mask_ratio": meta_src.get("mask_ratio", cfg["mask_ratio"]),
        "mask_patch_size": meta_src.get("mask_patch_size", cfg["mask_patch_size"]),
        "threshold_method": meta_src.get("threshold_method", cfg["threshold_method"]),
        "k_fraction": meta_src.get("k_fraction", cfg["k_fraction"]),
        "img_size": meta_src.get("img_size", cfg["img_size"]),
    }
    results["dataset_split"] = meta_src.get("dataset_split", dataset_split)

    anomalous_indices = [idx_num for idx_num, lbl in enumerate(test_labels) if lbl == 1]
    results["anomalous_indices"] = anomalous_indices
    results["total_test_images"] = len(test_images)
    results["scores"] = results["scores"].tolist()
    results.pop("error_maps", None)
    return results


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
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    run_heatmap: bool = False,
    force_retrain: bool = False,
    model_hash: str | None = None,
    trial: Any | None = None,
) -> dict[str, Any]:
    """Run the complete Keras CAE anomaly detection pipeline for one MVTec category.

    This is the main entry point called by the Streamlit application. It:
    1. Resolves cached model or configures new training parameters.
    2. Loads train (normal only) and test images as numpy arrays using exact parameters.
    3. Applies modular preprocessing transforms consistent with model state.
    4. Normalises images to [0, 1].
    5. Builds and trains the Keras CAE with MIM + SSIM+MSE + AdamW (or loads from cache).
    6. Scores all test images using Top-K pooling.
    7. Computes an adaptive threshold from normal validation scores.
    8. Evaluates with image-level AUROC and pixel-level AUPIMO.
    9. Optionally computes Reconstruction Error Heatmap overlays for every anomalous test image.

    Args:
        data_root: Path to the MVTec AD dataset root directory.
        category: MVTec category to train and evaluate on (e.g., 'bottle', 'wood').
        img_size: Size (height and width) to resize base images to.
        crop_size: Size of the sliding window crops extracted from the base image.
        crop_stride: Stride of the sliding window.
        latent_channels: Number of channels in the convolutional bottleneck.
        epochs: Number of training epochs.
        batch_size: Training batch size (number of crops, not full images).
        mask_ratio: Fraction of patches to mask during Masked Image Modeling training.
        mask_patch_size: Side length of each masked region within a crop.
        threshold_method: ``"quantile"`` or ``"mahalanobis"`` for adaptive threshold.
        k_fraction: Top-K fraction for image-level anomaly score pooling.
        pipeline: Optional configuration list or PreprocessingPipeline object.
        run_heatmap: Whether to compute Reconstruction Error heatmap overlays for anomalous images.
        force_retrain: If True, bypass the cache and force training of a new model.
        model_hash: Optional specific model hash to load directly from registry.
        trial: Optional Optuna trial for hyperparameter optimization and pruning.

    Returns:
        Dictionary with all results (metrics, scores, heatmap, optional anomaly heatmaps).
    """
    tf = _require_tf()

    manifest = build_mvtec_manifest(data_root)
    fair_split = build_fair_evaluation_split(manifest, category)
    split_evidence = fair_split.evidence()
    cache_evidence = {**split_evidence, **fair_metric_evidence()}

    cached, cfg, proc_pipeline, norm_prep = _resolve_cae_cache_and_config(
        category=category,
        img_size=img_size,
        crop_size=crop_size,
        crop_stride=crop_stride,
        latent_channels=latent_channels,
        epochs=epochs,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        mask_patch_size=mask_patch_size,
        threshold_method=threshold_method,
        k_fraction=k_fraction,
        pipeline=pipeline,
        force_retrain=force_retrain,
        model_hash=model_hash,
        cache_evidence=cache_evidence,
    )

    if cached is not None:
        registry_dir, meta = cached
        logger.info(
            "Found newest cached model matching parameters (Hash: %s, Dir: %s). Loading from disk...",
            cfg["model_hash"],
            registry_dir,
        )
        model = tf.keras.models.load_model(cfg["model_path"], compile=False)
        loss_history = meta.get("loss_history", {"train": [], "val_good": [], "val_anomalous": []})
        active_meta = meta
    else:
        loss_history = {"train": [], "val_good": [], "val_anomalous": []}
        active_meta = {}

    logger.info(
        "=== Keras CAE Pipeline: category='%s', img_size=%d, hash='%s' ===",
        cfg["category"],
        cfg["img_size"],
        cfg["model_hash"],
    )

    train_paths = fair_split.fitting_paths
    val_paths = fair_split.validation_paths
    test_df = fair_split.test
    test_paths = fair_split.test_paths
    test_labels = test_df["is_anomaly"].astype(int).to_numpy()
    mask_paths = test_df["mask_path"].tolist()

    logger.info("Train (normal): %d | Val (normal): %d | Test: %d", len(train_paths), len(val_paths), len(test_paths))

    if len(proc_pipeline) > 0:
        logger.info("Applying %d preprocessing steps.", len(proc_pipeline))

    train_images_uint8 = _load_images_as_numpy(train_paths, cfg["img_size"], proc_pipeline)
    val_images_uint8 = _load_images_as_numpy(val_paths, cfg["img_size"], proc_pipeline)
    test_images_uint8 = _load_images_as_numpy(test_paths, cfg["img_size"], proc_pipeline)

    augmenter = get_augmenter(cfg["category"])
    augmented = augment_batch(train_images_uint8, augmenter)
    train_images_uint8 = np.concatenate([train_images_uint8, augmented], axis=0)
    logger.info("After augmentation: %d training images.", len(train_images_uint8))

    train_images = train_images_uint8.astype(np.float32) / 255.0
    test_images = test_images_uint8.astype(np.float32) / 255.0
    val_good_images = val_images_uint8.astype(np.float32) / 255.0

    train_crops = extract_crops(train_images, cfg["crop_size"], cfg["crop_stride"])
    val_good_crops = (
        extract_crops(val_good_images, cfg["crop_size"], cfg["crop_stride"]) if len(val_good_images) > 0 else None
    )

    dataset_split = {
        **cache_evidence,
        "test_normal": int(sum(1 for label_val in test_labels if label_val == 0)),
        "test_anomalous": int(sum(1 for label_val in test_labels if label_val == 1)),
    }

    if cached is None:
        model, loss_history, active_meta = _train_and_save_cae_model(
            cfg=cfg,
            norm_prep=norm_prep,
            dataset_split=dataset_split,
            train_crops=train_crops,
            val_good_crops=val_good_crops,
            val_an_crops=None,
            trial=trial,
        )

    logger.info("Extracting crops for val_good images and predicting...")
    val_good_reconstructed_crops = model.predict(val_good_crops, batch_size=cfg["batch_size"], verbose=0)
    val_good_reconstructed = stitch_crops(
        val_good_reconstructed_crops,
        len(val_good_images),
        cfg["img_size"],
        cfg["img_size"],
        cfg["crop_size"],
        cfg["crop_stride"],
    )

    normal_scores, _ = compute_image_scores(
        model, val_good_images, k_fraction=cfg["k_fraction"], reconstructions=val_good_reconstructed
    )
    threshold = compute_adaptive_threshold(normal_scores, method=cfg["threshold_method"])

    logger.info("Extracting crops for all test images and predicting...")
    test_crops = extract_crops(test_images, cfg["crop_size"], cfg["crop_stride"])
    test_reconstructed_crops = model.predict(test_crops, batch_size=cfg["batch_size"], verbose=0)
    test_reconstructed = stitch_crops(
        test_reconstructed_crops,
        len(test_images),
        cfg["img_size"],
        cfg["img_size"],
        cfg["crop_size"],
        cfg["crop_stride"],
    )

    gt_masks = _load_masks_as_numpy(mask_paths, cfg["img_size"])
    results = evaluate_cae(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        gt_masks=gt_masks,
        threshold=threshold,
        k_fraction=cfg["k_fraction"],
        output_dir=cfg["registry_dir"],
        reconstructions=test_reconstructed,
    )

    results = _build_cae_result_dict(
        results=results,
        cfg=cfg,
        loss_history=loss_history,
        active_meta=active_meta,
        dataset_split=dataset_split,
        threshold=threshold,
        norm_prep=norm_prep,
        test_images=test_images,
        test_labels=test_labels,
    )

    if run_heatmap and results["anomalous_indices"]:
        results["heatmap_overlays"] = _compute_cae_heatmaps(
            model=model,
            test_images=test_images,
            test_reconstructed=test_reconstructed,
            gt_masks=gt_masks,
            anomalous_indices=results["anomalous_indices"],
        )

    return results
