"""Evaluation metrics extraction, heatmap overlay serialization, and result formatting for PatchCore."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from anomalib.engine import Engine
from sklearn.metrics import f1_score, roc_auc_score

from app.core.logger import logger
from app.domain.evaluation import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    BaselineResult,
    ConfusionMatrix,
    EvaluationArtifacts,
    ImageEvaluationMetrics,
    PixelEvaluationMetrics,
    Thresholds,
)
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    compute_and_save_pr_metrics,
    compute_image_confusion_metrics,
    compute_shared_pixel_metrics,
)
from app.pipelines.evaluation.scoring import compute_adaptive_threshold
from app.pipelines.modelling.patchcore.visualization import (
    _generate_test_heatmaps,
    _RawScoreImageVisualizer,
)
from app.pipelines.modelling.patchcore.visualization import (
    _load_heatmap_overlays as _load_heatmap_overlays,
)
from app.pipelines.modelling.patchcore.visualization import (
    _save_heatmap_overlays as _save_heatmap_overlays,
)


def _to_float(val: Any) -> float:
    """Safely convert a scalar, PyTorch tensor, or numeric value to a float.

    Args:
        val: Value to convert to float.

    Returns:
        Float representation of the value.
    """
    if hasattr(val, "item") and callable(val.item):
        try:
            return float(val.item())
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _tensor_to_numpy(val: Any) -> np.ndarray[Any, Any] | None:
    """Safely extract a 1D NumPy array from a PyTorch tensor attribute.

    Args:
        val: Potential tensor object.

    Returns:
        1D float NumPy array if val is a tensor, else None.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().reshape(-1)
    return None


def _collect_batch_tensors(
    batch: Any,
    score_attr: str,
    label_attr: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Extract score and label arrays from a batch if both attributes exist and are tensors.

    Args:
        batch: Anomalib prediction batch object.
        score_attr: Attribute name containing anomaly scores or maps.
        label_attr: Attribute name containing ground truth labels or masks.

    Returns:
        Tuple of (score_array, label_array) or None if either attribute is missing.
    """
    scores = _tensor_to_numpy(getattr(batch, score_attr, None))
    labels = _tensor_to_numpy(getattr(batch, label_attr, None))

    if scores is not None and labels is not None:
        return scores, labels
    return None


def _process_and_save_level(
    scores_list: list[np.ndarray[Any, Any]],
    labels_list: list[np.ndarray[Any, Any]],
    output_path: Path,
    level: str,
    aupimo: float | None = None,
    fpr_bounds: tuple[float, float] | None = None,
) -> None:
    """Concatenate prediction lists and save precision-recall metrics if data is present.

    Args:
        scores_list: List of anomaly score arrays.
        labels_list: List of ground-truth label arrays.
        output_path: Output .npz destination path.
        level: Evaluation level tag ("image" or "pixel").
        aupimo: Genuine AUPIMO score computed on full 2D maps, when available.
        fpr_bounds: False Positive Rate integration bounds, when available.
    """
    if not (scores_list and labels_list):
        return

    y_score = np.concatenate(scores_list)
    y_true = np.concatenate(labels_list)

    compute_and_save_pr_metrics(
        y_true,
        y_score,
        output_path,
        level=level,
        aupimo=aupimo,
        fpr_bounds=fpr_bounds,
    )
    logger.info("Saved %s-level PR metrics to %s", level, output_path)


def _calibrate_validation_thresholds(
    engine: Engine,
    model: Any,
    validation_dataloader: Any,
    model_name: str,
) -> tuple[float, float]:
    """Compute frozen deployment thresholds using normal validation samples.

    Args:
        engine: Anomalib engine instance.
        model: Trained model.
        validation_dataloader: Loader containing only shared normal validation images.
        model_name: Model name for log messages.

    Returns:
        Tuple of (img_threshold, pix_threshold).
    """
    visualizer = getattr(model, "visualizer", None)
    if isinstance(visualizer, _RawScoreImageVisualizer):
        visualizer.render_enabled = False

    validation_predictions = engine.predict(model=model, dataloaders=validation_dataloader)
    if not validation_predictions:
        raise RuntimeError(f"{model_name} prediction returned no validation batches")

    validation_scores: list[np.ndarray[Any, Any]] = []
    validation_pixel_scores: list[np.ndarray[Any, Any]] = []
    for batch in validation_predictions:
        scores = _tensor_to_numpy(getattr(batch, "pred_score", None))
        maps = _tensor_to_numpy(getattr(batch, "anomaly_map", None))
        if scores is not None:
            validation_scores.append(scores)
        if maps is not None:
            validation_pixel_scores.append(maps)

    if not validation_scores:
        raise RuntimeError(f"{model_name} validation predictions did not contain image scores")
    if not validation_pixel_scores:
        raise RuntimeError(f"{model_name} validation predictions did not contain anomaly maps")

    img_threshold = compute_adaptive_threshold(
        np.concatenate(validation_scores),
        method="quantile",
        quantile=PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    )
    pix_threshold = compute_adaptive_threshold(
        np.concatenate(validation_pixel_scores),
        method="quantile",
        quantile=PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    )

    if isinstance(visualizer, _RawScoreImageVisualizer):
        visualizer.pixel_threshold = pix_threshold
        visualizer.render_enabled = True

    return img_threshold, pix_threshold


def _extract_batch_maps_and_masks(
    batch: Any,
) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any] | None]]:
    """Convert and validate anomaly map and ground truth mask tensors from a batch.

    Args:
        batch: Anomalib prediction batch item.

    Returns:
        Tuple of (maps_list, masks_list).

    Raises:
        ValueError: If maps and masks have mismatched dimensions or invalid shapes.
    """
    map_tensor = getattr(batch, "anomaly_map", None)
    mask_tensor = getattr(batch, "gt_mask", None)
    if map_tensor is None or mask_tensor is None:
        return [], []

    maps = map_tensor.detach().cpu().numpy() if isinstance(map_tensor, torch.Tensor) else np.asarray(map_tensor)
    masks = mask_tensor.detach().cpu().numpy() if isinstance(mask_tensor, torch.Tensor) else np.asarray(mask_tensor)

    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if maps.ndim != 3 or masks.ndim != 3 or maps.shape != masks.shape:
        raise ValueError(
            "AUPIMO requires matching PatchCore maps and masks with shape (N, H, W); "
            f"got maps={maps.shape}, masks={masks.shape}"
        )

    return (
        [np.asarray(item, dtype=np.float32) for item in maps],
        [np.asarray(item, dtype=np.uint8) for item in masks],
    )


def _collect_test_predictions(
    predictions: list[Any],
    model_name: str,
) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any] | None], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Parse test batch predictions into anomaly maps, masks, and image-level scores/labels.

    Args:
        predictions: Batches returned by engine.predict.
        model_name: Model name for error diagnostics.

    Returns:
        Tuple of (anomaly_maps, ground_truth_masks, image_scores_np, image_labels_np).

    Raises:
        RuntimeError: If anomaly maps or masks are missing.
        ValueError: If both classes are not present in test labels.
    """
    image_scores: list[np.ndarray[Any, Any]] = []
    image_labels: list[np.ndarray[Any, Any]] = []
    anomaly_maps: list[np.ndarray[Any, Any]] = []
    ground_truth_masks: list[np.ndarray[Any, Any] | None] = []

    for batch in predictions:
        maps, masks = _extract_batch_maps_and_masks(batch)
        if maps and masks:
            anomaly_maps.extend(maps)
            ground_truth_masks.extend(masks)

        if img := _collect_batch_tensors(batch, "pred_score", "gt_label"):
            image_scores.append(img[0])
            image_labels.append(img[1])

    if not anomaly_maps or not ground_truth_masks:
        raise RuntimeError(f"{model_name} predictions did not contain full anomaly maps and ground-truth masks")

    image_scores_np = np.concatenate(image_scores)
    image_labels_np = np.concatenate(image_labels).astype(np.uint8)
    if len(np.unique(image_labels_np)) < 2:
        raise ValueError(f"{model_name} image AUROC requires both normal and anomalous test labels")

    return anomaly_maps, ground_truth_masks, image_scores_np, image_labels_np


def _compute_and_persist_metrics(
    anomaly_maps: list[np.ndarray[Any, Any]],
    ground_truth_masks: list[np.ndarray[Any, Any] | None],
    image_scores_np: np.ndarray[Any, Any],
    image_labels_np: np.ndarray[Any, Any],
    img_threshold: float,
    pix_threshold: float,
    base_dir: Path,
    model_name: str,
) -> tuple[ImageEvaluationMetrics, PixelEvaluationMetrics]:
    """Calculate shared evaluation metrics and save pixel/image PR files.

    Args:
        anomaly_maps: Raw continuous anomaly score maps.
        ground_truth_masks: Ground-truth binary segmentation masks.
        image_scores_np: Aggregated continuous image-level anomaly scores.
        image_labels_np: Binary ground-truth image defect labels.
        img_threshold: Image classification threshold calibrated on normal validation set.
        pix_threshold: Pixel segmentation threshold calibrated on normal validation set.
        base_dir: Directory to save serialized metric arrays (.npz).
        model_name: Human-readable model identifier.

    Returns:
        Tuple of (ImageEvaluationMetrics, PixelEvaluationMetrics).
    """
    image_auroc = float(roc_auc_score(image_labels_np, image_scores_np))
    shared_pixel_metrics, canonical_maps, canonical_masks = compute_shared_pixel_metrics(
        anomaly_maps, ground_truth_masks, image_labels_np
    )
    fpr_bounds = AUPIMO_FPR_BOUNDS
    pixel_aupimo = float(shared_pixel_metrics["pixel_aupimo"])
    pixel_auroc = float(shared_pixel_metrics["pixel_auroc"])

    anomaly_map_min = float(canonical_maps.min())
    anomaly_map_max = float(canonical_maps.max())
    anomaly_map_range = anomaly_map_max - anomaly_map_min

    logger.info(
        "%s full-map AUPIMO: %.6f at FPR bounds %s | maps min=%.8f max=%.8f range=%.8f",
        model_name,
        pixel_aupimo,
        fpr_bounds,
        anomaly_map_min,
        anomaly_map_max,
        anomaly_map_range,
    )

    _process_and_save_level(
        [canonical_maps.reshape(-1)],
        [canonical_masks.reshape(-1)],
        base_dir / "pixel_metrics.npz",
        level="pixel",
        aupimo=pixel_aupimo,
        fpr_bounds=fpr_bounds,
    )
    _process_and_save_level(
        [image_scores_np],
        [image_labels_np],
        base_dir / "image_metrics.npz",
        level="image",
    )

    confusion_dict = compute_image_confusion_metrics(image_labels_np, image_scores_np, img_threshold)
    confusion = ConfusionMatrix(
        true_positives=int(confusion_dict["true_positives"]),
        false_positives=int(confusion_dict["false_positives"]),
        false_negatives=int(confusion_dict["false_negatives"]),
        true_negatives=int(confusion_dict["true_negatives"]),
    )
    image_metrics = ImageEvaluationMetrics(
        auroc=image_auroc,
        f1_score=float(confusion_dict["f1_score"]),
        precision=float(confusion_dict["precision"]),
        recall=float(confusion_dict["recall"]),
        threshold=img_threshold,
        confusion=confusion,
    )

    pixel_scores_np = canonical_maps.reshape(-1)
    pixel_labels_np = canonical_masks.reshape(-1)
    pix_preds = (pixel_scores_np > pix_threshold).astype(int)
    manual_pixel_f1 = float(f1_score(pixel_labels_np, pix_preds))

    pixel_metrics = PixelEvaluationMetrics(
        auroc=pixel_auroc,
        aupimo=pixel_aupimo,
        f1_score=manual_pixel_f1,
        threshold=pix_threshold,
        anomaly_map_min=anomaly_map_min,
        anomaly_map_max=anomaly_map_max,
        anomaly_map_range=anomaly_map_range,
    )

    return image_metrics, pixel_metrics


def extract_and_save_pr_metrics(
    engine: Engine,
    model: Any,
    validation_dataloader: Any,
    test_dataloader: Any,
    base_dir: Path,
    run_heatmap: bool = False,
    model_name: str = "PatchCore",
) -> EvaluationArtifacts:
    """Extract model predictions and persist Precision-Recall metrics for visual analysis.

    Args:
        engine: Anomalib engine instance.
        model: Trained model.
        validation_dataloader: Loader containing only shared normal validation images.
        test_dataloader: Loader containing the unchanged official test partition.
        base_dir: Output directory for metrics.
        run_heatmap: Whether to compute heatmap overlays.
        model_name: Human-readable model name used in diagnostics.

    Returns:
        Structured EvaluationArtifacts container (also unpackable as 18-tuple for backward compatibility).
    """
    try:
        logger.info("Extracting predictions for PR curve metrics...")
        img_threshold, pix_threshold = _calibrate_validation_thresholds(
            engine, model, validation_dataloader, model_name
        )

        raw_predictions = engine.predict(model=model, dataloaders=test_dataloader)
        if not raw_predictions:
            raise RuntimeError(f"{model_name} prediction returned no test batches")

        anomaly_maps, ground_truth_masks, image_scores_np, image_labels_np = _collect_test_predictions(
            raw_predictions, model_name
        )

        image_metrics, pixel_metrics = _compute_and_persist_metrics(
            anomaly_maps,
            ground_truth_masks,
            image_scores_np,
            image_labels_np,
            img_threshold,
            pix_threshold,
            base_dir,
            model_name,
        )

        heatmap_overlays: dict[int, dict[str, list[Any]]] = {}
        anomalous_indices: list[int] = []
        if run_heatmap:
            heatmap_overlays, anomalous_indices = _generate_test_heatmaps(raw_predictions)

        return EvaluationArtifacts(
            image_metrics=image_metrics,
            pixel_metrics=pixel_metrics,
            thresholds=Thresholds(image=img_threshold, pixel=pix_threshold),
            heatmap_overlays=heatmap_overlays,
            anomalous_indices=anomalous_indices,
        )

    except Exception as e:
        logger.exception("Could not compute %s evaluation metrics: %s", model_name, e)
        raise


def format_results(
    test_results: list[Mapping[str, float]] | None,
    category: str,
    base_dir: Path,
    manual_image_f1: float = 0.0,
    manual_pixel_f1: float = 0.0,
    manual_image_prec: float = 0.0,
    manual_image_rec: float = 0.0,
    img_threshold: float = 0.0,
    pixel_threshold: float = 0.0,
    pixel_auroc: float = 0.0,
    pixel_aupimo: float = 0.0,
    anomaly_map_min: float = 0.0,
    anomaly_map_max: float = 0.0,
    anomaly_map_range: float = 0.0,
    heatmap_overlays: dict[int, dict[str, list[Any]]] | None = None,
    anomalous_indices: list[int] | None = None,
    fpr_limit: float = 1e-4,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    dataset_split: dict[str, Any] | None = None,
    model_hash: str = "",
    metadata: dict[str, Any] | None = None,
    true_positives: int = 0,
    false_positives: int = 0,
    false_negatives: int = 0,
    true_negatives: int = 0,
    artifacts: EvaluationArtifacts | None = None,
) -> BaselineResult:
    """Format Anomalib engine evaluation output into a structured response schema.

    Args:
        test_results: A list of metric mappings from Anomalib.
        category: The component category name.
        base_dir: Base directory to save metrics to.
        manual_image_f1: Manually calculated image-level F1 score.
        manual_pixel_f1: Manually calculated pixel-level F1 score.
        manual_image_prec: Manually calculated image-level Precision score.
        manual_image_rec: Manually calculated image-level Recall score.
        img_threshold: Manually calculated image-level classification threshold.
        pixel_threshold: Normal-validation threshold used to create predicted masks.
        pixel_auroc: Pixel AUROC from the shared canonical metric path.
        pixel_aupimo: Full-map AUPIMO computed by Anomalib.
        anomaly_map_min: Minimum PatchCore anomaly-map value.
        anomaly_map_max: Maximum PatchCore anomaly-map value.
        anomaly_map_range: Range of PatchCore anomaly-map values.
        heatmap_overlays: Dictionary of precomputed heatmap overlays.
        anomalous_indices: List of image indices corresponding to anomalies.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
        preprocessing_steps: Optional list of active preprocessing steps.
        hyperparameters: Optional dictionary of model hyperparameters.
        dataset_split: Optional dataset partition sample counts.
        model_hash: Unique 12-char model hash.
        metadata: Full metadata dictionary.
        true_positives: Image-level true-positive count.
        false_positives: Image-level false-positive count.
        false_negatives: Image-level false-negative count.
        true_negatives: Image-level true-negative count.
        artifacts: Optional strongly-typed EvaluationArtifacts container. If passed,
            individual metric values are automatically derived from it.

    Returns:
        A dictionary containing structured image_level and pixel_level results.
    """
    if not np.isclose(fpr_limit, AUPIMO_FPR_BOUNDS[1]):
        raise ValueError(f"fair-eval-v1 requires fpr_limit={AUPIMO_FPR_BOUNDS[1]}")

    if artifacts is not None:
        artifacts = EvaluationArtifacts.from_tuple(artifacts)
        manual_image_f1 = artifacts.image_metrics.f1_score
        manual_pixel_f1 = artifacts.pixel_metrics.f1_score
        manual_image_prec = artifacts.image_metrics.precision
        manual_image_rec = artifacts.image_metrics.recall
        img_threshold = artifacts.thresholds.image
        pixel_threshold = artifacts.thresholds.pixel
        pixel_auroc = artifacts.pixel_metrics.auroc
        pixel_aupimo = artifacts.pixel_metrics.aupimo
        anomaly_map_min = artifacts.pixel_metrics.anomaly_map_min
        anomaly_map_max = artifacts.pixel_metrics.anomaly_map_max
        anomaly_map_range = artifacts.pixel_metrics.anomaly_map_range
        heatmap_overlays = artifacts.heatmap_overlays
        anomalous_indices = artifacts.anomalous_indices
        true_positives = artifacts.image_metrics.confusion.true_positives
        false_positives = artifacts.image_metrics.confusion.false_positives
        false_negatives = artifacts.image_metrics.confusion.false_negatives
        true_negatives = artifacts.image_metrics.confusion.true_negatives

    res_dict: Mapping[str, float] = test_results[0] if test_results else {}

    return {
        "category": category,
        "image_level": {
            "auroc": _to_float(res_dict.get("image_AUROC", 0.0)),
            "f1_score": manual_image_f1,
            "precision": manual_image_prec,
            "recall": manual_image_rec,
            "threshold": img_threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "metrics_path": str(base_dir / "image_metrics.npz"),
        },
        "pixel_level": {
            "auroc": pixel_auroc,
            "f1_score": manual_pixel_f1,
            "threshold": pixel_threshold,
            "aupimo_score": pixel_aupimo,
            "fpr_lower_bound": 1e-5,
            "fpr_upper_bound": fpr_limit,
            "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
            "canonical_height": CANONICAL_MAP_SIZE[0],
            "canonical_width": CANONICAL_MAP_SIZE[1],
            "aupimo": pixel_aupimo,
            "anomaly_map_min": anomaly_map_min,
            "anomaly_map_max": anomaly_map_max,
            "anomaly_map_range": anomaly_map_range,
            "metrics_path": str(base_dir / "pixel_metrics.npz"),
        },
        "raw_results": {k: _to_float(v) for k, v in res_dict.items()},
        "heatmap_overlays": heatmap_overlays or {},
        "anomalous_indices": anomalous_indices or [],
        "preprocessing_steps": preprocessing_steps or [],
        "hyperparameters": hyperparameters or {},
        "dataset_split": dataset_split or {},
        "model_hash": model_hash,
        "metadata": metadata or {},
    }
