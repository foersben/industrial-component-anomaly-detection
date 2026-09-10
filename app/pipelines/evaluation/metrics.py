"""Precision-recall metric calculation and persistence functions."""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)

CANONICAL_MAP_SIZE = (256, 256)
AUPIMO_FPR_BOUNDS = (1e-5, 1e-4)
AUPIMO_NUM_THRESHOLDS = 50_000
PIXEL_METRICS_VERSION = "shared-pixel-metrics-v1"


def compute_image_auroc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """Compute image-level Area Under the ROC Curve (AUROC).

    Args:
        scores: 1D array of image-level anomaly scores, shape (N,). Higher = more anomalous.
        binary_labels: 1D binary array, shape (N,). 0 = normal, 1 = anomalous.

    Returns:
        AUROC value in [0, 1]. 1.0 = perfect; 0.5 = random; 0.0 = perfectly inverted.

    Raises:
        ValueError: If fewer than 2 distinct classes are present in binary_labels.
    """
    if len(np.unique(binary_labels)) < 2:
        raise ValueError("Image AUROC requires both normal and anomalous labels")

    auroc: float = float(roc_auc_score(binary_labels, scores))
    logger.info("Image-Level AUROC: %.4f", auroc)
    return auroc


def compute_aupimo(
    anomaly_maps: list[np.ndarray],
    gt_masks: list[np.ndarray | None],
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
) -> float:
    """Compute pixel-level AUPIMO using anomalib's implementation.

    AUPIMO (Area Under Per-Image Overlap) integrates per-image pixel overlap between
    predicted anomaly maps and ground truth defect masks. It does so only over an
    extremely narrow and industrially realistic FPR range (default: 10⁻⁵ to 10⁻⁴).

    Args:
        anomaly_maps: List of 2D pixel anomaly score maps, one per test image.
            Each map has shape (H, W) with float values >= 0 (higher = more anomalous).
        gt_masks: List of 2D ground truth binary masks, one per test image.
            Each mask has shape (H, W) with values 0 (normal) or 1 (defect).
            Use None for images with no ground truth mask (normal images).
        fpr_bounds: Tuple (lower_fpr, upper_fpr) defining the integration interval.
            Default (1e-5, 1e-4) matches the MVTec AD benchmark standard.

    Returns:
        AUPIMO score in [0, 1]. Higher is better.

    Raises:
        ImportError: If anomalib or torch is unavailable.
        ValueError: If the maps or masks cannot define AUPIMO at the requested bounds.
        RuntimeError: If anomalib cannot compute the metric at the requested bounds.
    """
    try:
        import torch
        from anomalib.data import ImageBatch
        from anomalib.metrics import AUPIMO
    except ImportError as exc:
        raise ImportError("anomalib and torch are required to compute AUPIMO") from exc

    # Format masks: normal images have zeros mask, defective have binary mask
    h, w = anomaly_maps[0].shape
    all_masks: list[np.ndarray] = []
    has_anomaly = False
    for mask in gt_masks:
        if mask is not None and np.any(mask > 0):
            all_masks.append(mask.astype(np.uint8))
            has_anomaly = True
        else:
            all_masks.append(np.zeros((h, w), dtype=np.uint8))

    if not has_anomaly:
        raise ValueError("AUPIMO requires at least one anomalous image with a ground-truth mask")

    pred_tensor = torch.tensor(np.stack(anomaly_maps), dtype=torch.float32)
    gt_tensor = torch.tensor(np.stack(all_masks), dtype=torch.bool)
    dummy_img = torch.zeros(len(anomaly_maps), 3, h, w, dtype=torch.float32)

    batch = ImageBatch(image=dummy_img, anomaly_map=pred_tensor, gt_mask=gt_tensor)

    aupimo_metric = AUPIMO(num_thresholds=50_000, fpr_bounds=fpr_bounds)
    aupimo_metric.update(batch)
    result = aupimo_metric.compute()
    if hasattr(result, "aupimo_scores"):
        score = float(result.aupimo_scores.nanmean().item())
    elif isinstance(result, tuple) and len(result) > 1:
        score = float(result[1].nanmean().item())
    elif isinstance(result, dict):
        score = float(next(iter(result.values())))
    else:
        score = float(result)

    if np.isnan(score):
        raise RuntimeError(
            f"AUPIMO computation returned NaN for fpr_bounds={fpr_bounds}. "
            "Verify that anomaly maps contain sufficiently diverse continuous scores."
        )

    logger.info("Pixel-Level AUPIMO: %.4f (bounds: %s)", score, fpr_bounds)
    return score


def fair_metric_evidence() -> dict[str, Any]:
    """Return the metric and calibration fields required for a fair cache hit."""
    return {
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "aupimo_fpr_bounds": list(AUPIMO_FPR_BOUNDS),
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "threshold_source": "normal_validation",
        "pixel_metrics_version": PIXEL_METRICS_VERSION,
    }


def _canonicalize_single_map(
    raw_map: Any,
    target_width: int,
    target_height: int,
    index: int,
) -> np.ndarray:
    """Validate and resize a single continuous anomaly map to canonical dimensions.

    Args:
        raw_map: Array-like object representing a 2D anomaly heatmap.
        target_width: Canonical output grid width (typically 256).
        target_height: Canonical output grid height (typically 256).
        index: Sample index within the evaluation batch for error reporting.

    Returns:
        A 2D float32 NumPy array resized with bilinear interpolation.

    Raises:
        ValueError: If the map is not a 2D array or contains non-finite values (NaN, Inf).
    """
    anomaly_map = np.asarray(raw_map)
    if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
        raise ValueError(f"Anomaly map {index} must be a finite two-dimensional array")
    resized_map = cv2.resize(
        anomaly_map.astype(np.float32),
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized_map.astype(np.float32, copy=False)


def _canonicalize_single_mask(
    raw_mask: Any,
    label: int,
    target_width: int,
    target_height: int,
    index: int,
) -> np.ndarray:
    """Validate, resize, and binarize a ground-truth segmentation mask.

    Args:
        raw_mask: Optional 2D ground-truth mask array or None for normal samples.
        label: Binary ground-truth image label (0 for normal, 1 for defective).
        target_width: Canonical output mask width (typically 256).
        target_height: Canonical output mask height (typically 256).
        index: Sample index within the evaluation batch for error reporting.

    Returns:
        A 2D uint8 NumPy array containing only binary values {0, 1}.

    Raises:
        ValueError: If a defective image lacks a mask, has an empty mask, if a normal image
            has a non-empty mask, or if the mask is not 2D.
    """
    if raw_mask is None:
        if label == 1:
            raise ValueError(f"Anomalous image {index} is missing its ground-truth mask")
        return np.zeros((target_height, target_width), dtype=np.uint8)

    mask = np.asarray(raw_mask)
    if mask.ndim != 2:
        raise ValueError(f"Ground-truth mask {index} must be two-dimensional")

    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    binarized = (resized_mask > 0).astype(np.uint8)

    if label == 1 and not binarized.any():
        raise ValueError(f"Anomalous image {index} has an empty ground-truth mask")
    if label == 0 and binarized.any():
        raise ValueError(f"Normal image {index} has a non-empty ground-truth mask")

    return binarized


def canonicalize_pixel_inputs(
    anomaly_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray | None] | np.ndarray,
    image_labels: np.ndarray | list[int],
    size: tuple[int, int] = CANONICAL_MAP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and resize full pixel maps and masks to the shared canonical resolution.

    Args:
        anomaly_maps: Sequence of continuous 2D anomaly heatmaps.
        masks: Sequence of ground-truth binary masks (or None for normal images).
        image_labels: Binary image-level defect labels (0 or 1).
        size: Target canonical resolution tuple (height, width). Defaults to 256x256.

    Returns:
        A tuple of stacked (canonical_maps, canonical_masks) NumPy arrays.

    Raises:
        ValueError: If inputs are empty, have mismatched lengths, or contain invalid labels.
    """
    maps_list = list(anomaly_maps)
    masks_list = list(masks)
    labels = np.asarray(image_labels, dtype=np.uint8).reshape(-1)
    if not maps_list:
        raise ValueError("Pixel evaluation requires at least one anomaly map")
    if len(maps_list) != len(masks_list) or len(maps_list) != len(labels):
        raise ValueError("Anomaly maps, masks, and image labels must have equal counts")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Image labels must be binary")

    target_height, target_width = size
    canonical_maps = [
        _canonicalize_single_map(raw_map, target_width, target_height, idx) for idx, raw_map in enumerate(maps_list)
    ]
    canonical_masks = [
        _canonicalize_single_mask(raw_mask, int(label), target_width, target_height, idx)
        for idx, (raw_mask, label) in enumerate(zip(masks_list, labels, strict=True))
    ]

    return np.stack(canonical_maps), np.stack(canonical_masks)


def compute_shared_pixel_metrics(
    anomaly_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray | None] | np.ndarray,
    image_labels: np.ndarray | list[int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute canonical pixel AUROC and genuine full-map AUPIMO."""
    canonical_maps, canonical_masks = canonicalize_pixel_inputs(anomaly_maps, masks, image_labels)
    flat_masks = canonical_masks.reshape(-1)
    if len(np.unique(flat_masks)) != 2:
        raise ValueError("Pixel AUROC requires both normal and anomalous pixels")
    pixel_auroc = float(roc_auc_score(flat_masks, canonical_maps.reshape(-1)))

    # Import lazily to avoid an evaluation-module import cycle and to keep this
    # helper patchable in focused tests.
    from app.pipelines.evaluation.cae_metrics import compute_aupimo

    pixel_aupimo = compute_aupimo(
        [item for item in canonical_maps],
        [item for item in canonical_masks],
        fpr_bounds=AUPIMO_FPR_BOUNDS,
    )
    return (
        {
            "pixel_auroc": pixel_auroc,
            "pixel_aupimo": pixel_aupimo,
            "aupimo_fpr_lower": AUPIMO_FPR_BOUNDS[0],
            "aupimo_fpr_upper": AUPIMO_FPR_BOUNDS[1],
            "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
            "canonical_height": CANONICAL_MAP_SIZE[0],
            "canonical_width": CANONICAL_MAP_SIZE[1],
            "pixel_metrics_version": PIXEL_METRICS_VERSION,
        },
        canonical_maps,
        canonical_masks,
    )


def compute_image_confusion_metrics(labels: Any, scores: Any, threshold: float) -> dict[str, float | int]:
    """Calculate image confusion counts and derived metrics at a frozen threshold."""
    y_true = np.asarray(labels, dtype=np.uint8).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_score):
        raise ValueError("Image labels and scores must be non-empty and have equal counts")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Image labels must be binary")
    predictions = (y_score > threshold).astype(np.uint8)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_negatives": int(true_negatives),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def save_evaluation_metrics(
    output_path: str | Path,
    precisions: Any,
    recalls: Any,
    thresholds: Any,
    aupimo: float | None = None,
    fpr_bounds: tuple[float, float] | None = None,
    level: str = "pixel",
) -> Path:
    """Save precision, recall, and threshold arrays to an ``.npz`` file.

    Args:
        output_path: Target filepath (e.g. 'results/Patchcore/bottle/pixel_metrics.npz').
        precisions: Precision values array.
        recalls: Recall values array.
        thresholds: Binarization thresholds array.
        aupimo: Genuine full-map AUPIMO score, when available.
        fpr_bounds: FPR integration bounds used for AUPIMO, when available.
        level: Evaluation level ('pixel' for localization, 'image' for classification).

    Returns:
        The saved Path object.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "precision": precisions,
        "recall": recalls,
        "thresholds": thresholds,
        "level": level,
    }
    if aupimo is not None:
        values["aupimo"] = aupimo
    if fpr_bounds is not None:
        values["aupimo_fpr_bounds"] = np.asarray(fpr_bounds, dtype=np.float64)
    np.savez(path, **values)
    return path


def compute_and_save_pr_metrics(
    y_true: Any,
    y_score: Any,
    output_path: str | Path,
    level: str = "pixel",
    aupimo: float | None = None,
    fpr_bounds: tuple[float, float] | None = None,
) -> Path:
    """Compute PR metrics and save them with an optional genuine AUPIMO score.

    Args:
        y_true: 1D array of ground truth binary labels (0 or 1).
        y_score: 1D array of predicted anomaly scores.
        output_path: Destination .npz file path.
        level: Evaluation level ('pixel' for localization, 'image' for classification).
        aupimo: Genuine full-map AUPIMO score computed separately from 2D maps.
        fpr_bounds: FPR integration bounds used for AUPIMO.

    Returns:
        The saved Path object.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)

    return save_evaluation_metrics(
        output_path,
        precision,
        recall,
        thresholds,
        aupimo=aupimo,
        fpr_bounds=fpr_bounds,
        level=level,
    )
