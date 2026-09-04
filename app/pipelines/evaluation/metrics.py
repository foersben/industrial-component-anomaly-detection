"""Precision-recall metric calculation and persistence functions."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score

CANONICAL_MAP_SIZE = (256, 256)
AUPIMO_FPR_BOUNDS = (1e-5, 1e-4)
AUPIMO_NUM_THRESHOLDS = 50_000
PIXEL_METRICS_VERSION = "shared-pixel-metrics-v1"


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


def canonicalize_pixel_inputs(
    anomaly_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray | None] | np.ndarray,
    image_labels: np.ndarray | list[int],
    size: tuple[int, int] = CANONICAL_MAP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and resize full pixel maps and masks to the shared resolution."""
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
    canonical_maps: list[np.ndarray] = []
    canonical_masks: list[np.ndarray] = []
    for index, (raw_map, raw_mask, label) in enumerate(zip(maps_list, masks_list, labels, strict=True)):
        anomaly_map = np.asarray(raw_map)
        if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
            raise ValueError(f"Anomaly map {index} must be a finite two-dimensional array")
        resized_map = cv2.resize(
            anomaly_map.astype(np.float32),
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        resized_mask: np.ndarray
        if raw_mask is None:
            if label == 1:
                raise ValueError(f"Anomalous image {index} is missing its ground-truth mask")
            resized_mask = np.zeros((target_height, target_width), dtype=np.uint8)
        else:
            mask = np.asarray(raw_mask)
            if mask.ndim != 2:
                raise ValueError(f"Ground-truth mask {index} must be two-dimensional")
            resized_mask = cv2.resize(
                mask.astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            resized_mask = (resized_mask > 0).astype(np.uint8)
            if label == 1 and not resized_mask.any():
                raise ValueError(f"Anomalous image {index} has an empty ground-truth mask")
            if label == 0 and resized_mask.any():
                raise ValueError(f"Normal image {index} has a non-empty ground-truth mask")

        canonical_maps.append(resized_map.astype(np.float32, copy=False))
        canonical_masks.append(resized_mask)

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
