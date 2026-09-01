"""Precision-recall metric calculation and persistence functions."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve


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
