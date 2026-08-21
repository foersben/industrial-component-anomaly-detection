"""Evaluation metrics calculation and persistence functions."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve


def compute_aupimo_lower_bound(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
    fpr_limit: float = 1e-4,
) -> float:
    """Computes the AUPIMO minimum threshold bound based on normal background samples.

    Args:
        y_true: 1D array of ground truth labels (0 = normal, 1 = anomalous).
        y_score: 1D array of predicted anomaly scores.
        fpr_limit: Maximum allowable False Positive Rate (default 1e-4 = 0.01%).

    Returns:
        The minimum threshold corresponding to the FPR limit, or 0.0 if no normal samples exist.
    """
    normal_scores = y_score[y_true == 0]
    if len(normal_scores) == 0:
        return 0.0
    return float(np.quantile(normal_scores, 1.0 - fpr_limit))


def save_evaluation_metrics(
    output_path: str | Path,
    precisions: Any,
    recalls: Any,
    thresholds: Any,
    t_aupimo_min: float = 0.0,
    aupimo: float = 0.0,
    level: str = "pixel",
) -> Path:
    """Saves precision, recall, threshold arrays, and AUPIMO bound to an .npz file.

    Args:
        output_path: Target filepath (e.g. 'results/Patchcore/bottle/pixel_metrics.npz').
        precisions: Precision values array.
        recalls: Recall values array.
        thresholds: Binarization thresholds array.
        t_aupimo_min: AUPIMO minimum threshold bound.
        aupimo: AUPIMO recall metric score at the threshold limit.
        level: Evaluation level ('pixel' for localization, 'image' for classification).

    Returns:
        The saved Path object.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        precision=precisions,
        recall=recalls,
        thresholds=thresholds,
        t_aupimo_min=t_aupimo_min,
        aupimo=aupimo,
        level=level,
    )
    return path


def compute_and_save_pr_metrics(
    y_true: Any,
    y_score: Any,
    output_path: str | Path,
    level: str = "pixel",
    fpr_limit: float = 1e-4,
) -> Path:
    """Computes PR metrics + AUPIMO threshold bound and saves them to .npz.

    Args:
        y_true: 1D array of ground truth binary labels (0 or 1).
        y_score: 1D array of predicted anomaly scores.
        output_path: Destination .npz file path.
        level: Evaluation level ('pixel' for localization, 'image' for classification).
        fpr_limit: Maximum allowable False Positive Rate (default 1e-4 = 0.01%).

    Returns:
        The saved Path object.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)

    t_aupimo_min = 0.0
    aupimo = 0.0

    if level == "pixel":
        t_aupimo_min = compute_aupimo_lower_bound(y_true_arr, y_score_arr, fpr_limit=fpr_limit)
        if t_aupimo_min > 0 and len(thresholds) > 0:
            idx = np.argmax(thresholds >= t_aupimo_min)
            if idx == 0 and thresholds[0] < t_aupimo_min:
                aupimo = 0.0
            else:
                aupimo = recall[idx]

    return save_evaluation_metrics(
        output_path,
        precision,
        recall,
        thresholds,
        t_aupimo_min=t_aupimo_min,
        aupimo=aupimo,
        level=level,
    )
