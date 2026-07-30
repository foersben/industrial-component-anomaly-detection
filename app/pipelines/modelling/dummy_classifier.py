"""Runs the dummy classifier evaluation to demonstrate the accuracy paradox.

This demonstrates the accuracy paradox in highly imbalanced industrial anomaly
detection.
"""

import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from app.core.logger import logger

# Suppress the timm deprecation warning caused by anomalib
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")


def run_real_data_dummy(data_root: str = "data/raw/mvtec_ad", category: str = "bottle") -> dict[str, Any]:
    """Run dummy classifier evaluation on real ground truth masks from dataset.

    Args:
        data_root: Path to root directory of MVTec AD dataset.
        category: MVTec AD category name (e.g. 'bottle').

    Returns:
        Dictionary containing evaluation metrics and summary lines.
    """
    gt_dir = Path(data_root) / category / "ground_truth"

    if not gt_dir.exists():
        msg = f"Error: Ground truth directory not found at {gt_dir}"
        logger.error(msg)
        return {"error": msg}

    all_true_pixels = []

    # Load all masks for the given category
    for mask_path in gt_dir.rglob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Flatten the 2D mask into a 1D array of pixels
            # any pixel > 0 is an anomaly (1), else background (0)
            binary_mask = np.where(mask > 0, 1, 0).flatten()
            all_true_pixels.append(binary_mask)

    if not all_true_pixels:
        msg = "No masks found to evaluate."
        logger.warning(msg)
        return {"error": msg}

    y_true = np.concatenate(all_true_pixels)

    # Dummy predictions: predict 0 (normal) for every single pixel
    y_pred = np.zeros_like(y_true)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    actual_ratio = (np.sum(y_true) / len(y_true)) * 100
    total_pixels = len(y_true)

    summary_lines = [
        f"--- Real Data Dummy Evaluation ({category}) ---",
        f"Total Pixels Evaluated: {total_pixels:,}",
        f"Actual Anomalous Pixel Ratio: {actual_ratio:.2f}%",
        f"Dummy Accuracy:  {accuracy * 100:.2f}%",
        f"Dummy Precision: {precision:.2f}",
        f"Dummy Recall:    {recall:.2f}",
    ]

    for line in summary_lines:
        logger.info(line)

    return {
        "category": category,
        "total_pixels": total_pixels,
        "actual_ratio": actual_ratio,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "summary": "\n".join(summary_lines),
    }


def run_dummy_evaluation(total_pixels: int = 1000000, anomaly_ratio: float = 0.015) -> float:
    """Run a dummy classifier evaluation to demonstrate the accuracy paradox.

    This demonstrates the accuracy paradox in highly imbalanced industrial anomaly
    detection.

    Args:
        total_pixels: Total number of pixels to simulate
        anomaly_ratio: Ratio of anomalous pixels (default 1.5%)

    Returns:
        The accuracy score
    """
    # 1. Simulate the true masks for a typical test set
    num_anomalies = int(total_pixels * anomaly_ratio)
    y_true_pixels = np.zeros(total_pixels)
    y_true_pixels[:num_anomalies] = 1

    # 2. The Dummy Classifier: Predicts 0 (normal) for everything
    y_pred_dummy = np.zeros(total_pixels)

    # 3. Calculate Accuracy
    dummy_accuracy = float(accuracy_score(y_true_pixels, y_pred_dummy))
    logger.info("Dummy Classifier Accuracy: %.2f%%", dummy_accuracy * 100)
    return dummy_accuracy


if __name__ == "__main__":
    run_dummy_evaluation()
