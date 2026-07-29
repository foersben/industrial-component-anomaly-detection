"""Runs the dummy classifier evaluation to demonstrate the accuracy paradox.

This demonstrates the accuracy paradox in highly imbalanced industrial anomaly
detection.
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Suppress the timm deprecation warning caused by anomalib
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")


def run_real_data_dummy(data_root: str = "data/raw/mvtec_ad", category: str = "bottle") -> None:
    """Run dummy classifier evaluation on real ground truth masks from dataset.

    Args:
        data_root: Path to root directory of MVTec AD dataset.
        category: MVTec AD category name (e.g. 'bottle').
    """
    gt_dir = Path(data_root) / category / "ground_truth"

    if not gt_dir.exists():
        print(f"Error: Ground truth directory not found at {gt_dir}")
        return

    all_true_pixels = []

    # Load all masks for the given category
    for mask_path in gt_dir.rglob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Flatten the 2D mask into a 1D array of pixels
            # Binarize: any pixel > 0 is an anomaly (1), else background (0)
            binary_mask = (mask > 0).astype(np.uint8).flatten()
            all_true_pixels.append(binary_mask)

    if not all_true_pixels:
        print("No masks found to evaluate.")
        return

    y_true = np.concatenate(all_true_pixels)

    # Dummy predictions: predict 0 (normal) for every single pixel
    y_pred = np.zeros_like(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    actual_ratio = (np.sum(y_true) / len(y_true)) * 100

    print(f"--- Real Data Dummy Evaluation ({category}) ---")
    print(f"Total Pixels Evaluated: {len(y_true):,}")
    print(f"Actual Anomalous Pixel Ratio: {actual_ratio:.2f}%")
    print(f"Dummy Accuracy:  {accuracy * 100:.2f}%")
    print(f"Dummy Precision: {precision:.2f}")
    print(f"Dummy Recall:    {recall:.2f}")


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
    dummy_accuracy = accuracy_score(y_true_pixels, y_pred_dummy)
    print(f"Dummy Classifier Accuracy: {dummy_accuracy * 100:.2f}%")
    return float(dummy_accuracy)


if __name__ == "__main__":
    run_dummy_evaluation()
