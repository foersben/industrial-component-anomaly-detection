"""Type definitions and protocol constants for the PatchCore pipeline."""

from typing import Any, TypedDict

from app.domain.data import FAIR_EVALUATION_PROTOCOL, FairEvaluationSplit

PATCHCORE_MODEL_SEED: int = 42
PATCHCORE_SCORE_SPACE: str = "raw"
PATCHCORE_IMAGE_THRESHOLD_QUANTILE: float = 0.95
PATCHCORE_PIXEL_THRESHOLD_QUANTILE: float = 0.99


class MetricLevelResult(TypedDict, total=False):
    """Schema for individual evaluation level metrics (image or pixel).

    Attributes:
        auroc: Area Under the Receiver Operating Characteristic Curve.
        average_precision: Average Precision score.
        f1_score: F1 score for the given metric level.
        precision: Precision score.
        recall: Recall score.
        threshold: Decision threshold for classification.
        aupimo_score: Integrated AUPIMO score.
        fpr_lower_bound: Lower bound for FPR integration.
        fpr_upper_bound: Upper bound for FPR integration.
        aupimo: AUPIMO score.
        anomaly_map_min: Minimum anomaly-map score over the test set.
        anomaly_map_max: Maximum anomaly-map score over the test set.
        anomaly_map_range: Difference between maximum and minimum map scores.
        metrics_path: Path to the .npz file containing precision, recall, and thresholds.
        true_positives: Count of true positive predictions.
        false_positives: Count of false positive predictions.
        false_negatives: Count of false negative predictions.
        true_negatives: Count of true negative predictions.
        aupimo_num_thresholds: Number of thresholds used in AUPIMO integration.
        canonical_height: Canonical evaluation map height.
        canonical_width: Canonical evaluation map width.
    """

    auroc: float
    average_precision: float
    f1_score: float
    precision: float
    recall: float
    threshold: float
    aupimo_score: float
    fpr_lower_bound: float
    fpr_upper_bound: float
    aupimo: float
    anomaly_map_min: float
    anomaly_map_max: float
    anomaly_map_range: float
    metrics_path: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    aupimo_num_thresholds: int
    canonical_height: int
    canonical_width: int


class BaselineResult(TypedDict, total=False):
    """Schema for overall PatchCore execution results.

    Attributes:
        category: The specific category being evaluated.
        image_level: Image-level evaluation metrics.
        pixel_level: Pixel-level evaluation metrics.
        raw_results: Raw evaluation results from the Anomalib engine.
        heatmap_overlays: Dictionary of generated heatmap overlays.
        anomalous_indices: List of test dataset indices that are anomalous.
        preprocessing_steps: List of active preprocessing step configurations.
        hyperparameters: Dictionary of model hyperparameters.
        dataset_split: Dictionary of dataset partition sample counts.
        model_hash: Unique 12-char model hash.
        metadata: Full metadata dictionary.
    """

    category: str
    image_level: MetricLevelResult
    pixel_level: MetricLevelResult
    raw_results: dict[str, float]
    heatmap_overlays: dict[int, dict[str, list[Any]]]
    anomalous_indices: list[int]
    preprocessing_steps: list[dict[str, Any]]
    hyperparameters: dict[str, Any]
    dataset_split: dict[str, Any]
    model_hash: str
    metadata: dict[str, Any]


__all__ = [
    "FAIR_EVALUATION_PROTOCOL",
    "PATCHCORE_IMAGE_THRESHOLD_QUANTILE",
    "PATCHCORE_MODEL_SEED",
    "PATCHCORE_PIXEL_THRESHOLD_QUANTILE",
    "PATCHCORE_SCORE_SPACE",
    "BaselineResult",
    "FairEvaluationSplit",
    "MetricLevelResult",
]
