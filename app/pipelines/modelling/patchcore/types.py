"""Type definitions and protocol constants for the PatchCore pipeline."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

PATCHCORE_MODEL_SEED: int = 42
PATCHCORE_SCORE_SPACE: str = "raw"
PATCHCORE_IMAGE_THRESHOLD_QUANTILE: float = 0.95
PATCHCORE_PIXEL_THRESHOLD_QUANTILE: float = 0.99


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion counts for anomalous detection at calibrated threshold.

    Attributes:
        true_positives: Count of correctly identified defective components.
        false_positives: Count of healthy components incorrectly flagged.
        false_negatives: Count of defective components missed.
        true_negatives: Count of healthy components correctly identified.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass(frozen=True)
class Thresholds:
    """Calibrated decision thresholds calibrated strictly on normal validation partition.

    Attributes:
        image: Image-level continuous score classification threshold.
        pixel: Pixel-level anomaly map localization threshold.
    """

    image: float
    pixel: float


@dataclass(frozen=True)
class ImageEvaluationMetrics:
    """Image-level anomaly classification metrics.

    Attributes:
        auroc: Area under the Receiver Operating Characteristic curve.
        f1_score: Harmonic mean of precision and recall at calibrated threshold.
        precision: Fraction of flagged components that are truly defective.
        recall: Fraction of defective components correctly detected.
        threshold: Frozen classification threshold applied.
        confusion: Full confusion matrix counts.
    """

    auroc: float
    f1_score: float
    precision: float
    recall: float
    threshold: float
    confusion: ConfusionMatrix


@dataclass(frozen=True)
class PixelEvaluationMetrics:
    """Pixel-level anomaly localization metrics computed on canonical 256x256 grids.

    Attributes:
        auroc: Pixel-level Area under the ROC curve.
        aupimo: Strict Area under the Per-Image Overlap curve within standard bounds.
        f1_score: Pixel-level binary segmentation F1 score.
        threshold: Frozen segmentation threshold applied.
        anomaly_map_min: Minimum raw continuous score over test set.
        anomaly_map_max: Maximum raw continuous score over test set.
        anomaly_map_range: Continuous score spread (max - min).
    """

    auroc: float
    aupimo: float
    f1_score: float
    threshold: float
    anomaly_map_min: float
    anomaly_map_max: float
    anomaly_map_range: float


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Structured evaluation artifacts, metrics, and visual overlays.

    Attributes:
        image_metrics: Detailed image-level classification performance metrics.
        pixel_metrics: Detailed pixel-level localization performance metrics.
        thresholds: Calibrated frozen thresholds.
        heatmap_overlays: Dictionary mapping test indices to overlay visual arrays.
        anomalous_indices: List of test dataset indices that are ground-truth anomalous.
    """

    image_metrics: ImageEvaluationMetrics
    pixel_metrics: PixelEvaluationMetrics
    thresholds: Thresholds
    heatmap_overlays: dict[int, dict[str, list[Any]]] = field(default_factory=dict)
    anomalous_indices: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        """Return length of legacy tuple representation."""
        return 18

    def __getitem__(self, index: int | slice) -> Any:
        """Support indexing and slicing for legacy tuple compatibility."""
        return tuple(self)[index]

    def __iter__(self) -> Iterator[Any]:
        """Support backwards-compatible unpacking into the legacy 18-tuple."""
        return iter(
            (
                self.image_metrics.f1_score,
                self.pixel_metrics.f1_score,
                self.image_metrics.precision,
                self.image_metrics.recall,
                self.thresholds.image,
                self.pixel_metrics.auroc,
                self.pixel_metrics.aupimo,
                self.pixel_metrics.anomaly_map_min,
                self.pixel_metrics.anomaly_map_max,
                self.pixel_metrics.anomaly_map_range,
                self.heatmap_overlays,
                self.anomalous_indices,
                self.image_metrics.confusion.true_positives,
                self.image_metrics.confusion.false_positives,
                self.image_metrics.confusion.false_negatives,
                self.image_metrics.confusion.true_negatives,
                self.thresholds.pixel,
                self.image_metrics.auroc,
            )
        )

    @classmethod
    def from_tuple(cls, values: Any) -> "EvaluationArtifacts":
        """Construct EvaluationArtifacts from a legacy 18-element tuple or return if already EvaluationArtifacts.

        Args:
            values: An EvaluationArtifacts instance or a sequence of 18 values matching the legacy tuple layout.

        Returns:
            An EvaluationArtifacts instance.

        Raises:
            ValueError: If values is not an EvaluationArtifacts and does not have exactly 18 elements.
        """
        if isinstance(values, cls):
            return values
        raw = tuple(values)
        if len(raw) != 18:
            raise ValueError(f"Expected 18 elements for legacy evaluation tuple, got {len(raw)}")
        (
            image_f1,
            pixel_f1,
            image_precision,
            image_recall,
            image_threshold,
            pixel_auroc,
            pixel_aupimo,
            anomaly_map_min,
            anomaly_map_max,
            anomaly_map_range,
            heatmap_overlays,
            anomalous_indices,
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
            pixel_threshold,
            image_auroc,
        ) = raw

        confusion = ConfusionMatrix(
            true_positives=int(true_positives),
            false_positives=int(false_positives),
            false_negatives=int(false_negatives),
            true_negatives=int(true_negatives),
        )
        thresholds = Thresholds(
            image=float(image_threshold),
            pixel=float(pixel_threshold),
        )
        image_metrics = ImageEvaluationMetrics(
            auroc=float(image_auroc),
            f1_score=float(image_f1),
            precision=float(image_precision),
            recall=float(image_recall),
            threshold=float(image_threshold),
            confusion=confusion,
        )
        pixel_metrics = PixelEvaluationMetrics(
            auroc=float(pixel_auroc),
            aupimo=float(pixel_aupimo),
            f1_score=float(pixel_f1),
            threshold=float(pixel_threshold),
            anomaly_map_min=float(anomaly_map_min),
            anomaly_map_max=float(anomaly_map_max),
            anomaly_map_range=float(anomaly_map_range),
        )
        return cls(
            image_metrics=image_metrics,
            pixel_metrics=pixel_metrics,
            thresholds=thresholds,
            heatmap_overlays=heatmap_overlays,
            anomalous_indices=anomalous_indices,
        )


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
