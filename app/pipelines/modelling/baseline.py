"""Runs the baseline Patchcore model on the MVTec AD dataset.

This serves as a functional baseline with PR-AUC evaluation.
"""

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from app.core.logger import logger
from app.pipelines.evaluation.metrics import compute_and_save_pr_metrics
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.factory import build_pipeline_from_configs

# Suppress the timm deprecation warning caused by anomalib
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")


class MetricLevelResult(TypedDict, total=False):
    """Schema for individual evaluation level metrics (image or pixel).

    Attributes:
        auroc: Area Under the Receiver Operating Characteristic Curve.
        f1_score: F1 score for the given metric level.
        t_aupimo_min: Minimum AUPIMO threshold bound (for pixel localization).
        metrics_path: Path to the .npz file containing precision, recall, and thresholds.
    """

    auroc: float
    f1_score: float
    t_aupimo_min: float
    metrics_path: str


class BaselineResult(TypedDict):
    """Schema for overall baseline execution results.

    Attributes:
        category: The specific category being evaluated.
        image_level: Image-level evaluation metrics.
        pixel_level: Pixel-level evaluation metrics.
        raw_results: Raw evaluation results from the anomalib engine.
    """

    category: str
    image_level: MetricLevelResult
    pixel_level: MetricLevelResult
    raw_results: dict[str, float]


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
    """Helper to safely extract a 1D NumPy array from a PyTorch tensor attribute.

    Args:
        val: Value to convert to NumPy array.

    Returns:
        NumPy array representation of the value if it is a PyTorch tensor, None otherwise.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().reshape(-1)
    return None


def _collect_batch_tensors(
    batch: Any,
    score_attr: str,
    label_attr: str,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
    """Extracts score and label arrays from a batch if both attributes exist and are tensors.

    Args:
        batch: Batch to extract tensors from.
        score_attr: Attribute name for scores.
        label_attr: Attribute name for labels.

    Returns:
        Tuple of score and label arrays if both attributes exist and are tensors, None otherwise.
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
    fpr_limit: float = 1e-4,
) -> None:
    """Concatenates prediction lists and saves metrics if data is present.

    Args:
        scores_list: List of anomaly scores.
        labels_list: List of ground truth labels.
        output_path: Path to save metrics to.
        level: Level of metrics (image or pixel).
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
    """
    if not (scores_list and labels_list):
        return

    y_score = np.concatenate(scores_list)
    y_true = np.concatenate(labels_list)

    compute_and_save_pr_metrics(y_true, y_score, output_path, level=level, fpr_limit=fpr_limit)
    logger.info("Saved %s-level PR metrics to %s", level, output_path)


def extract_and_save_pr_metrics(
    engine: Engine,
    model: Patchcore,
    datamodule: MVTecAD,
    base_dir: Path,
    fpr_limit: float = 1e-4,
) -> None:
    """Extract model predictions and persist Precision-Recall metrics for visual analysis.

    Args:
        engine: Anomalib engine instance.
        model: Trained model.
        datamodule: Dataset object.
        base_dir: Output directory for metrics.
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
    """
    try:
        logger.info("Extracting predictions for PR curve metrics...")
        predictions = engine.predict(model=model, datamodule=datamodule)
        if not predictions:
            return

        pixel_scores, pixel_labels = [], []
        image_scores, image_labels = [], []

        for batch in predictions:
            if px := _collect_batch_tensors(batch, "anomaly_map", "gt_mask"):
                pixel_scores.append(px[0])
                pixel_labels.append(px[1])

            if img := _collect_batch_tensors(batch, "pred_score", "gt_label"):
                image_scores.append(img[0])
                image_labels.append(img[1])

        _process_and_save_level(
            pixel_scores, pixel_labels, base_dir / "pixel_metrics.npz", level="pixel", fpr_limit=fpr_limit
        )
        _process_and_save_level(
            image_scores, image_labels, base_dir / "image_metrics.npz", level="image", fpr_limit=fpr_limit
        )

    except Exception as e:
        logger.warning("Could not auto-save evaluation metrics.npz: %s", e)


def format_results(
    test_results: list[Mapping[str, float]] | None,
    category: str,
    base_dir: Path,
) -> BaselineResult:
    """Format anomalib engine evaluation output into a structured response schema.

    Args:
        test_results: Evaluation results from anomalib engine.
        category: The specific category being evaluated.
        base_dir: Base directory to save metrics to.

    Returns:
        Structured dictionary containing test metrics and artifact paths.
    """
    res_dict: Mapping[str, float] = test_results[0] if test_results else {}

    t_aupimo_min = 0.0
    pixel_file = base_dir / "pixel_metrics.npz"
    if pixel_file.exists():
        try:
            data = np.load(pixel_file)
            if "t_aupimo_min" in data:
                t_aupimo_min = float(data["t_aupimo_min"])
        except Exception:
            pass

    return {
        "category": category,
        "image_level": {
            "auroc": _to_float(res_dict.get("image_AUROC", 0.0)),
            "f1_score": _to_float(res_dict.get("image_F1Score", 0.0)),
            "metrics_path": str(base_dir / "image_metrics.npz"),
        },
        "pixel_level": {
            "auroc": _to_float(res_dict.get("pixel_AUROC", 0.0)),
            "f1_score": _to_float(res_dict.get("pixel_F1Score", 0.0)),
            "t_aupimo_min": t_aupimo_min,
            "metrics_path": str(base_dir / "pixel_metrics.npz"),
        },
        "raw_results": {k: _to_float(v) for k, v in res_dict.items()},
    }


def run_baseline(
    data_root: str = "data/raw/mvtec_ad",
    category: str = "bottle",
    fpr_limit: float = 1e-4,
    preprocessing_steps: list[dict[str, Any]] | None = None,
) -> BaselineResult:
    """Run the baseline Patchcore model on the MVTec AD dataset.

    Args:
        data_root: Path to the root directory of the MVTec AD dataset.
        category: The specific category to evaluate (e.g., 'bottle').
        fpr_limit: Maximum allowable False Positive Rate for AUPIMO threshold.
        preprocessing_steps: List of preprocessing step configurations.

    Returns:
        Structured dictionary containing test metrics and artifact paths.
    """
    pipeline = build_pipeline_from_configs(preprocessing_steps)

    logger.info("Configured preprocessing pipeline with %d steps.", len(pipeline))

    base_dir = Path("results") / "Patchcore" / category

    transform_adapter = PreprocessingTransformAdapter(pipeline)

    # 1. Initialize dataset, model, and engine
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
    )

    if len(pipeline) > 0:
        # Setup the datamodule datasets so train_data and test_data are instantiated
        datamodule.setup()

        # Assign the adapter transform to the underlying datasets
        train_data = getattr(datamodule, "train_data", None)
        if train_data is not None:
            train_data.transform = transform_adapter

        test_data = getattr(datamodule, "test_data", None)
        if test_data is not None:
            test_data.transform = transform_adapter

    model = Patchcore(backbone="resnet18")
    engine = Engine(accelerator="gpu", devices=1)

    # 2. Fit and Test
    logger.info("Fitting Patchcore model on %s category...", category)
    engine.fit(model, datamodule)

    logger.info("Testing Patchcore model...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # 3. Extract PR metrics and build summary
    extract_and_save_pr_metrics(engine, model, datamodule, base_dir, fpr_limit)

    return format_results(test_results, category, base_dir)


if __name__ == "__main__":
    run_baseline()
