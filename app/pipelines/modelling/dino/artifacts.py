"""Artifact persistence, serialization, and summary utilities for DINO pipelines."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from app.core.logger import logger
from app.domain.data import FAIR_EVALUATION_PROTOCOL
from app.domain.evaluation import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
)
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    PIXEL_METRICS_VERSION,
)
from app.pipelines.modelling.anomalib.visualization import (
    load_heatmap_overlays,
    print_anomalib_results_table,
    save_heatmap_overlays,
)
from app.pipelines.modelling.patchcore.evaluation import format_results

if TYPE_CHECKING:
    from app.domain.evaluation import BaselineResult, EvaluationArtifacts
    from app.pipelines.modelling.dino.types import DINOVariant, MaskingMode


def load_completed_category_result(
    base_dir: Path,
    category: str,
    model_hash: str,
    run_heatmap: bool,
    load_heatmaps: bool = True,
) -> BaselineResult | None:
    """Load a complete category result without retaining saved heatmap pixels.

    Args:
        base_dir: Directory containing candidate evaluation artifacts.
        category: MVTec category expected in the metadata.
        model_hash: Unique SHA-256 fingerprint expected in the metadata.
        run_heatmap: Whether heatmap overlays are required to declare completion.
        load_heatmaps: Whether to deserialize heatmap pixels into the returned result.

    Returns:
        Reconstituted BaselineResult if complete artifacts exist, or None.
    """
    metadata_path = base_dir / "metadata.json"
    required_artifacts = (base_dir / "image_metrics.npz", base_dir / "pixel_metrics.npz")
    if not metadata_path.is_file() or not all(path.is_file() for path in required_artifacts):
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("category") != category or metadata.get("hash") != model_hash:
            return None
        heatmap_path = metadata.get("heatmap_overlays_path")
        if run_heatmap and (not heatmap_path or not (base_dir / str(heatmap_path)).is_file()):
            return None

        heatmap_overlays = load_heatmap_overlays(base_dir / str(heatmap_path)) if heatmap_path and load_heatmaps else {}
        raw_results = metadata["raw_results"]
        result = format_results(
            test_results=[raw_results],
            category=category,
            base_dir=base_dir,
            manual_image_f1=float(metadata["image_f1"]),
            manual_pixel_f1=float(metadata["pixel_f1"]),
            manual_image_prec=float(metadata["image_precision"]),
            manual_image_rec=float(metadata["image_recall"]),
            img_threshold=float(metadata["image_threshold"]),
            pixel_threshold=float(metadata["pixel_threshold"]),
            pixel_auroc=float(metadata["pixel_auroc"]),
            pixel_aupimo=float(metadata["pixel_aupimo"]),
            anomaly_map_min=float(metadata["anomaly_map_min"]),
            anomaly_map_max=float(metadata["anomaly_map_max"]),
            anomaly_map_range=float(metadata["anomaly_map_range"]),
            heatmap_overlays=heatmap_overlays,
            anomalous_indices=list(metadata.get("anomalous_indices", [])),
            preprocessing_steps=list(metadata.get("preprocessing_steps", [])),
            hyperparameters=dict(metadata.get("hyperparameters", {})),
            dataset_split=dict(metadata.get("dataset_split", {})),
            model_hash=model_hash,
            metadata=metadata,
            true_positives=int(metadata["true_positives"]),
            false_positives=int(metadata["false_positives"]),
            false_negatives=int(metadata["false_negatives"]),
            true_negatives=int(metadata["true_negatives"]),
        )
        result["image_level"]["average_precision"] = float(metadata["image_average_precision"])
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring incomplete DINO artifacts in %s: %s", base_dir, exc)
        return None


def persist_dino_artifacts_and_format(
    category: str,
    base_dir: Path,
    model_hash: str,
    configuration_hash: str,
    model_generation: Literal["dinov2", "dinov3"],
    variant: DINOVariant,
    encoder_name: str,
    num_neighbors: int,
    masking: MaskingMode,
    use_masking: bool,
    raw_prep_list: list[dict[str, Any]],
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    artifacts: EvaluationArtifacts,
    image_average_precision: float,
    fpr_limit: float,
    model_seed: int,
) -> BaselineResult:
    """Serialize DINO run metadata and format standardized baseline result dictionary.

    Args:
        category: MVTec category evaluated.
        base_dir: Base directory where artifacts are saved.
        model_hash: Unique run hash string.
        configuration_hash: Stable hash for equivalent model configurations.
        model_generation: Encoder architecture family.
        variant: Scorer variant ('baseline' or 'enhanced').
        encoder_name: Model backbone identifier.
        num_neighbors: Nearest neighbor count.
        masking: Foreground masking policy mode.
        use_masking: Evaluated boolean decision for masking.
        raw_prep_list: Normalized preprocessing configuration list.
        hyperparameters: Model hyperparameter dictionary.
        split_info: Dataset partition sample counts.
        artifacts: Structured EvaluationArtifacts container.
        image_average_precision: Computed area under Precision-Recall curve.
        fpr_limit: Maximum false-positive rate for AUPIMO integration.
        model_seed: Deterministic random seed used for the run.

    Returns:
        Structured BaselineResult dictionary complying with fair-eval-v1.
    """
    raw_results: dict[str, float] = {
        "image_F1Score": artifacts.image_metrics.f1_score,
        "image_Precision": artifacts.image_metrics.precision,
        "image_Recall": artifacts.image_metrics.recall,
        "image_AUROC": artifacts.image_metrics.auroc,
        "image_AP": image_average_precision,
        "pixel_AUROC": artifacts.pixel_metrics.auroc,
        "pixel_F1Score": artifacts.pixel_metrics.f1_score,
        "pixel_AUPIMO": artifacts.pixel_metrics.aupimo,
    }
    print_anomalib_results_table(raw_results)

    heatmap_archive = save_heatmap_overlays(artifacts.heatmap_overlays, base_dir / "heatmap_overlays.npz")
    four_panel_dir = base_dir / "four_panel_images"
    metadata = {
        "hash": model_hash,
        "configuration_hash": configuration_hash,
        "model_type": f"{model_generation}_enhanced_knn" if variant == "enhanced" else f"{model_generation}_knn",
        "model_generation": model_generation,
        "category": category,
        "encoder_name": encoder_name,
        "num_neighbors": num_neighbors,
        "masking_mode": masking,
        "masking": use_masking,
        "coreset_subsampling": False,
        "variant": variant,
        "preprocessing_steps": raw_prep_list,
        "hyperparameters": hyperparameters,
        "dataset_split": split_info,
        "protocol": FAIR_EVALUATION_PROTOCOL,
        "threshold_source": "normal_validation",
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "pixel_metrics_version": PIXEL_METRICS_VERSION,
        "image_f1": artifacts.image_metrics.f1_score,
        "image_precision": artifacts.image_metrics.precision,
        "image_recall": artifacts.image_metrics.recall,
        "image_auroc": artifacts.image_metrics.auroc,
        "image_average_precision": image_average_precision,
        "pixel_f1": artifacts.pixel_metrics.f1_score,
        "pixel_auroc": artifacts.pixel_metrics.aupimo,
        "pixel_aupimo": artifacts.pixel_metrics.aupimo,
        "true_positives": artifacts.image_metrics.confusion.true_positives,
        "false_positives": artifacts.image_metrics.confusion.false_positives,
        "false_negatives": artifacts.image_metrics.confusion.false_negatives,
        "true_negatives": artifacts.image_metrics.confusion.true_negatives,
        "image_threshold": artifacts.thresholds.image,
        "pixel_threshold": artifacts.thresholds.pixel,
        "aupimo_fpr_bounds": list(AUPIMO_FPR_BOUNDS),
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "anomaly_map_min": artifacts.pixel_metrics.anomaly_map_min,
        "anomaly_map_max": artifacts.pixel_metrics.anomaly_map_max,
        "anomaly_map_range": artifacts.pixel_metrics.anomaly_map_range,
        "heatmap_overlays_path": heatmap_archive.name if heatmap_archive is not None else None,
        "four_panel_images_path": four_panel_dir.name if four_panel_dir.is_dir() else None,
        "anomalous_indices": artifacts.anomalous_indices,
        "raw_results": raw_results,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (base_dir / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    results = format_results(
        test_results=[raw_results],
        category=category,
        base_dir=base_dir,
        artifacts=artifacts,
        fpr_limit=fpr_limit,
        preprocessing_steps=raw_prep_list,
        hyperparameters=hyperparameters,
        dataset_split=split_info,
        model_hash=model_hash,
        metadata=metadata,
    )
    results["image_level"]["average_precision"] = image_average_precision
    return results


def save_all_category_summary(
    registry_base: Path | str,
    encoder_name: str,
    masking: MaskingMode,
    num_neighbors: int,
    category_results: dict[str, BaselineResult],
    macro_average: dict[str, float],
    model_name: str = "DINO",
) -> None:
    """Persist compact machine-readable summaries beside category artifacts.

    Args:
        registry_base: Target directory where summary files are placed.
        encoder_name: Name of the encoder backbone used.
        masking: Foreground masking policy mode.
        num_neighbors: Nearest neighbor count.
        category_results: Mapping of category names to their BaselineResults.
        macro_average: Unweighted metric averages across categories.
        model_name: Label used in the written JSON summary.
    """
    output_dir = Path(registry_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = tuple(macro_average)
    category_metrics: dict[str, dict[str, float]] = {}
    for name, result in category_results.items():
        category_metrics[name] = {
            "image_f1": float(result["image_level"]["f1_score"]),
            "image_recall": float(result["image_level"]["recall"]),
            "image_precision": float(result["image_level"]["precision"]),
            "image_auroc": float(result["image_level"]["auroc"]),
            "image_average_precision": float(result["image_level"]["average_precision"]),
            "pixel_f1": float(result["pixel_level"]["f1_score"]),
            "pixel_auroc": float(result["pixel_level"]["auroc"]),
            "pixel_aupimo": float(result["pixel_level"]["aupimo"]),
        }

    summary = {
        "model": model_name,
        "variant": "baseline_single_layer_1nn" if num_neighbors == 1 else f"baseline_single_layer_{num_neighbors}nn",
        "encoder_name": encoder_name,
        "masking_mode": masking,
        "categories": len(category_results),
        "protocol": FAIR_EVALUATION_PROTOCOL,
        "macro_average": macro_average,
        "category_metrics": category_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "category_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("category", *metric_keys))
        writer.writeheader()
        for name, metrics in category_metrics.items():
            writer.writerow({"category": name, **metrics})
