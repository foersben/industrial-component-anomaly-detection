"""Frozen DINOv2 patch-token nearest-neighbour baseline for MVTec AD."""

import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO

from app.core.logger import logger
from app.domain.data import FAIR_EVALUATION_PROTOCOL, build_fair_evaluation_split, build_mvtec_manifest
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    PIXEL_METRICS_VERSION,
    fair_metric_evidence,
)
from app.pipelines.modelling.baseline import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    BaselineResult,
    _configure_patchcore_partitions,
    _normalize_preprocessing_steps,
    _print_patchcore_results_table,
    _RawScoreImageVisualizer,
    _save_heatmap_overlays,
    _seed_patchcore_run,
    extract_and_save_pr_metrics,
    format_results,
)
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.base import PreprocessingPipeline
from app.pipelines.preprocessing.factory import build_pipeline_from_configs

DINO_V2_ENCODER = "vit_small_patch14_dinov2"
DINO_V2_INPUT_SIZE = 252
DINO_V2_BATCH_SIZE = 4
MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
ANOMALY_DINO_MASKED_CATEGORIES = frozenset({"capsule", "hazelnut", "pill", "screw", "toothbrush"})
MaskingMode = Literal["off", "on", "published"]


class AllCategoriesResult(TypedDict):
    """Results and unweighted macro averages for all MVTec categories."""

    category: Literal["all"]
    masking_mode: MaskingMode
    categories: dict[str, BaselineResult]
    macro_average: dict[str, float]


def resolve_masking(masking: MaskingMode, category: str) -> bool:
    """Resolve an explicit or published full-shot AnomalyDINO masking policy."""
    if masking == "off":
        return False
    if masking == "on":
        return True
    if masking == "published":
        return category in ANOMALY_DINO_MASKED_CATEGORIES
    raise ValueError("masking must be one of: off, on, published")


def _load_completed_category_result(
    base_dir: Path,
    category: str,
    model_hash: str,
    run_heatmap: bool,
) -> BaselineResult | None:
    """Load a complete category result without retaining saved heatmap pixels."""
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
            heatmap_overlays={},
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
        logger.warning("Ignoring incomplete DINOv2 artifacts in %s: %s", base_dir, exc)
        return None


def _release_accelerator_memory() -> None:
    """Collect cyclic trainer state and release unused CUDA allocations."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _run_dinov2_category(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    encoder_name: str = DINO_V2_ENCODER,
    num_neighbors: int = 1,
    masking: MaskingMode = "published",
    run_heatmap: bool = False,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    registry_base: Path | str = "data/models/dinov2",
    model_seed: int = PATCHCORE_MODEL_SEED,
    reuse_complete: bool = False,
) -> BaselineResult:
    """Evaluate frozen DINOv2 patch tokens with a normal-only nearest-neighbour bank.

    The fitting partition supplies the memory bank, the normal validation
    partition supplies both deployment thresholds, and the official test set is
    used exactly once for reporting. Background masking and coreset sampling are
    deliberately disabled to keep this baseline minimal and untuned.

    Args:
        data_root: Root directory of MVTec AD.
        category: MVTec category to evaluate.
        pipeline: Optional preprocessing pipeline or configuration list.
        fpr_limit: Upper AUPIMO false-positive-rate bound fixed by the fair protocol.
        encoder_name: Pretrained DINOv2 encoder exposed by Anomalib/timm.
        num_neighbors: Number of normal patch neighbours averaged per patch.
        masking: PCA foreground-mask policy: disabled, enabled, or the published
            full-shot MVTec category policy.
        run_heatmap: Whether to render overlays for anomalous test images.
        preprocessing_steps: Deprecated alias for a preprocessing configuration list.
        registry_base: Directory in which evaluation artifacts are written.
        model_seed: Shared deterministic model and data-loader seed.
        reuse_complete: Reuse complete artifacts for this exact configuration.

    Returns:
        Results using the same schema and metric artifacts as PatchCore.
    """
    if not np.isclose(fpr_limit, AUPIMO_FPR_BOUNDS[1]):
        raise ValueError(f"fair-eval-v1 requires fpr_limit={AUPIMO_FPR_BOUNDS[1]}")
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be at least 1")
    if "dinov2" not in encoder_name:
        raise ValueError("encoder_name must identify a pretrained DINOv2 encoder")
    use_masking = resolve_masking(masking, category)

    steps_config = pipeline if pipeline is not None else preprocessing_steps
    if isinstance(steps_config, PreprocessingPipeline):
        proc_pipeline = steps_config
        raw_prep_list: list[dict[str, Any]] = []
    else:
        proc_pipeline = build_pipeline_from_configs(steps_config)
        raw_prep_list = _normalize_preprocessing_steps(steps_config)

    manifest = build_mvtec_manifest(data_root)
    fair_split = build_fair_evaluation_split(manifest, category)
    cache_evidence = {
        **fair_split.evidence(),
        **fair_metric_evidence(),
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    }
    identity = {
        "category": category,
        "encoder_name": encoder_name,
        "num_neighbors": num_neighbors,
        "masking_mode": masking,
        "masking": use_masking,
        "coreset_subsampling": False,
        "preprocessing_steps": raw_prep_list,
        "evaluation": cache_evidence,
    }
    model_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    base_dir = Path(registry_base) / model_hash
    base_dir.mkdir(parents=True, exist_ok=True)
    if reuse_complete:
        completed_result = _load_completed_category_result(base_dir, category, model_hash, run_heatmap)
        if completed_result is not None:
            logger.info("Reusing complete DINOv2 result for %s (Hash: %s)", category, model_hash)
            return completed_result

    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=DINO_V2_BATCH_SIZE,
        eval_batch_size=DINO_V2_BATCH_SIZE,
        val_split_mode="none",
    )
    transform_adapter = PreprocessingTransformAdapter(proc_pipeline)
    _configure_patchcore_partitions(
        datamodule,
        fair_split,
        transform_adapter if len(proc_pipeline) > 0 else None,
    )

    _seed_patchcore_run(model_seed)
    model = AnomalyDINO(
        num_neighbours=num_neighbors,
        encoder_name=encoder_name,
        masking=use_masking,
        coreset_subsampling=False,
        post_processor=False,
        evaluator=False,
        visualizer=_RawScoreImageVisualizer(),
    )
    # Anomalib already executes this extractor under torch.no_grad(). Freezing
    # the parameter flags as well makes the transfer-learning contract explicit
    # and prevents future trainer changes from accidentally enabling updates.
    model.model.feature_encoder.requires_grad_(False)
    engine = Engine(accelerator="auto", devices=1, deterministic=True)
    train_dataloader = datamodule.train_dataloader()
    validation_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    logger.info("Building DINOv2 normal patch bank for %s (Hash: %s)...", category, model_hash)
    engine.fit(model, train_dataloaders=train_dataloader)
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
    ) = extract_and_save_pr_metrics(
        engine,
        model,
        validation_dataloader,
        test_dataloader,
        base_dir,
        run_heatmap,
        model_name="DINOv2",
    )

    with np.load(base_dir / "image_metrics.npz", allow_pickle=False) as image_metrics:
        precision_curve = np.asarray(image_metrics["precision"], dtype=np.float64)
        recall_curve = np.asarray(image_metrics["recall"], dtype=np.float64)
    image_average_precision = float(-np.sum(np.diff(recall_curve) * precision_curve[:-1]))

    split_info = {
        **cache_evidence,
        "test_normal": int((fair_split.test["is_anomaly"] == 0).sum()),
        "test_anomalous": int(fair_split.test["is_anomaly"].astype(bool).sum()),
    }
    hyperparameters = {
        "encoder_name": encoder_name,
        "input_size": DINO_V2_INPUT_SIZE,
        "patch_size": 14,
        "num_neighbors": num_neighbors,
        "masking_mode": masking,
        "masking": use_masking,
        "coreset_subsampling": False,
        "train_batch_size": DINO_V2_BATCH_SIZE,
        "eval_batch_size": DINO_V2_BATCH_SIZE,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
    }
    raw_results: dict[str, float] = {
        "image_F1Score": image_f1,
        "image_Precision": image_precision,
        "image_Recall": image_recall,
        "image_AUROC": image_auroc,
        "image_AP": image_average_precision,
        "pixel_AUROC": pixel_auroc,
        "pixel_F1Score": pixel_f1,
        "pixel_AUPIMO": pixel_aupimo,
    }
    _print_patchcore_results_table(raw_results)

    heatmap_archive = _save_heatmap_overlays(heatmap_overlays, base_dir / "heatmap_overlays.npz")
    metadata = {
        "hash": model_hash,
        "model_type": "dinov2_knn",
        "category": category,
        "encoder_name": encoder_name,
        "num_neighbors": num_neighbors,
        "masking_mode": masking,
        "masking": use_masking,
        "coreset_subsampling": False,
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
        "image_f1": image_f1,
        "image_precision": image_precision,
        "image_recall": image_recall,
        "image_auroc": image_auroc,
        "image_average_precision": image_average_precision,
        "pixel_f1": pixel_f1,
        "pixel_auroc": pixel_auroc,
        "pixel_aupimo": pixel_aupimo,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "aupimo_fpr_bounds": list(AUPIMO_FPR_BOUNDS),
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "anomaly_map_min": anomaly_map_min,
        "anomaly_map_max": anomaly_map_max,
        "anomaly_map_range": anomaly_map_range,
        "heatmap_overlays_path": heatmap_archive.name if heatmap_archive is not None else None,
        "anomalous_indices": anomalous_indices,
        "raw_results": raw_results,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (base_dir / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    results = format_results(
        test_results=[raw_results],
        category=category,
        base_dir=base_dir,
        manual_image_f1=image_f1,
        manual_pixel_f1=pixel_f1,
        manual_image_prec=image_precision,
        manual_image_rec=image_recall,
        img_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        pixel_auroc=pixel_auroc,
        pixel_aupimo=pixel_aupimo,
        anomaly_map_min=anomaly_map_min,
        anomaly_map_max=anomaly_map_max,
        anomaly_map_range=anomaly_map_range,
        heatmap_overlays=heatmap_overlays,
        anomalous_indices=anomalous_indices,
        fpr_limit=fpr_limit,
        preprocessing_steps=raw_prep_list,
        hyperparameters=hyperparameters,
        dataset_split=split_info,
        model_hash=model_hash,
        metadata=metadata,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
    )
    results["image_level"]["average_precision"] = image_average_precision
    return results


def run_dinov2_baseline(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    encoder_name: str = DINO_V2_ENCODER,
    num_neighbors: int = 1,
    masking: MaskingMode = "published",
    run_heatmap: bool = False,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    registry_base: Path | str = "data/models/dinov2",
    model_seed: int = PATCHCORE_MODEL_SEED,
) -> BaselineResult | AllCategoriesResult:
    """Run one MVTec category or all canonical categories sequentially.

    ``category="all"`` is reporting-only orchestration: every category still
    receives its own fixed split, normal feature bank, validation thresholds,
    official test evaluation, and artifact directory.
    """
    if category != "all":
        return _run_dinov2_category(
            data_root=data_root,
            category=category,
            pipeline=pipeline,
            fpr_limit=fpr_limit,
            encoder_name=encoder_name,
            num_neighbors=num_neighbors,
            masking=masking,
            run_heatmap=run_heatmap,
            preprocessing_steps=preprocessing_steps,
            registry_base=registry_base,
            model_seed=model_seed,
        )

    category_results: dict[str, BaselineResult] = {}
    for name in MVTEC_CATEGORIES:
        try:
            result = _run_dinov2_category(
                data_root=data_root,
                category=name,
                pipeline=pipeline,
                fpr_limit=fpr_limit,
                encoder_name=encoder_name,
                num_neighbors=num_neighbors,
                masking=masking,
                run_heatmap=run_heatmap,
                preprocessing_steps=preprocessing_steps,
                registry_base=registry_base,
                model_seed=model_seed,
                reuse_complete=True,
            )
            # Heatmaps are already persisted as a compressed artifact. Keeping
            # their nested Python lists for every category can consume many GB.
            result["heatmap_overlays"] = {}
            category_results[name] = result
            del result
        finally:
            _release_accelerator_memory()
    metric_paths = {
        "image_f1": ("image_level", "f1_score"),
        "image_recall": ("image_level", "recall"),
        "image_precision": ("image_level", "precision"),
        "image_auroc": ("image_level", "auroc"),
        "image_average_precision": ("image_level", "average_precision"),
        "pixel_f1": ("pixel_level", "f1_score"),
        "pixel_auroc": ("pixel_level", "auroc"),
        "pixel_aupimo": ("pixel_level", "aupimo"),
    }

    def metric_value(result: BaselineResult, level: str, key: str) -> float:
        """Read one numeric metric through the heterogeneous result schema."""
        result_data: Any = result
        return float(result_data[level][key])

    macro_average = {
        metric: float(np.mean([metric_value(result, level, key) for result in category_results.values()]))
        for metric, (level, key) in metric_paths.items()
    }
    logger.info("DINOv2 all-category macro averages: %s", macro_average)
    return {
        "category": "all",
        "masking_mode": masking,
        "categories": category_results,
        "macro_average": macro_average,
    }


if __name__ == "__main__":
    run_dinov2_baseline()
