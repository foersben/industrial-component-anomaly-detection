"""End-to-end PatchCore anomaly detection pipeline orchestrator."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from app.core.logger import logger
from app.domain.data import (
    FAIR_EVALUATION_PROTOCOL,
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    AUPIMO_NUM_THRESHOLDS,
    CANONICAL_MAP_SIZE,
    PIXEL_METRICS_VERSION,
    fair_metric_evidence,
)
from app.pipelines.modelling.patchcore.dataset import (
    _configure_patchcore_partitions,
    _seed_patchcore_run,
)
from app.pipelines.modelling.patchcore.evaluation import (
    _load_heatmap_overlays,
    _save_heatmap_overlays,
    _to_float,
    extract_and_save_pr_metrics,
    format_results,
)
from app.pipelines.modelling.patchcore.registry import (
    _normalize_preprocessing_steps,
    find_cached_patchcore_model,
)
from app.pipelines.modelling.patchcore.types import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    BaselineResult,
    EvaluationArtifacts,
)
from app.pipelines.modelling.patchcore.visualization import (
    _print_patchcore_results_table,
    _RawScoreImageVisualizer,
)
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.base import PreprocessingPipeline
from app.pipelines.preprocessing.factory import build_pipeline_from_configs


def _load_cached_patchcore_result(
    cached_dir: Path,
    meta: dict[str, Any],
    category: str,
    backbone: str,
    feature_layers: tuple[str, ...],
    coreset_sampling_ratio: float,
    num_neighbors: int,
    fpr_limit: float,
    raw_prep_list: list[dict[str, Any]],
    load_heatmaps: bool = False,
) -> BaselineResult:
    """Load, validate, and construct standard BaselineResult from a verified cache directory.

    Args:
        cached_dir: Path to directory containing cached model and metric arrays.
        meta: Loaded metadata dictionary from metadata.json.
        category: Component category name.
        backbone: Feature extractor backbone identifier.
        feature_layers: Extracted feature layer names.
        coreset_sampling_ratio: Coreset subsampling ratio.
        num_neighbors: Nearest neighbor count.
        fpr_limit: Bound for AUPIMO integration.
        raw_prep_list: Normalized preprocessing configurations.
        load_heatmaps: Whether to deserialize all cached heatmap arrays eagerly.

    Returns:
        Structured BaselineResult matching fair-eval-v1 protocol.

    Raises:
        FileNotFoundError: If metric npz files are absent.
        ValueError: If genuine AUPIMO or required metrics are missing.
    """
    logger.info("Found cached Patchcore model in %s. Loading evaluation metrics...", cached_dir)
    pixel_file = cached_dir / "pixel_metrics.npz"
    image_file = cached_dir / "image_metrics.npz"

    if not pixel_file.exists() or not image_file.exists():
        raise FileNotFoundError("Fair PatchCore cache is missing required metric artifacts")
    with np.load(pixel_file, allow_pickle=False) as pixel_data:
        if "aupimo" not in pixel_data:
            raise ValueError("Fair PatchCore pixel metrics are missing genuine AUPIMO")
        aupimo = float(pixel_data["aupimo"])

    required_metric_keys = {
        "image_auroc",
        "pixel_auroc",
        "manual_image_f1",
        "manual_pixel_f1",
        "manual_image_prec",
        "manual_image_rec",
        "img_threshold",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    }
    if missing_keys := required_metric_keys.difference(meta):
        raise ValueError(f"Fair PatchCore metadata is missing metrics: {sorted(missing_keys)}")

    heatmap_overlays_path = meta.get("heatmap_overlays_path")
    if heatmap_overlays_path and load_heatmaps:
        try:
            heatmap_overlays = _load_heatmap_overlays(cached_dir / Path(heatmap_overlays_path).name)
        except (OSError, ValueError) as e:
            logger.warning("Could not load cached PatchCore heatmaps: %s", e)
            heatmap_overlays = {}
    else:
        heatmap_overlays = meta.get("heatmap_overlays", {})

    hyperparams = meta.get(
        "hyperparameters",
        {
            "backbone": meta.get("backbone", backbone),
            "feature_layers": meta.get("feature_layers", feature_layers),
            "coreset_sampling_ratio": meta.get("coreset_sampling_ratio", coreset_sampling_ratio),
            "num_neighbors": meta.get("num_neighbors", num_neighbors),
            "fpr_limit": meta.get("fpr_limit", fpr_limit),
            "train_batch_size": 16,
            "eval_batch_size": 16,
        },
    )

    return {
        "category": category,
        "image_level": {
            "auroc": float(meta["image_auroc"]),
            "f1_score": float(meta["manual_image_f1"]),
            "precision": float(meta["manual_image_prec"]),
            "recall": float(meta["manual_image_rec"]),
            "threshold": float(meta["img_threshold"]),
            "true_positives": int(meta["true_positives"]),
            "false_positives": int(meta["false_positives"]),
            "false_negatives": int(meta["false_negatives"]),
            "true_negatives": int(meta["true_negatives"]),
            "metrics_path": str(image_file),
        },
        "pixel_level": {
            "auroc": float(meta["pixel_auroc"]),
            "f1_score": float(meta["manual_pixel_f1"]),
            **({"threshold": float(meta["pixel_threshold"])} if "pixel_threshold" in meta else {}),
            "aupimo_score": aupimo,
            "fpr_lower_bound": 1e-5,
            "fpr_upper_bound": fpr_limit,
            "aupimo": aupimo,
            "anomaly_map_min": float(meta.get("anomaly_map_min", 0.0)),
            "anomaly_map_max": float(meta.get("anomaly_map_max", 0.0)),
            "anomaly_map_range": float(meta.get("anomaly_map_range", 0.0)),
            "metrics_path": str(pixel_file),
        },
        "raw_results": {k: _to_float(v) for k, v in meta.get("raw_results", {}).items()},
        "heatmap_overlays": heatmap_overlays,
        "anomalous_indices": meta.get("anomalous_indices", []),
        "preprocessing_steps": meta.get("preprocessing_steps", raw_prep_list),
        "hyperparameters": hyperparams,
        "dataset_split": meta.get("dataset_split", {}),
        "model_hash": meta.get("hash", cached_dir.name),
        "metadata": meta,
    }


def run_patchcore_pipeline(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    backbone: str = "resnet18",
    feature_layers: tuple[str, ...] = ("layer2", "layer3"),
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    run_heatmap: bool = False,
    force_retrain: bool = False,
    model_hash: str | None = None,
    registry_base: Path | str = "data/models/patchcore",
    model_seed: int = PATCHCORE_MODEL_SEED,
) -> BaselineResult:
    """Run the PatchCore anomaly detection pipeline on the MVTec AD dataset.

    Args:
        data_root: Root directory of MVTec AD.
        category: Category to evaluate.
        pipeline: Optional list of preprocessing step configurations or pipeline.
        fpr_limit: Maximum allowable False Positive Rate.
        backbone: Feature extractor backbone (e.g. 'resnet18', 'wide_resnet50_2').
        feature_layers: Layers to extract features from.
        coreset_sampling_ratio: Ratio for coreset subsampling.
        num_neighbors: Number of nearest neighbors for scoring.
        run_heatmap: Whether to compute heatmap overlays.
        force_retrain: If True, ignores cache and forces a full re-fit.
        model_hash: Optional target model hash to search for.
        registry_base: Base directory path for Patchcore model registry.
        model_seed: Seed controlling PatchCore coreset sampling and data-loader workers.

    Returns:
        Structured evaluation metrics conforming to fair-eval-v1.
    """
    if not np.isclose(fpr_limit, AUPIMO_FPR_BOUNDS[1]):
        raise ValueError(f"fair-eval-v1 requires fpr_limit={AUPIMO_FPR_BOUNDS[1]}")

    if isinstance(pipeline, PreprocessingPipeline):
        proc_pipeline = pipeline
        raw_prep_list: list[dict[str, Any]] = []
    else:
        proc_pipeline = build_pipeline_from_configs(pipeline)
        raw_prep_list = _normalize_preprocessing_steps(pipeline)

    manifest = build_mvtec_manifest(data_root)
    fair_split = build_fair_evaluation_split(manifest, category)
    split_evidence = fair_split.evidence()
    cache_evidence = {
        **split_evidence,
        **fair_metric_evidence(),
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    }
    norm_prep_str = json.dumps(raw_prep_list, sort_keys=True)
    layer_str = "_".join(feature_layers)
    hp_string = (
        f"{category}_{backbone}_{layer_str}_{coreset_sampling_ratio}_{num_neighbors}_{fpr_limit}_{norm_prep_str}_"
        f"{json.dumps(cache_evidence, sort_keys=True)}_{CANONICAL_MAP_SIZE}_{AUPIMO_FPR_BOUNDS}_"
        f"{AUPIMO_NUM_THRESHOLDS}_{PIXEL_METRICS_VERSION}"
    )
    computed_hash = hashlib.sha256(hp_string.encode()).hexdigest()[:12]

    cached = find_cached_patchcore_model(
        category=category,
        backbone=backbone,
        feature_layers=feature_layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        fpr_limit=fpr_limit,
        pipeline=raw_prep_list,
        target_hash=model_hash,
        registry_base=registry_base,
        expected_split_evidence=cache_evidence,
    )
    if model_hash and cached is None:
        raise FileNotFoundError(
            f"Cached PatchCore run {model_hash} is missing or does not match the current dataset protocol."
        )

    if cached is not None and not force_retrain:
        cached_dir, meta = cached
        return _load_cached_patchcore_result(
            cached_dir=cached_dir,
            meta=meta,
            category=category,
            backbone=backbone,
            feature_layers=feature_layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            fpr_limit=fpr_limit,
            raw_prep_list=raw_prep_list,
        )

    # A rejected or explicitly bypassed cache must never be overwritten.
    effective_hash = computed_hash
    logger.info("Configured preprocessing pipeline with %d steps.", len(proc_pipeline))
    base_dir = Path(registry_base) / effective_hash
    if base_dir.exists():
        archived_hash = hashlib.sha256(f"{computed_hash}:{time.time_ns()}".encode()).hexdigest()[:12]
        archived_dir = Path(registry_base) / archived_hash
        base_dir.rename(archived_dir)
        archived_metadata_path = archived_dir / "metadata.json"
        if archived_metadata_path.is_file():
            archived_metadata = json.loads(archived_metadata_path.read_text(encoding="utf-8"))
            archived_metadata["hash"] = archived_hash
            archived_metadata["configuration_hash"] = computed_hash
            archived_metadata_path.write_text(json.dumps(archived_metadata, indent=4), encoding="utf-8")
    base_dir.mkdir(parents=True)
    transform_adapter = PreprocessingTransformAdapter(proc_pipeline)

    # 1. Initialize dataset, model, and engine
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        val_split_mode="none",
    )
    _configure_patchcore_partitions(
        datamodule,
        fair_split,
        transform_adapter if len(proc_pipeline) > 0 else None,
    )

    # PatchCore's coreset is sampled stochastically. Seed immediately before construction.
    _seed_patchcore_run(model_seed)
    model = Patchcore(
        backbone=backbone,
        layers=feature_layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        post_processor=False,
        evaluator=False,
        visualizer=_RawScoreImageVisualizer(output_dir=base_dir / "four_panel_images"),
    )
    engine = Engine(accelerator="gpu", devices=1, deterministic=True)

    # 2. Fit
    train_dataloader = datamodule.train_dataloader()
    validation_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    logger.info("Fitting Patchcore model on %s category (Hash: %s)...", category, effective_hash)
    engine.fit(model, train_dataloaders=train_dataloader)

    # 3. Extract PR metrics and build summary
    artifacts = EvaluationArtifacts.from_tuple(
        extract_and_save_pr_metrics(
            engine,
            model,
            validation_dataloader,
            test_dataloader,
            base_dir,
            run_heatmap,
        )
    )

    # Extract dataset split counts
    split_info = {
        **cache_evidence,
        "test_normal": (fair_split.test["is_anomaly"] == 0).sum(),
        "test_anomalous": fair_split.test["is_anomaly"].astype(bool).sum(),
    }

    hyperparams = {
        "backbone": backbone,
        "feature_layers": feature_layers,
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "fpr_limit": fpr_limit,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
    }

    raw_results_dict = {
        "image_AUROC": artifacts.image_metrics.auroc,
        "image_F1Score": artifacts.image_metrics.f1_score,
        "image_Precision": artifacts.image_metrics.precision,
        "image_Recall": artifacts.image_metrics.recall,
        "pixel_AUROC": artifacts.pixel_metrics.auroc,
        "pixel_F1Score": artifacts.pixel_metrics.f1_score,
        "pixel_AUPIMO": artifacts.pixel_metrics.aupimo,
    }
    test_results: list[Mapping[str, float]] = [raw_results_dict]
    _print_patchcore_results_table(raw_results_dict)

    logger.info(
        "PatchCore evaluation summary | image: AUROC=%.6f F1=%.6f precision=%.6f recall=%.6f threshold=%.6f",
        artifacts.image_metrics.auroc,
        artifacts.image_metrics.f1_score,
        artifacts.image_metrics.precision,
        artifacts.image_metrics.recall,
        artifacts.thresholds.image,
    )
    logger.info(
        "PatchCore evaluation summary | confusion: TP=%d FP=%d FN=%d TN=%d",
        artifacts.image_metrics.confusion.true_positives,
        artifacts.image_metrics.confusion.false_positives,
        artifacts.image_metrics.confusion.false_negatives,
        artifacts.image_metrics.confusion.true_negatives,
    )
    logger.info(
        "PatchCore evaluation summary | pixel: AUROC=%.6f F1=%.6f AUPIMO=%.6f",
        artifacts.pixel_metrics.auroc,
        artifacts.pixel_metrics.f1_score,
        artifacts.pixel_metrics.aupimo,
    )

    heatmap_archive = _save_heatmap_overlays(artifacts.heatmap_overlays, base_dir / "heatmap_overlays.npz")
    four_panel_dir = base_dir / "four_panel_images"
    if heatmap_archive is not None:
        logger.info("Saved compressed PatchCore heatmaps to %s", heatmap_archive)

    metadata = {
        "hash": effective_hash,
        "configuration_hash": computed_hash,
        "model_type": "patchcore",
        "category": category,
        "backbone": backbone,
        "feature_layers": list(feature_layers),
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "fpr_limit": fpr_limit,
        "preprocessing_steps": raw_prep_list,
        "hyperparameters": hyperparams,
        "dataset_split": split_info,
        "protocol": FAIR_EVALUATION_PROTOCOL,
        "threshold_source": "normal_validation",
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "pixel_metrics_version": PIXEL_METRICS_VERSION,
        "image_auroc": raw_results_dict["image_AUROC"],
        "pixel_auroc": artifacts.pixel_metrics.auroc,
        "manual_image_f1": artifacts.image_metrics.f1_score,
        "manual_pixel_f1": artifacts.pixel_metrics.f1_score,
        "manual_image_prec": artifacts.image_metrics.precision,
        "manual_image_rec": artifacts.image_metrics.recall,
        "true_positives": artifacts.image_metrics.confusion.true_positives,
        "false_positives": artifacts.image_metrics.confusion.false_positives,
        "false_negatives": artifacts.image_metrics.confusion.false_negatives,
        "true_negatives": artifacts.image_metrics.confusion.true_negatives,
        "img_threshold": artifacts.thresholds.image,
        "pixel_threshold": artifacts.thresholds.pixel,
        "pixel_aupimo": artifacts.pixel_metrics.aupimo,
        "aupimo_fpr_bounds": [1e-5, fpr_limit],
        "aupimo_num_thresholds": AUPIMO_NUM_THRESHOLDS,
        "canonical_height": CANONICAL_MAP_SIZE[0],
        "canonical_width": CANONICAL_MAP_SIZE[1],
        "anomaly_map_min": artifacts.pixel_metrics.anomaly_map_min,
        "anomaly_map_max": artifacts.pixel_metrics.anomaly_map_max,
        "anomaly_map_range": artifacts.pixel_metrics.anomaly_map_range,
        "heatmap_overlays_path": heatmap_archive.name if heatmap_archive is not None else None,
        "four_panel_images_path": four_panel_dir.name if four_panel_dir.is_dir() else None,
        "anomalous_indices": artifacts.anomalous_indices,
        "raw_results": raw_results_dict,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        with open(base_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info("Saved Patchcore model metadata to %s", base_dir / "metadata.json")
    except Exception as e:
        logger.warning("Could not save Patchcore metadata.json: %s", e)

    return format_results(
        test_results=test_results,
        category=category,
        base_dir=base_dir,
        artifacts=artifacts,
        fpr_limit=fpr_limit,
        preprocessing_steps=raw_prep_list,
        hyperparameters=hyperparams,
        dataset_split=split_info,
        model_hash=effective_hash,
        metadata=metadata,
    )
