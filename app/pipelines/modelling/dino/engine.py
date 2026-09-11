"""Shared execution and evaluation engine for DINO foundation model baselines."""

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO

from app.core.logger import logger
from app.domain.categories import ANOMALY_DINO_MASKED_CATEGORIES, MVTEC_CATEGORIES
from app.domain.data import build_fair_evaluation_split, build_mvtec_manifest
from app.pipelines.evaluation.metrics import (
    AUPIMO_FPR_BOUNDS,
    fair_metric_evidence,
)
from app.pipelines.modelling.anomalib.dataset import (
    configure_anomalib_partitions,
    seed_anomalib_run,
)
from app.pipelines.modelling.anomalib.visualization import RawScoreImageVisualizer
from app.pipelines.modelling.dino.artifacts import (
    load_completed_category_result,
    persist_dino_artifacts_and_format,
    save_all_category_summary,
)
from app.pipelines.modelling.dino.enhanced import EnhancedAnomalyDINOModel
from app.pipelines.modelling.dino.types import AllCategoriesResult, BaselineResult, DINOVariant, MaskingMode
from app.pipelines.modelling.patchcore import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    EvaluationArtifacts,
    extract_and_save_pr_metrics,
)
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.base import PreprocessingPipeline
from app.pipelines.preprocessing.factory import (
    build_pipeline_from_configs,
    normalize_preprocessing_steps,
)

# Backwards compatibility aliases
_configure_patchcore_partitions = configure_anomalib_partitions
_seed_patchcore_run = seed_anomalib_run

METRIC_PATHS: dict[str, tuple[str, str]] = {
    "image_f1": ("image_level", "f1_score"),
    "image_recall": ("image_level", "recall"),
    "image_precision": ("image_level", "precision"),
    "image_auroc": ("image_level", "auroc"),
    "image_average_precision": ("image_level", "average_precision"),
    "pixel_f1": ("pixel_level", "f1_score"),
    "pixel_auroc": ("pixel_level", "auroc"),
    "pixel_aupimo": ("pixel_level", "aupimo"),
}


def resolve_masking(masking: MaskingMode, category: str) -> bool:
    """Resolve an explicit or published full-shot AnomalyDINO masking policy.

    Args:
        masking: Active masking policy mode ('off', 'on', or 'published').
        category: MVTec object or texture category name.

    Returns:
        True if foreground PCA masking should be applied, False otherwise.

    Raises:
        ValueError: If masking mode is invalid.
    """
    if masking == "off":
        return False
    if masking == "on":
        return True
    if masking == "published":
        return category in ANOMALY_DINO_MASKED_CATEGORIES
    raise ValueError("masking must be one of: off, on, published")


def release_accelerator_memory() -> None:
    """Collect cyclic trainer state and release unused CUDA allocations."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_dino_identity_and_hash(
    category: str,
    encoder_name: str,
    num_neighbors: int,
    masking: MaskingMode,
    use_masking: bool,
    raw_prep_list: list[dict[str, Any]],
    cache_evidence: dict[str, Any],
    model_generation: Literal["dinov2", "dinov3"],
    variant: DINOVariant,
    feature_layers: tuple[int, ...],
    position_radius: int,
    spatial_weight: float,
    density_neighbors: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Build the serializable identity dictionary and unique hash for a DINO experiment.

    Args:
        category: MVTec category being evaluated.
        encoder_name: Name of the pretrained feature encoder.
        num_neighbors: Nearest neighbors count.
        masking: Active masking policy mode.
        use_masking: Evaluated boolean decision for masking.
        raw_prep_list: Normalized preprocessing configurations.
        cache_evidence: Protocol split and metric version metadata.
        model_generation: Architectural generation ('dinov2' or 'dinov3').
        variant: Feature extractor variant ('baseline' or 'enhanced').
        feature_layers: Block indices for enhanced scoring.
        position_radius: Grid search radius for spatial matching.
        spatial_weight: Weight factor for spatial distance penalty.
        density_neighbors: Neighbor count for density estimation.

    Returns:
        A tuple of (identity_dictionary, 12_char_hex_hash, enhanced_scorer_parameters).

    Raises:
        ValueError: If variant is invalid.
    """
    identity: dict[str, Any] = {
        "category": category,
        "encoder_name": encoder_name,
        "num_neighbors": num_neighbors,
        "masking_mode": masking,
        "masking": use_masking,
        "coreset_subsampling": False,
        "preprocessing_steps": raw_prep_list,
        "evaluation": cache_evidence,
    }
    if model_generation != "dinov2":
        identity["model_generation"] = model_generation

    enhanced_scorer: dict[str, Any] = {}
    if variant == "enhanced":
        enhanced_scorer = {
            "feature_layers": list(feature_layers),
            "feature_normalization": "l2_after_pca_mask",
            "position_radius": position_radius,
            "spatial_weight": spatial_weight,
            "position_fallback": "global_with_spatial_penalty",
            "density_neighbors": density_neighbors,
            "density_normalization": "global_median_quarter_power_clipped_0.5_2.0",
            "image_aggregation": "patchcore_neighborhood_reweighting",
        }
        identity["enhanced_scorer"] = enhanced_scorer
    elif variant != "baseline":
        raise ValueError("variant must be one of: baseline, enhanced")

    model_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return identity, model_hash, enhanced_scorer


def instantiate_dino_model(
    num_neighbors: int,
    encoder_name: str,
    use_masking: bool,
    model_generation: Literal["dinov2", "dinov3"],
    variant: DINOVariant,
    input_size: int,
    feature_layers: tuple[int, ...],
    position_radius: int,
    spatial_weight: float,
    density_neighbors: int,
) -> AnomalyDINO:
    """Instantiate and configure the AnomalyDINO model with frozen feature extractor.

    Args:
        num_neighbors: Nearest neighbors count for memory bank lookup.
        encoder_name: Identifier for the backbone encoder.
        use_masking: Whether foreground PCA patch masking is active.
        model_generation: Architectural generation ('dinov2' or 'dinov3').
        variant: Model variant ('baseline' or 'enhanced').
        input_size: Canonical square input dimension.
        feature_layers: Block indices for enhanced multi-layer feature extraction.
        position_radius: Position search radius for enhanced spatial matching.
        spatial_weight: Weight penalty for off-center matches.
        density_neighbors: Neighbor count for density estimation.

    Returns:
        Configured AnomalyDINO instance with gradient updates disabled on encoder.
    """
    model_kwargs: dict[str, Any] = {
        "num_neighbours": num_neighbors,
        "encoder_name": encoder_name,
        "masking": use_masking,
        "coreset_subsampling": False,
        "post_processor": False,
        "evaluator": False,
        "visualizer": RawScoreImageVisualizer(),
    }
    if model_generation == "dinov3":
        model_kwargs["pre_processor"] = AnomalyDINO.configure_pre_processor((input_size, input_size))

    model = AnomalyDINO(**model_kwargs)
    if variant == "enhanced":
        model.model = EnhancedAnomalyDINOModel(
            num_neighbours=num_neighbors,
            encoder_name=encoder_name,
            masking=use_masking,
            feature_layers=feature_layers,
            position_radius=position_radius,
            spatial_weight=spatial_weight,
            density_neighbours=density_neighbors,
        )
    model.model.feature_encoder.requires_grad_(False)
    return model


def run_dino_category(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    encoder_name: str = "vit_small_patch14_dinov2",
    num_neighbors: int = 1,
    masking: MaskingMode = "published",
    run_heatmap: bool = False,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    registry_base: Path | str = "data/models/dinov2",
    model_seed: int = PATCHCORE_MODEL_SEED,
    reuse_complete: bool = False,
    variant: DINOVariant = "baseline",
    feature_layers: tuple[int, ...] = (8, 10, 11),
    position_radius: int = 1,
    spatial_weight: float = 0.05,
    density_neighbors: int = 5,
    model_generation: Literal["dinov2", "dinov3"] = "dinov2",
    model_name: str = "DINOv2",
    input_size: int = 252,
    patch_size: int = 14,
    batch_size: int = 4,
    manifest: pd.DataFrame | None = None,
) -> BaselineResult:
    """Evaluate frozen DINO patch tokens with a normal-only nearest-neighbour bank.

    Args:
        data_root: Root directory of MVTec AD.
        category: MVTec category to evaluate.
        pipeline: Optional preprocessing pipeline or configuration list.
        fpr_limit: Upper AUPIMO false-positive-rate bound fixed by the fair protocol.
        encoder_name: Pretrained DINO encoder exposed by Anomalib/timm.
        num_neighbors: Number of normal patch neighbours averaged per patch.
        masking: PCA foreground-mask policy: disabled, enabled, or published.
        run_heatmap: Whether to render overlays for anomalous test images.
        preprocessing_steps: Legacy parameter preserved for backward compatibility.
        registry_base: Directory in which evaluation artifacts are written.
        model_seed: Shared deterministic model and data-loader seed.
        reuse_complete: Reuse complete artifacts for this exact configuration.
        variant: Stock final-block scorer or enhanced multi-layer scorer.
        feature_layers: Transformer block indices used by the enhanced scorer.
        position_radius: Patch-grid search radius used by the enhanced scorer.
        spatial_weight: Spatial-distance penalty used by the enhanced scorer.
        density_neighbors: Normal neighbours used to estimate local density.
        model_generation: Encoder family ('dinov2' or 'dinov3').
        model_name: Human-readable model label used in logs.
        input_size: Square model input size recorded in the run metadata.
        patch_size: Encoder patch size recorded in the run metadata.
        batch_size: Batch size for training datamodule.
        manifest: Optional prebuilt dataset manifest for all-category orchestration.

    Returns:
        Results using the standardized BaselineResult schema.

    Raises:
        ValueError: If validation bounds or encoder specifications are invalid.
        RuntimeError: If memory bank fitting produces zero normal patches.
    """
    if not np.isclose(fpr_limit, AUPIMO_FPR_BOUNDS[1]):
        raise ValueError(f"fair-eval-v1 requires fpr_limit={AUPIMO_FPR_BOUNDS[1]}")
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be at least 1")
    if model_generation not in {"dinov2", "dinov3"}:
        raise ValueError("model_generation must be one of: dinov2, dinov3")
    if model_generation not in encoder_name:
        raise ValueError(f"encoder_name must identify a pretrained {model_name} encoder")
    use_masking = resolve_masking(masking, category)

    steps_config = pipeline if pipeline is not None else preprocessing_steps
    if isinstance(steps_config, PreprocessingPipeline):
        proc_pipeline = steps_config
        raw_prep_list: list[dict[str, Any]] = []
    else:
        proc_pipeline = build_pipeline_from_configs(steps_config)
        raw_prep_list = normalize_preprocessing_steps(steps_config)

    active_manifest = manifest if manifest is not None else build_mvtec_manifest(data_root)
    fair_split = build_fair_evaluation_split(active_manifest, category)
    cache_evidence = {
        **fair_split.evidence(),
        **fair_metric_evidence(),
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "image_threshold_quantile": PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
        "pixel_threshold_quantile": PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    }

    _identity, configuration_hash, enhanced_scorer = build_dino_identity_and_hash(
        category=category,
        encoder_name=encoder_name,
        num_neighbors=num_neighbors,
        masking=masking,
        use_masking=use_masking,
        raw_prep_list=raw_prep_list,
        cache_evidence=cache_evidence,
        model_generation=model_generation,
        variant=variant,
        feature_layers=feature_layers,
        position_radius=position_radius,
        spatial_weight=spatial_weight,
        density_neighbors=density_neighbors,
    )

    active_hash = configuration_hash
    base_dir = Path(registry_base) / active_hash
    if reuse_complete:
        completed_result = load_completed_category_result(
            base_dir, category, active_hash, run_heatmap, load_heatmaps=False
        )
        if completed_result is not None:
            logger.info("Reusing complete %s result for %s (Hash: %s)", model_name, category, active_hash)
            return completed_result

    if base_dir.exists():
        archived_hash = hashlib.sha256(f"{configuration_hash}:{time.time_ns()}".encode()).hexdigest()[:12]
        archived_dir = Path(registry_base) / archived_hash
        base_dir.rename(archived_dir)
        archived_metadata_path = archived_dir / "metadata.json"
        if archived_metadata_path.is_file():
            archived_metadata = json.loads(archived_metadata_path.read_text(encoding="utf-8"))
            archived_metadata["hash"] = archived_hash
            archived_metadata["configuration_hash"] = configuration_hash
            archived_metadata_path.write_text(json.dumps(archived_metadata, indent=4), encoding="utf-8")
    base_dir.mkdir(parents=True)

    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        val_split_mode="none",
    )
    transform_adapter = PreprocessingTransformAdapter(proc_pipeline)
    configure_anomalib_partitions(
        datamodule,
        fair_split,
        transform_adapter if len(proc_pipeline) > 0 else None,
    )

    seed_anomalib_run(model_seed)
    model = instantiate_dino_model(
        num_neighbors=num_neighbors,
        encoder_name=encoder_name,
        use_masking=use_masking,
        model_generation=model_generation,
        variant=variant,
        input_size=input_size,
        feature_layers=feature_layers,
        position_radius=position_radius,
        spatial_weight=spatial_weight,
        density_neighbors=density_neighbors,
    )
    if isinstance(model.visualizer, RawScoreImageVisualizer):
        model.visualizer.output_dir = base_dir / "four_panel_images"

    engine = Engine(accelerator="auto", devices=1, deterministic=True)
    train_dataloader = datamodule.train_dataloader()
    validation_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    logger.info("Building %s normal patch bank for %s (Hash: %s)...", model_name, category, active_hash)
    engine.fit(model, train_dataloaders=train_dataloader)
    memory_bank = getattr(model.model, "memory_bank", None)
    if memory_bank is not None and hasattr(memory_bank, "numel") and memory_bank.numel() == 0:
        raise RuntimeError(
            f"{model_name} fitting produced an empty memory bank for category '{category}'. "
            "Check that the fitting loader contains normal images and that foreground masking retains patches."
        )

    artifacts = EvaluationArtifacts.from_tuple(
        extract_and_save_pr_metrics(
            engine,
            model,
            validation_dataloader,
            test_dataloader,
            base_dir,
            run_heatmap,
            model_name=model_name,
        )
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
        "input_size": input_size,
        "patch_size": patch_size,
        "num_neighbors": num_neighbors,
        "masking_mode": masking,
        "masking": use_masking,
        "coreset_subsampling": False,
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "model_seed": model_seed,
        "score_space": PATCHCORE_SCORE_SPACE,
        "variant": variant,
    }
    if variant == "enhanced":
        hyperparameters.update(enhanced_scorer)

    return persist_dino_artifacts_and_format(
        category=category,
        base_dir=base_dir,
        model_hash=active_hash,
        configuration_hash=configuration_hash,
        model_generation=model_generation,
        variant=variant,
        encoder_name=encoder_name,
        num_neighbors=num_neighbors,
        masking=masking,
        use_masking=use_masking,
        raw_prep_list=raw_prep_list,
        hyperparameters=hyperparameters,
        split_info=split_info,
        artifacts=artifacts,
        image_average_precision=image_average_precision,
        fpr_limit=fpr_limit,
        model_seed=model_seed,
    )


def compute_macro_average(category_results: dict[str, BaselineResult]) -> dict[str, float]:
    """Compute unweighted macro averages across all evaluated categories.

    Args:
        category_results: Mapping of category names to BaselineResults.

    Returns:
        Dictionary of mean values for image- and pixel-level metrics.
    """
    macro_average: dict[str, float] = {}
    for metric, (level, key) in METRIC_PATHS.items():
        values = [float(result[level][key]) for result in category_results.values()]  # type: ignore[literal-required]
        macro_average[metric] = float(np.mean(values))
    return macro_average


def run_dino_all_categories(
    data_root: Path | str,
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None,
    fpr_limit: float,
    encoder_name: str,
    num_neighbors: int,
    masking: MaskingMode,
    run_heatmap: bool,
    registry_base: Path | str,
    model_seed: int,
    variant: DINOVariant,
    feature_layers: tuple[int, ...],
    position_radius: int,
    spatial_weight: float,
    density_neighbors: int,
    model_generation: Literal["dinov2", "dinov3"],
    model_name: str,
    input_size: int,
    patch_size: int,
    batch_size: int,
    reuse_complete: bool = False,
    save_summary_files: bool = False,
) -> AllCategoriesResult:
    """Orchestrate sequential evaluation across all canonical MVTec categories.

    Args:
        data_root: Root directory of MVTec AD.
        pipeline: Preprocessing pipeline or configuration list.
        fpr_limit: Fixed fair-eval AUPIMO FPR limit.
        encoder_name: Backbone encoder identifier.
        num_neighbors: Nearest neighbor count.
        masking: Foreground masking policy mode.
        run_heatmap: Whether to render test-image overlays.
        registry_base: Target artifact directory.
        model_seed: Deterministic random seed.
        variant: Scorer variant ('baseline' or 'enhanced').
        feature_layers: Multi-layer indices for enhanced scoring.
        position_radius: Spatial search radius.
        spatial_weight: Spatial distance penalty.
        density_neighbors: Neighbor count for density estimation.
        model_generation: Architectural family ('dinov2' or 'dinov3').
        model_name: Label for logs and summaries.
        input_size: Square model input dimension.
        patch_size: Encoder patch dimension.
        batch_size: DataLoader batch size.
        reuse_complete: Reuse complete artifacts for each category.
        save_summary_files: If True, writes summary.json and category_metrics.csv.

    Returns:
        AllCategoriesResult containing per-category results and macro averages.
    """
    manifest = build_mvtec_manifest(data_root)
    category_results: dict[str, BaselineResult] = {}
    for name in MVTEC_CATEGORIES:
        try:
            result = run_dino_category(
                data_root=data_root,
                category=name,
                pipeline=pipeline,
                fpr_limit=fpr_limit,
                encoder_name=encoder_name,
                num_neighbors=num_neighbors,
                masking=masking,
                run_heatmap=run_heatmap,
                registry_base=registry_base,
                model_seed=model_seed,
                reuse_complete=reuse_complete,
                variant=variant,
                feature_layers=feature_layers,
                position_radius=position_radius,
                spatial_weight=spatial_weight,
                density_neighbors=density_neighbors,
                model_generation=model_generation,
                model_name=model_name,
                input_size=input_size,
                patch_size=patch_size,
                batch_size=batch_size,
                manifest=manifest,
            )
            result["heatmap_overlays"] = {}
            category_results[name] = result
            del result
        finally:
            release_accelerator_memory()

    macro_average = compute_macro_average(category_results)
    if save_summary_files:
        save_all_category_summary(
            registry_base=registry_base,
            encoder_name=encoder_name,
            masking=masking,
            num_neighbors=num_neighbors,
            category_results=category_results,
            macro_average=macro_average,
            model_name=model_name,
        )
    logger.info("%s all-category macro averages: %s", model_name, macro_average)
    return {
        "category": "all",
        "masking_mode": masking,
        "categories": category_results,
        "macro_average": macro_average,
    }


__all__ = [
    "build_dino_identity_and_hash",
    "compute_macro_average",
    "instantiate_dino_model",
    "load_completed_category_result",
    "persist_dino_artifacts_and_format",
    "release_accelerator_memory",
    "resolve_masking",
    "run_dino_all_categories",
    "run_dino_category",
    "save_all_category_summary",
]
