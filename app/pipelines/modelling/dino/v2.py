"""Frozen DINOv2 patch-token nearest-neighbour baseline for MVTec AD."""

from pathlib import Path
from typing import Any

from app.pipelines.modelling.dino.engine import run_dino_all_categories, run_dino_category
from app.pipelines.modelling.dino.types import AllCategoriesResult, BaselineResult, DINOVariant, MaskingMode
from app.pipelines.modelling.patchcore import PATCHCORE_MODEL_SEED
from app.pipelines.preprocessing.base import PreprocessingPipeline

DINO_V2_ENCODER = "vit_small_patch14_dinov2"
DINO_V2_INPUT_SIZE = 252
DINO_V2_PATCH_SIZE = 14
DINO_V2_BATCH_SIZE = 4
ENHANCED_DINO_LAYERS = (8, 10, 11)


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
    variant: DINOVariant = "baseline",
    feature_layers: tuple[int, ...] = ENHANCED_DINO_LAYERS,
    position_radius: int = 1,
    spatial_weight: float = 0.05,
    density_neighbors: int = 5,
) -> BaselineResult | AllCategoriesResult:
    """Run one MVTec category or all canonical categories with frozen DINOv2.

    ``category="all"`` orchestrates sequential evaluation across all 15 categories,
    ensuring each receives its fixed split, normal feature bank, and threshold calibration.

    Args:
        data_root: Root directory of MVTec AD dataset.
        category: Category name or 'all' for full benchmark evaluation.
        pipeline: Preprocessing pipeline or list of step configs.
        fpr_limit: Fair-eval AUPIMO upper FPR bound (fixed at 1e-4).
        encoder_name: Pretrained DINOv2 vision transformer encoder identifier.
        num_neighbors: Number of normal patch neighbours to query.
        masking: Foreground PCA masking policy ('off', 'on', or 'published').
        run_heatmap: Whether to render anomalous test-image heatmaps.
        preprocessing_steps: Legacy parameter preserved for backward compatibility.
        registry_base: Output root directory for evaluation artifacts.
        model_seed: Deterministic model and data loader seed.
        variant: Feature extractor variant ('baseline' or 'enhanced').
        feature_layers: Transformer block indices for enhanced scoring.
        position_radius: Patch search radius for enhanced spatial matching.
        spatial_weight: Penalty weight for spatial distance in enhanced scoring.
        density_neighbors: Number of neighbours for local density estimation.

    Returns:
        BaselineResult for a single category, or AllCategoriesResult for 'all'.
    """
    if category != "all":
        return run_dino_category(
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
            variant=variant,
            feature_layers=feature_layers,
            position_radius=position_radius,
            spatial_weight=spatial_weight,
            density_neighbors=density_neighbors,
            model_generation="dinov2",
            model_name="DINOv2",
            input_size=DINO_V2_INPUT_SIZE,
            patch_size=DINO_V2_PATCH_SIZE,
            batch_size=DINO_V2_BATCH_SIZE,
        )

    return run_dino_all_categories(
        data_root=data_root,
        pipeline=pipeline,
        fpr_limit=fpr_limit,
        encoder_name=encoder_name,
        num_neighbors=num_neighbors,
        masking=masking,
        run_heatmap=run_heatmap,
        registry_base=registry_base,
        model_seed=model_seed,
        variant=variant,
        feature_layers=feature_layers,
        position_radius=position_radius,
        spatial_weight=spatial_weight,
        density_neighbors=density_neighbors,
        model_generation="dinov2",
        model_name="DINOv2",
        input_size=DINO_V2_INPUT_SIZE,
        patch_size=DINO_V2_PATCH_SIZE,
        batch_size=DINO_V2_BATCH_SIZE,
        save_summary_files=False,
    )


__all__ = [
    "DINO_V2_BATCH_SIZE",
    "DINO_V2_ENCODER",
    "DINO_V2_INPUT_SIZE",
    "DINO_V2_PATCH_SIZE",
    "ENHANCED_DINO_LAYERS",
    "run_dinov2_baseline",
]
