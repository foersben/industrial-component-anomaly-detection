"""Frozen DINOv3 patch-token nearest-neighbour baseline for MVTec AD."""

from pathlib import Path
from typing import Any

from app.pipelines.modelling.dino.engine import run_dino_all_categories, run_dino_category
from app.pipelines.modelling.dino.types import AllCategoriesResult, BaselineResult, MaskingMode
from app.pipelines.modelling.patchcore import PATCHCORE_MODEL_SEED
from app.pipelines.preprocessing.base import PreprocessingPipeline

DINO_V3_ENCODER = "vit_small_patch16_dinov3.lvd1689m"
DINO_V3_INPUT_SIZE = 256
DINO_V3_PATCH_SIZE = 16


def _run_dinov3_category(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    encoder_name: str = DINO_V3_ENCODER,
    num_neighbors: int = 1,
    masking: MaskingMode = "off",
    run_heatmap: bool = False,
    registry_base: Path | str = "data/models/dinov3",
    model_seed: int = PATCHCORE_MODEL_SEED,
    reuse_complete: bool = False,
) -> BaselineResult:
    """Run one DINOv3 category through the shared fair DINO engine.

    Args:
        data_root: Root directory of MVTec AD dataset.
        category: MVTec category to evaluate.
        pipeline: Optional preprocessing pipeline or configuration list.
        fpr_limit: Fair-eval AUPIMO upper FPR bound (fixed at 1e-4).
        encoder_name: Pretrained DINOv3 encoder identifier.
        num_neighbors: Number of normal patch neighbours to query.
        masking: Foreground masking policy (must be 'off' for DINOv3).
        run_heatmap: Whether to render test-image heatmaps.
        registry_base: Target artifact directory.
        model_seed: Deterministic model and data loader seed.
        reuse_complete: Return saved evaluation artifacts for an exact cache hit.

    Returns:
        BaselineResult for the category.

    Raises:
        ValueError: If masking is not 'off'.
    """
    if masking != "off":
        raise ValueError(
            "DINOv3 requires masking='off': Anomalib's published PCA threshold is calibrated for DINOv2 "
            "and can produce an empty DINOv3 memory bank."
        )
    return run_dino_category(
        data_root=data_root,
        category=category,
        pipeline=pipeline,
        fpr_limit=fpr_limit,
        encoder_name=encoder_name,
        num_neighbors=num_neighbors,
        masking=masking,
        run_heatmap=run_heatmap,
        registry_base=registry_base,
        model_seed=model_seed,
        reuse_complete=reuse_complete,
        model_generation="dinov3",
        model_name="DINOv3",
        input_size=DINO_V3_INPUT_SIZE,
        patch_size=DINO_V3_PATCH_SIZE,
        batch_size=4,
    )


def run_dinov3_baseline(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    encoder_name: str = DINO_V3_ENCODER,
    num_neighbors: int = 1,
    masking: MaskingMode = "off",
    run_heatmap: bool = False,
    registry_base: Path | str = "data/models/dinov3",
    model_seed: int = PATCHCORE_MODEL_SEED,
    reuse_complete: bool = False,
) -> BaselineResult | AllCategoriesResult:
    """Run the fair frozen-DINOv3 baseline for one or all MVTec categories.

    Args:
        data_root: Root directory of MVTec AD dataset.
        category: MVTec category or 'all' for full benchmark evaluation.
        pipeline: Optional preprocessing pipeline or configuration list.
        fpr_limit: Fair-eval AUPIMO upper FPR bound (fixed at 1e-4).
        encoder_name: Pretrained DINOv3 encoder identifier.
        num_neighbors: Number of normal patch neighbours to query.
        masking: Foreground masking policy (must be 'off' for DINOv3).
        run_heatmap: Whether to render test-image heatmaps.
        registry_base: Target artifact directory.
        model_seed: Deterministic model and data loader seed.
        reuse_complete: Return saved evaluation artifacts for an exact cache hit.

    Returns:
        BaselineResult for a single category, or AllCategoriesResult for 'all'.
    """
    if category != "all":
        return _run_dinov3_category(
            data_root=data_root,
            category=category,
            pipeline=pipeline,
            fpr_limit=fpr_limit,
            encoder_name=encoder_name,
            num_neighbors=num_neighbors,
            masking=masking,
            run_heatmap=run_heatmap,
            registry_base=registry_base,
            model_seed=model_seed,
            reuse_complete=reuse_complete,
        )

    if masking != "off":
        raise ValueError(
            "DINOv3 requires masking='off': Anomalib's published PCA threshold is calibrated for DINOv2 "
            "and can produce an empty DINOv3 memory bank."
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
        variant="baseline",
        feature_layers=(),
        position_radius=0,
        spatial_weight=0.0,
        density_neighbors=0,
        model_generation="dinov3",
        model_name="DINOv3",
        input_size=DINO_V3_INPUT_SIZE,
        patch_size=DINO_V3_PATCH_SIZE,
        batch_size=4,
        reuse_complete=reuse_complete,
        save_summary_files=True,
    )


__all__ = [
    "DINO_V3_ENCODER",
    "DINO_V3_INPUT_SIZE",
    "DINO_V3_PATCH_SIZE",
    "run_dinov3_baseline",
]
