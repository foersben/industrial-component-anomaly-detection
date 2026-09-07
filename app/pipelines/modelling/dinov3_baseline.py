"""Frozen DINOv3 patch-token nearest-neighbour baseline for MVTec AD."""

import csv
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

from app.core.logger import logger
from app.domain.data import build_mvtec_manifest
from app.pipelines.modelling.baseline import PATCHCORE_MODEL_SEED, BaselineResult
from app.pipelines.modelling.dinov2_baseline import (
    MVTEC_CATEGORIES,
    MaskingMode,
    _release_accelerator_memory,
    _run_dinov2_category,
)
from app.pipelines.preprocessing.base import PreprocessingPipeline

DINO_V3_ENCODER = "vit_small_patch16_dinov3.lvd1689m"
DINO_V3_INPUT_SIZE = 256
DINO_V3_PATCH_SIZE = 16


class AllCategoriesResult(TypedDict):
    """Results and unweighted macro averages for all MVTec categories."""

    category: Literal["all"]
    masking_mode: MaskingMode
    categories: dict[str, BaselineResult]
    macro_average: dict[str, float]


def _save_all_category_summary(
    registry_base: Path | str,
    encoder_name: str,
    masking: MaskingMode,
    num_neighbors: int,
    category_results: dict[str, BaselineResult],
    macro_average: dict[str, float],
) -> None:
    """Persist compact machine-readable summaries beside category artifacts."""
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
        "model": "DINOv3",
        "variant": "baseline_single_layer_1nn" if num_neighbors == 1 else f"baseline_single_layer_{num_neighbors}nn",
        "encoder_name": encoder_name,
        "masking_mode": masking,
        "categories": len(category_results),
        "protocol": "fair-eval-v1",
        "macro_average": macro_average,
        "category_metrics": category_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "category_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("category", *metric_keys))
        writer.writeheader()
        for name, metrics in category_metrics.items():
            writer.writerow({"category": name, **metrics})


def _run_dinov3_category(
    data_root: Path | str = "data/raw/mvtec_ad",
    category: str = "bottle",
    pipeline: list[dict[str, Any]] | PreprocessingPipeline | None = None,
    fpr_limit: float = 1e-4,
    encoder_name: str = DINO_V3_ENCODER,
    num_neighbors: int = 1,
    masking: MaskingMode = "off",
    run_heatmap: bool = False,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    registry_base: Path | str = "data/models/dinov3",
    model_seed: int = PATCHCORE_MODEL_SEED,
    reuse_complete: bool = False,
    manifest: pd.DataFrame | None = None,
) -> BaselineResult:
    """Run one DINOv3 category through the shared fair DINO evaluator."""
    if masking != "off":
        raise ValueError(
            "DINOv3 requires masking='off': Anomalib's published PCA threshold is calibrated for DINOv2 "
            "and can produce an empty DINOv3 memory bank."
        )
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
        reuse_complete=reuse_complete,
        model_generation="dinov3",
        model_name="DINOv3",
        input_size=DINO_V3_INPUT_SIZE,
        patch_size=DINO_V3_PATCH_SIZE,
        manifest=manifest,
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
    preprocessing_steps: list[dict[str, Any]] | None = None,
    registry_base: Path | str = "data/models/dinov3",
    model_seed: int = PATCHCORE_MODEL_SEED,
) -> BaselineResult | AllCategoriesResult:
    """Run the fair frozen-DINOv3 baseline for one or all MVTec categories.

    The scorer, split, validation-derived thresholds, metrics, and aggregation
    are identical to the DINOv2 baseline. The frozen encoder, its native
    patch-aligned input size, and the explicitly disabled masking differ.
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
            preprocessing_steps=preprocessing_steps,
            registry_base=registry_base,
            model_seed=model_seed,
        )

    manifest = build_mvtec_manifest(data_root)
    category_results: dict[str, BaselineResult] = {}
    for name in MVTEC_CATEGORIES:
        try:
            result = _run_dinov3_category(
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
                manifest=manifest,
            )
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
        result_data: Any = result
        return float(result_data[level][key])

    macro_average = {
        metric: float(np.mean([metric_value(result, level, key) for result in category_results.values()]))
        for metric, (level, key) in metric_paths.items()
    }
    _save_all_category_summary(
        registry_base,
        encoder_name,
        masking,
        num_neighbors,
        category_results,
        macro_average,
    )
    logger.info("DINOv3 all-category macro averages: %s", macro_average)
    return {
        "category": "all",
        "masking_mode": masking,
        "categories": category_results,
        "macro_average": macro_average,
    }


if __name__ == "__main__":
    run_dinov3_baseline()
