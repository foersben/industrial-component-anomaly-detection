"""Canonical PatchCore modelling pipeline subpackage."""

from app.domain.data import FAIR_EVALUATION_PROTOCOL, FairEvaluationSplit
from app.pipelines.modelling.patchcore.dataset import (
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.pipelines.modelling.patchcore.evaluation import (
    extract_and_save_pr_metrics,
    format_results,
)
from app.pipelines.modelling.patchcore.pipeline import run_patchcore_pipeline
from app.pipelines.modelling.patchcore.registry import (
    delete_cached_patchcore_model,
    find_cached_patchcore_model,
    list_trashed_patchcore_models,
    normalize_preprocessing_steps,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
)
from app.pipelines.modelling.patchcore.types import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    BaselineResult,
    MetricLevelResult,
)

__all__ = [
    "FAIR_EVALUATION_PROTOCOL",
    "PATCHCORE_IMAGE_THRESHOLD_QUANTILE",
    "PATCHCORE_MODEL_SEED",
    "PATCHCORE_PIXEL_THRESHOLD_QUANTILE",
    "PATCHCORE_SCORE_SPACE",
    "BaselineResult",
    "FairEvaluationSplit",
    "MetricLevelResult",
    "build_fair_evaluation_split",
    "build_mvtec_manifest",
    "delete_cached_patchcore_model",
    "extract_and_save_pr_metrics",
    "find_cached_patchcore_model",
    "format_results",
    "list_trashed_patchcore_models",
    "normalize_preprocessing_steps",
    "purge_patchcore_trash",
    "restore_cached_patchcore_model",
    "run_patchcore_pipeline",
]
