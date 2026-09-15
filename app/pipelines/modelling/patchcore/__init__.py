"""Canonical PatchCore modelling pipeline subpackage."""

from app.core.registry import (
    delete_cached_model as delete_cached_patchcore_model,
)
from app.core.registry import (
    list_trashed_models as list_trashed_patchcore_models,
)
from app.core.registry import (
    purge_trash as purge_patchcore_trash,
)
from app.core.registry import (
    restore_cached_model as restore_cached_patchcore_model,
)
from app.domain.data import (
    FAIR_EVALUATION_PROTOCOL,
    FairEvaluationSplit,
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.domain.evaluation import (
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    BaselineResult,
    ConfusionMatrix,
    EvaluationArtifacts,
    ImageEvaluationMetrics,
    MetricLevelResult,
    PixelEvaluationMetrics,
    Thresholds,
)
from app.pipelines.modelling.patchcore.evaluation import (
    extract_and_save_pr_metrics,
    format_results,
)
from app.pipelines.modelling.patchcore.pipeline import run_patchcore_pipeline
from app.pipelines.modelling.patchcore.registry import find_cached_patchcore_model

__all__ = [
    "FAIR_EVALUATION_PROTOCOL",
    "PATCHCORE_IMAGE_THRESHOLD_QUANTILE",
    "PATCHCORE_MODEL_SEED",
    "PATCHCORE_PIXEL_THRESHOLD_QUANTILE",
    "PATCHCORE_SCORE_SPACE",
    "BaselineResult",
    "ConfusionMatrix",
    "EvaluationArtifacts",
    "FairEvaluationSplit",
    "ImageEvaluationMetrics",
    "MetricLevelResult",
    "PixelEvaluationMetrics",
    "Thresholds",
    "build_fair_evaluation_split",
    "build_mvtec_manifest",
    "delete_cached_patchcore_model",
    "extract_and_save_pr_metrics",
    "find_cached_patchcore_model",
    "format_results",
    "list_trashed_patchcore_models",
    "purge_patchcore_trash",
    "restore_cached_patchcore_model",
    "run_patchcore_pipeline",
]
