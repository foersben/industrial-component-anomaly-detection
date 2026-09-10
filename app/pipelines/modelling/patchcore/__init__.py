"""Canonical PatchCore modelling pipeline subpackage."""

from app.domain.data import FAIR_EVALUATION_PROTOCOL, FairEvaluationSplit
from app.pipelines.modelling.patchcore.dataset import (
    _configure_patchcore_partitions,
    _dataset_with_ordered_paths,
    _seed_patchcore_run,
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.pipelines.modelling.patchcore.evaluation import (
    _collect_batch_tensors,
    _load_heatmap_overlays,
    _process_and_save_level,
    _save_heatmap_overlays,
    _tensor_to_numpy,
    _to_float,
    extract_and_save_pr_metrics,
    format_results,
)
from app.pipelines.modelling.patchcore.pipeline import run_patchcore_pipeline
from app.pipelines.modelling.patchcore.registry import (
    _normalize_preprocessing_steps,
    delete_cached_patchcore_model,
    find_cached_patchcore_model,
    list_trashed_patchcore_models,
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
from app.pipelines.modelling.patchcore.visualization import (
    _add_panel_headers,
    _print_patchcore_results_table,
    _RawScoreImageVisualizer,
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
    "_RawScoreImageVisualizer",
    "_add_panel_headers",
    "_collect_batch_tensors",
    "_configure_patchcore_partitions",
    "_dataset_with_ordered_paths",
    "_load_heatmap_overlays",
    "_normalize_preprocessing_steps",
    "_print_patchcore_results_table",
    "_process_and_save_level",
    "_save_heatmap_overlays",
    "_seed_patchcore_run",
    "_tensor_to_numpy",
    "_to_float",
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
