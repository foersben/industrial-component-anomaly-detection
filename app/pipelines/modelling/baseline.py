"""Backward-compatibility module for PatchCore baseline.

This module re-exports all components from `app.pipelines.modelling.patchcore`
to maintain backward compatibility with existing scripts, notebooks, and tests.
New code should import directly from `app.pipelines.modelling.patchcore`.
"""

from typing import Any

import app.pipelines.modelling.patchcore as _pc
from app.pipelines.modelling.patchcore import (
    FAIR_EVALUATION_PROTOCOL,
    PATCHCORE_IMAGE_THRESHOLD_QUANTILE,
    PATCHCORE_MODEL_SEED,
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    PATCHCORE_SCORE_SPACE,
    BaselineResult,
    FairEvaluationSplit,
    _add_panel_headers,
    _collect_batch_tensors,
    _configure_patchcore_partitions,
    _dataset_with_ordered_paths,
    _load_heatmap_overlays,
    _normalize_preprocessing_steps,
    _print_patchcore_results_table,
    _process_and_save_level,
    _RawScoreImageVisualizer,
    _save_heatmap_overlays,
    _seed_patchcore_run,
    _tensor_to_numpy,
    _to_float,
    build_fair_evaluation_split,
    build_mvtec_manifest,
    delete_cached_patchcore_model,
    extract_and_save_pr_metrics,
    find_cached_patchcore_model,
    format_results,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
)


def run_baseline(*args: Any, **kwargs: Any) -> BaselineResult:
    """Run PatchCore pipeline via backward-compatible baseline entrypoint."""
    # Synchronize monkeypatched attributes from baseline namespace to patchcore during execution
    patched = {}
    for attr in ("build_mvtec_manifest", "build_fair_evaluation_split"):
        if attr in globals() and getattr(_pc, attr, None) is not globals()[attr]:
            patched[attr] = getattr(_pc, attr)
            setattr(_pc, attr, globals()[attr])
    try:
        return _pc.run_patchcore_pipeline(*args, **kwargs)
    finally:
        for attr, orig in patched.items():
            setattr(_pc, attr, orig)


run_patchcore_pipeline = run_baseline

__all__ = [
    "FAIR_EVALUATION_PROTOCOL",
    "PATCHCORE_IMAGE_THRESHOLD_QUANTILE",
    "PATCHCORE_MODEL_SEED",
    "PATCHCORE_PIXEL_THRESHOLD_QUANTILE",
    "PATCHCORE_SCORE_SPACE",
    "BaselineResult",
    "FairEvaluationSplit",
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
    "run_baseline",
    "run_patchcore_pipeline",
]
