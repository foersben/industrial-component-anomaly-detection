"""Modelling pipelines."""

from app.pipelines.modelling.autoencoder import (
    ConvAutoencoder,
    evaluate_autoencoder,
    run_autoencoder_pipeline,
)
from app.pipelines.modelling.dinov2_baseline import run_dinov2_baseline
from app.pipelines.modelling.dinov3_baseline import run_dinov3_baseline
from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy
from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline
from app.pipelines.modelling.patchcore import (
    delete_cached_patchcore_model,
    find_cached_patchcore_model,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
    run_patchcore_pipeline,
)

__all__ = [
    "ConvAutoencoder",
    "delete_cached_patchcore_model",
    "evaluate_autoencoder",
    "find_cached_patchcore_model",
    "list_trashed_patchcore_models",
    "purge_patchcore_trash",
    "restore_cached_patchcore_model",
    "run_autoencoder_pipeline",
    "run_dinov2_baseline",
    "run_dinov3_baseline",
    "run_dummy_evaluation",
    "run_keras_cae_pipeline",
    "run_patchcore_pipeline",
    "run_real_data_dummy",
]
