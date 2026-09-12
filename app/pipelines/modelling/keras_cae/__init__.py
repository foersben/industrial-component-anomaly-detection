"""Keras Convolutional Autoencoder (CAE) modelling subpackage."""

from app.core.registry import (
    delete_cached_model,
    list_trashed_models,
    purge_trash,
    restore_cached_model,
)
from app.pipelines.modelling.keras_cae.cae_keras import build_cae
from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline
from app.pipelines.modelling.keras_cae.crops import extract_crops, stitch_crops
from app.pipelines.modelling.keras_cae.registry import find_cached_model

__all__ = [
    "build_cae",
    "delete_cached_model",
    "extract_crops",
    "find_cached_model",
    "list_trashed_models",
    "purge_trash",
    "restore_cached_model",
    "run_keras_cae_pipeline",
    "stitch_crops",
]
