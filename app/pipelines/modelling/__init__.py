"""Modelling pipelines."""

from app.domain.data import MVTecImageDataset, build_mvtec_manifest
from app.pipelines.modelling.autoencoder import (
    ConvAutoencoder,
    evaluate_autoencoder,
    run_autoencoder_pipeline,
)
from app.pipelines.modelling.baseline import run_baseline
from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

__all__ = [
    "ConvAutoencoder",
    "MVTecImageDataset",
    "build_mvtec_manifest",
    "evaluate_autoencoder",
    "run_autoencoder_pipeline",
    "run_baseline",
    "run_dummy_evaluation",
    "run_real_data_dummy",
]
