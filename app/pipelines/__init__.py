"""Anomaly detection pipelines for industrial components."""

from .modelling.autoencoder import (
    ConvAutoencoder,
    evaluate_autoencoder,
    run_autoencoder_pipeline,
)
from .modelling.baseline import run_baseline
from .modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

__all__ = [
    "ConvAutoencoder",
    "evaluate_autoencoder",
    "run_autoencoder_pipeline",
    "run_baseline",
    "run_dummy_evaluation",
    "run_real_data_dummy",
]
