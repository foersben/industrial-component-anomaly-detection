"""Multi-Stage Autoencoder (AE) Pipeline Package.

This package provides a modular, multi-step autoencoder architecture for industrial
component anomaly detection.
"""

from app.pipelines.multi_stage_ae.cae_pipeline import delete_cached_model, run_keras_cae_pipeline

__all__ = [
    "delete_cached_model",
    "run_keras_cae_pipeline",
]
