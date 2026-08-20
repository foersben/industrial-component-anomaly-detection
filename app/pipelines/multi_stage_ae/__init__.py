"""Multi-Stage Autoencoder (AE) Pipeline Package.

This package provides a modular, multi-step autoencoder architecture for industrial
component anomaly detection.
"""

from app.pipelines.multi_stage_ae.cae_pipeline import run_keras_cae_pipeline

__all__ = [
    "run_keras_cae_pipeline",
]
