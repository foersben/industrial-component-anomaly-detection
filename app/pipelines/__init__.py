"""Anomaly detection pipelines for industrial components."""

from .modelling.baseline import run_baseline
from .modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

__all__ = ["run_baseline", "run_dummy_evaluation", "run_real_data_dummy"]
