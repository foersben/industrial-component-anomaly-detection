"""Anomaly detection pipelines for industrial components."""

from .baseline import run_baseline
from .dummy_classifier import run_dummy_evaluation, run_real_data_dummy

__all__ = ["run_baseline", "run_dummy_evaluation", "run_real_data_dummy"]
