"""Preprocessing factory module for building preprocessing pipelines."""

from typing import Any

from app.pipelines.preprocessing.base import BasePreprocessingStep, PreprocessingPipeline
from app.pipelines.preprocessing.steps import CLAHEStep, ForegroundMaskStep, GaussianBlurStep

# Registry mapping configuration keys to classes
STEP_REGISTRY: dict[str, type[BasePreprocessingStep]] = {
    CLAHEStep.name: CLAHEStep,
    GaussianBlurStep.name: GaussianBlurStep,
    ForegroundMaskStep.name: ForegroundMaskStep,
}


def build_pipeline_from_configs(
    configs: list[dict[str, Any]] | None,
) -> PreprocessingPipeline:
    """Build a PreprocessingPipeline from a list of dict configs.

    Example input:
        [
            {"name": "clahe", "params": {"clip_limit": 3.0}},
            {"name": "gaussian_blur", "params": {"kernel_size": 3}}
        ]

    Args:
        configs: List of preprocessing step configurations.

    Returns:
        PreprocessingPipeline with steps added from configs.
    """
    pipeline = PreprocessingPipeline()
    if not configs:
        return pipeline

    for config in configs:
        step_name = config.get("name")
        params = config.get("params", {})
        if step_name in STEP_REGISTRY:
            step_cls = STEP_REGISTRY[step_name]
            pipeline.add_step(step_cls(**params))

    return pipeline


def normalize_preprocessing_steps(
    steps: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize a list of preprocessing step dicts for deterministic hashing and registry lookups.

    Sorts parameter keys alphabetically and strips extraneous or unhashable attributes.

    Args:
        steps: List of raw preprocessing step dictionaries (e.g. `[{"name": "clahe", "params": {...}}]`).

    Returns:
        List of normalized step dictionaries with sorted param dictionaries.
    """
    if not steps:
        return []

    normalized: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict):
            item: dict[str, Any] = {"name": step.get("name")}
            params = step.get("params")
            if isinstance(params, dict):
                item["params"] = dict(sorted(params.items()))
            normalized.append(item)
    return normalized
