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
