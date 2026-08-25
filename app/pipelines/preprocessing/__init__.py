"""Preprocessing subpackage for image preprocessing steps and pipelines."""

from app.pipelines.preprocessing.adapter import PreprocessedAnomalibDataset, PreprocessingTransformAdapter
from app.pipelines.preprocessing.augmentation import (
    ObjectAugmenter,
    TextureAugmenter,
    augment_batch,
    get_augmenter,
)
from app.pipelines.preprocessing.base import BasePreprocessingStep, PreprocessingPipeline
from app.pipelines.preprocessing.factory import STEP_REGISTRY, build_pipeline_from_configs
from app.pipelines.preprocessing.segmentation import OtsuCannySegmentor, extract_largest_component
from app.pipelines.preprocessing.steps import CLAHEStep, ForegroundMaskStep, GaussianBlurStep

__all__ = [
    "STEP_REGISTRY",
    "BasePreprocessingStep",
    "CLAHEStep",
    "ForegroundMaskStep",
    "GaussianBlurStep",
    "ObjectAugmenter",
    "OtsuCannySegmentor",
    "PreprocessedAnomalibDataset",
    "PreprocessingPipeline",
    "PreprocessingTransformAdapter",
    "TextureAugmenter",
    "augment_batch",
    "build_pipeline_from_configs",
    "extract_largest_component",
    "get_augmenter",
]
