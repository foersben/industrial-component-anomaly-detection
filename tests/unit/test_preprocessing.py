"""Unit tests for the preprocessing subpackage."""

import numpy as np
import torch
from PIL import Image

from app.pipelines.preprocessing import (
    CLAHEStep,
    GaussianBlurStep,
    PreprocessingPipeline,
    PreprocessingTransformAdapter,
    build_pipeline_from_configs,
)


def test_clahe_step() -> None:
    """Test CLAHEStep execution."""
    step = CLAHEStep(clip_limit=3.0)

    assert step.name == "clahe"

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = step(img)

    assert res.shape == (64, 64, 3)


def test_gaussian_blur_step() -> None:
    """Test GaussianBlurStep execution."""
    step = GaussianBlurStep(kernel_size=5, sigma=1.0)

    assert step.name == "gaussian_blur"

    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    res = step(img)

    assert res.shape == (64, 64, 3)


def test_preprocessing_pipeline_chaining() -> None:
    """Test chaining multiple steps in a PreprocessingPipeline."""
    pipeline = PreprocessingPipeline()
    pipeline.add_step(CLAHEStep(clip_limit=2.0)).add_step(GaussianBlurStep(kernel_size=3))

    assert len(pipeline) == 2

    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    res = pipeline(img)

    assert res.shape == (64, 64, 3)


def test_factory_build_pipeline() -> None:
    """Test building pipeline from dict configurations."""
    configs = [
        {"name": "clahe", "params": {"clip_limit": 2.5}},
        {"name": "gaussian_blur", "params": {"kernel_size": 5}},
    ]
    pipeline = build_pipeline_from_configs(configs)

    assert len(pipeline) == 2
    assert isinstance(pipeline.steps[0], CLAHEStep)
    assert isinstance(pipeline.steps[1], GaussianBlurStep)


def test_transform_adapter_pil() -> None:
    """Test PreprocessingTransformAdapter on PIL Image."""
    pipeline = PreprocessingPipeline([CLAHEStep()])
    adapter = PreprocessingTransformAdapter(pipeline)
    pil_img = Image.new("RGB", (64, 64), color="red")
    res = adapter(pil_img)

    assert isinstance(res, Image.Image)


def test_transform_adapter_tensor() -> None:
    """Test PreprocessingTransformAdapter on PyTorch Tensor (C x H x W)."""
    pipeline = PreprocessingPipeline([CLAHEStep()])
    adapter = PreprocessingTransformAdapter(pipeline)
    tensor_img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    res = adapter(tensor_img)

    assert isinstance(res, torch.Tensor)
    assert res.shape == (3, 64, 64)
