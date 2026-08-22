"""Shared pytest fixtures, configuration, and synthetic generators for the test suite.

This module provides common fixtures for asynchronous testing, temporary mock MVTec AD
datasets of varying sizes, pre-compiled lightweight Keras CAE models, and synthetic batch
generators used across unit and integration tests.
"""

import asyncio
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide a dedicated, session-scoped event loop for async tests.

    Yields:
        asyncio.AbstractEventLoop: An isolated event loop instance.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mvtec_dataset(tmp_path: Path) -> str:
    """Create a minimal mock MVTec AD directory structure with train, test, and ground truth files.

    Generates 10 normal training images, 2 normal test images, and 2 anomalous test images
    with associated ground truth masks for category 'bottle'.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        str: Absolute string path to the mock dataset root directory.
    """
    dataset_dir = tmp_path / "mock_mvtec"
    category_dir = dataset_dir / "bottle"

    train_good_dir = category_dir / "train" / "good"
    test_good_dir = category_dir / "test" / "good"
    test_defect_dir = category_dir / "test" / "defect"
    ground_truth_dir = category_dir / "ground_truth" / "defect"

    for d in [train_good_dir, test_good_dir, test_defect_dir, ground_truth_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def create_image(
        path: Path, size: tuple[int, int] = (32, 32), color: tuple[int, int, int] = (128, 128, 128)
    ) -> None:
        img = Image.new("RGB", size, color=color)
        img.save(path)

    def create_mask(path: Path, size: tuple[int, int] = (32, 32), color: int = 0) -> None:
        img = Image.new("L", size, color=color)
        img.save(path)

    for i in range(10):
        create_image(train_good_dir / f"{i:03d}.png")

    for i in range(2):
        create_image(test_good_dir / f"{i:03d}.png")
        create_image(test_defect_dir / f"{i:03d}.png", color=(255, 0, 0))
        create_mask(ground_truth_dir / f"{i:03d}_mask.png", color=255)

    return str(dataset_dir)


@pytest.fixture
def mock_large_mvtec_dataset(tmp_path: Path) -> str:
    """Create a larger mock MVTec AD dataset suitable for train/val split isolation testing.

    Generates 20 normal training images, 10 normal test images, and 10 anomalous test
    images with ground truth masks.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        str: Absolute string path to the mock dataset root directory.
    """
    dataset_dir = tmp_path / "mock_mvtec_large"
    category_dir = dataset_dir / "bottle"

    train_good_dir = category_dir / "train" / "good"
    test_good_dir = category_dir / "test" / "good"
    test_defect_dir = category_dir / "test" / "defect"
    ground_truth_dir = category_dir / "ground_truth" / "defect"

    for d in [train_good_dir, test_good_dir, test_defect_dir, ground_truth_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        img.save(train_good_dir / f"{i:03d}.png")

    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        img.save(test_good_dir / f"{i:03d}.png")
        img_defect = Image.new("RGB", (32, 32), color=(255, 0, 0))
        img_defect.save(test_defect_dir / f"{i:03d}.png")
        mask = Image.new("L", (32, 32), color=255)
        mask.save(ground_truth_dir / f"{i:03d}_mask.png")

    return str(dataset_dir)


@pytest.fixture
def synthetic_crop_batch() -> Callable[..., np.ndarray]:
    """Provide a factory function returning random float32 batches normalized in [0, 1].

    Returns:
        Callable[..., np.ndarray]: Factory function accepting a tensor shape tuple.
    """

    def _get_batch(shape: tuple[int, ...] = (8, 32, 32, 3)) -> np.ndarray:
        return np.random.rand(*shape).astype(np.float32)

    return _get_batch


@pytest.fixture
def mock_keras_cae() -> Any:
    """Construct a minimal compiled Keras CAE model for fast inference testing.

    Returns:
        Any: Compiled tf.keras.Model with crop_size=32 and latent_channels=8.
    """
    from app.pipelines.multi_stage_ae.cae_keras import build_cae

    return build_cae(crop_size=32, latent_channels=8)


@pytest.fixture
def mock_fast_training() -> Generator[None, None, None]:
    """Patch train_cae to return dummy loss history without executing training epochs.

    Yields:
        None
    """

    def _mock_train_cae(*_args: Any, **_kwargs: Any) -> dict[str, list[float]]:
        return {"train": [0.1], "val_good": [0.1], "val_anomalous": [0.2]}

    with patch("app.pipelines.multi_stage_ae.cae_pipeline.train_cae", side_effect=_mock_train_cae):
        yield
