"""Shared pytest fixtures and test configuration."""

import asyncio
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Provide a dedicated, session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mvtec_dataset(tmp_path: Path) -> str:
    """Creates a minimal mock MVTec AD directory structure."""
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

    create_image(train_good_dir / "000.png")
    create_image(test_good_dir / "000.png")
    create_image(test_defect_dir / "000.png", color=(255, 0, 0))
    create_mask(ground_truth_dir / "000_mask.png", color=255)

    return str(category_dir)


@pytest.fixture
def synthetic_crop_batch() -> Callable[..., np.ndarray]:
    """Returns random float32 batches normalized in [0, 1]."""

    def _get_batch(shape: tuple[int, ...] = (8, 32, 32, 3)) -> np.ndarray:
        return np.random.rand(*shape).astype(np.float32)

    return _get_batch


@pytest.fixture
def mock_keras_cae() -> Any:
    """Returns a minimal compiled CAE model for fast inference testing."""
    from app.pipelines.multi_stage_ae.cae_keras import build_cae

    return build_cae(crop_size=32, latent_channels=8)
