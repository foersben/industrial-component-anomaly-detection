import asyncio
from collections.abc import Generator
import os
import shutil

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
def mock_mvtec_dataset(tmp_path):
    """Creates a minimal mock MVTec AD directory structure."""
    dataset_dir = tmp_path / "mock_mvtec"
    category_dir = dataset_dir / "bottle"

    train_good_dir = category_dir / "train" / "good"
    test_good_dir = category_dir / "test" / "good"
    test_defect_dir = category_dir / "test" / "defect"
    ground_truth_dir = category_dir / "ground_truth" / "defect"

    for d in [train_good_dir, test_good_dir, test_defect_dir, ground_truth_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def create_image(path, size=(32, 32), color=(128, 128, 128)):
        img = Image.new('RGB', size, color=color)
        img.save(path)

    def create_mask(path, size=(32, 32), color=0):
        img = Image.new('L', size, color=color)
        img.save(path)

    create_image(train_good_dir / "000.png")
    create_image(test_good_dir / "000.png")
    create_image(test_defect_dir / "000.png", color=(255, 0, 0))
    create_mask(ground_truth_dir / "000_mask.png", color=255)

    return str(category_dir)

@pytest.fixture
def synthetic_crop_batch():
    """Returns random float32 batches normalized in [0, 1]."""
    def _get_batch(shape=(8, 32, 32, 3)):
        return np.random.rand(*shape).astype(np.float32)
    return _get_batch

@pytest.fixture
def mock_keras_cae():
    """Returns a minimal compiled CAE model for fast inference testing."""
    from app.pipelines.multi_stage_ae.cae_keras import build_cae
    model = build_cae(crop_size=32, latent_channels=8)
    # The model is compiled inside build_cae
    return model
