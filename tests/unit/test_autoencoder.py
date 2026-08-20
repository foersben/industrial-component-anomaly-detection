"""Unit tests for Convolutional Autoencoder modelling and evaluation."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from app.pipelines.modelling.autoencoder import ConvAutoencoder, evaluate_autoencoder, run_autoencoder_pipeline


def test_conv_autoencoder_forward_shape() -> None:
    """Verify ConvAutoencoder preserves input spatial dimensions."""
    model = ConvAutoencoder(in_channels=3, latent_channels=16)
    x = torch.randn(2, 3, 32, 32)
    rec = model(x)
    assert rec.shape == (2, 3, 32, 32)


def test_evaluate_autoencoder() -> None:
    """Verify evaluation helper computes scores and classification report."""
    model = ConvAutoencoder(in_channels=3, latent_channels=8)

    # 4 samples: 2 normal (label 0), 2 defective (label 1)
    x = torch.rand(4, 3, 16, 16)
    y = torch.tensor([0, 0, 1, 1])
    paths = ["good1", "good2", "def1", "def2"]

    dataset = TensorDataset(x, y)
    # Wrap in custom loader yielding (x, y, path)
    loader = [(x_batch, y_batch, paths) for x_batch, y_batch in DataLoader(dataset, batch_size=4)]

    scores, true_labels, metrics = evaluate_autoencoder(model, loader)  # type: ignore[arg-type]

    assert len(scores) == 4
    assert len(true_labels) == 4
    assert "threshold" in metrics
    assert "accuracy" in metrics
    assert "report" in metrics
    assert isinstance(metrics["threshold"], float)


def test_run_autoencoder_pipeline_with_mock_dataset(tmp_path: Path) -> None:
    """Run minimal end-to-end autoencoder pipeline on temporary mock dataset."""
    # Create minimal mock mvtec dataset structure
    for split, defect, name in [
        ("train", "good", "000.png"),
        ("train", "good", "001.png"),
        ("test", "good", "000.png"),
        ("test", "defect", "001.png"),
    ]:
        img_dir = tmp_path / "bottle" / split / defect
        img_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32)).save(img_dir / name)

    metrics = run_autoencoder_pipeline(
        data_root=str(tmp_path),
        category="bottle",
        epochs=1,
        batch_size=2,
        latent_dim=16,
        img_size=32,
    )

    assert metrics["category"] == "bottle"
    assert metrics["epochs"] == 1
    assert "threshold" in metrics
    assert "final_train_loss" in metrics
