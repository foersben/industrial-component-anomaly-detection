"""Unit tests for modelling dataset helpers."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from app.pipelines.modelling import MVTecImageDataset, build_mvtec_manifest


def test_build_mvtec_manifest_links_images_and_masks(tmp_path: Path) -> None:
    """Build manifest rows for normal training and anomalous test images."""
    good_path = tmp_path / "bottle" / "train" / "good" / "001.png"
    defect_path = tmp_path / "bottle" / "test" / "broken" / "002.png"
    mask_path = tmp_path / "bottle" / "ground_truth" / "broken" / "002_mask.png"
    for path in (good_path, defect_path, mask_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8)).save(path)

    manifest = build_mvtec_manifest(tmp_path)

    assert list(manifest["image_id"]) == ["001", "002"]
    assert list(manifest["is_anomaly"]) == [False, True]
    assert pd.isna(manifest.loc[0, "mask_path"])
    assert manifest.loc[1, "mask_path"] == str(mask_path.resolve())
    assert set(manifest[["width", "height", "mode"]].itertuples(index=False, name=None)) == {(12, 8, "RGB")}


def test_build_mvtec_manifest_rejects_missing_root(tmp_path: Path) -> None:
    """Report a clear error for a missing dataset directory."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_mvtec_manifest(tmp_path / "missing")


def test_mvtec_image_dataset_returns_tensor_label_and_path(tmp_path: Path) -> None:
    """Load manifest images using the notebook-facing tuple contract."""
    image_path = tmp_path / "grayscale.png"
    Image.new("L", (6, 4)).save(image_path)
    frame = pd.DataFrame({"path": [str(image_path)], "is_anomaly": [True]})
    dataset = MVTecImageDataset(frame, transform=transforms.ToTensor())

    image, label, path = dataset[0]

    assert len(dataset) == 1
    assert image.shape == torch.Size([3, 4, 6])
    assert label == 1
    assert path == str(image_path)
