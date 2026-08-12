"""Dataset helpers shared by modelling experiments."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png"})
MANIFEST_COLUMNS = [
    "path",
    "image_id",
    "product",
    "split",
    "defect_type",
    "is_anomaly",
    "width",
    "height",
    "mode",
    "mask_path",
]


def _read_image_metadata(path: Path) -> tuple[int, int, str]:
    """Read image dimensions and colour mode without decoding all pixels."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read image metadata for {path}") from error

    return width, height, mode


def build_mvtec_manifest(root: str | Path) -> pd.DataFrame:
    """Build a deterministic manifest of MVTec AD train and test images.

    Ground-truth masks are linked through ``mask_path`` rather than included as
    samples. A missing mask is represented by ``None``.

    Args:
        root: Directory containing MVTec product directories.

    Returns:
        One row per input image.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
        NotADirectoryError: If the dataset root is not a directory.
        ValueError: If an image is unreadable or no images are found.
    """
    dataset_root = Path(root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec dataset directory does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"MVTec dataset path is not a directory: {dataset_root}")
    dataset_root = dataset_root.resolve()

    rows: list[dict[str, object]] = []
    for product_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for split in ("train", "test"):
            split_dir = product_dir / split
            if not split_dir.is_dir():
                continue

            for defect_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                is_anomaly = defect_dir.name != "good"
                image_paths = sorted(
                    path for path in defect_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                for image_path in image_paths:
                    width, height, mode = _read_image_metadata(image_path)
                    mask_path = product_dir / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                    rows.append(
                        {
                            "path": str(image_path.resolve()),
                            "image_id": image_path.stem,
                            "product": product_dir.name,
                            "split": split,
                            "defect_type": defect_dir.name,
                            "is_anomaly": is_anomaly,
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "mask_path": str(mask_path.resolve()) if mask_path.is_file() else None,
                        }
                    )

    if not rows:
        raise ValueError(f"No MVTec images were found below: {dataset_root}")

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


class MVTecImageDataset(Dataset[tuple[Tensor, int, str]]):
    """Load manifest images as ``(tensor, anomaly label, path)`` tuples."""

    def __init__(self, frame: pd.DataFrame, transform: Callable[[Image.Image], Tensor]) -> None:
        """Initialize the dataset from a manifest subset and image transform.

        Args:
            frame: Manifest rows containing ``path`` and ``is_anomaly``.
            transform: Callable converting an RGB PIL image to a tensor.

        Raises:
            ValueError: If a required manifest column is missing.
        """
        required_columns = {"path", "is_anomaly"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"frame is missing required columns: {sorted(missing_columns)}")

        self.frame = frame.loc[:, ["path", "is_anomaly"]].reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of manifest rows."""
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        """Load and transform one image."""
        row = self.frame.iloc[index]
        path = str(row["path"])
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor, int(row["is_anomaly"]), path
