"""Dataset helpers shared by modelling experiments."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from app.core.logger import logger
from app.pipelines.preprocessing import PreprocessingTransformAdapter, build_pipeline_from_configs

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
    """Read image dimensions and colour mode without decoding all pixels.

    Args:
        path: The path to the image.

    Returns:
        A tuple containing the width, height, and colour mode of the image.

    Raises:
        ValueError: If the image cannot be read.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError) as error:
        logger.error("Could not read image metadata for %s: %s", path, error)
        raise ValueError(f"Could not read image metadata for {path}") from error

    return width, height, mode


class MVTecImageDataset(Dataset[tuple[Tensor, int, str]]):
    """Load manifest images as ``(tensor, anomaly label, path)`` tuples.

    Attributes:
        frame: Manifest rows containing ``path`` and ``is_anomaly``.
        transform: Callable converting an RGB PIL image to a tensor.
    """

    def __init__(self, frame: pd.DataFrame, transform: Callable[[Image.Image], Tensor]) -> None:
        """Initialize the dataset from a manifest subset and image transform.

        Args:
            frame: Manifest rows containing ``path`` and ``is_anomaly``.
            transform: Callable converting an RGB PIL image to a tensor.

        Raises:
            ValueError: If a required manifest column is missing.
        """
        required_columns = {"path", "is_anomaly"}
        if missing_columns := required_columns.difference(frame.columns):
            logger.error("frame is missing required columns: %s", sorted(missing_columns))
            raise ValueError(f"frame is missing required columns: {sorted(missing_columns)}")

        self.frame = frame.loc[:, ["path", "is_anomaly"]].reset_index(drop=True).copy()
        self.transform = transform
        logger.info("Initialized dataset with %d images", len(self.frame))

    def __len__(self) -> int:
        """Return the number of manifest rows.

        Returns:
            Number of manifest rows.
        """
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        """Load and transform one image.

        Args:
            index: The index of the image to load.

        Returns:
            A tuple containing the transformed image tensor, the anomaly label, and the path to the image.
        """
        row = self.frame.iloc[index]
        path = str(row["path"])
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor, int(row["is_anomaly"]), path


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
    logger.info("Building manifest for MVTec dataset at %s", root)
    dataset_root = Path(root).expanduser()
    if not dataset_root.exists():
        logger.error("MVTec dataset directory does not exist: %s", dataset_root)
        raise FileNotFoundError(f"MVTec dataset directory does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        logger.error("MVTec dataset path is not a directory: %s", dataset_root)
        raise NotADirectoryError(f"MVTec dataset path is not a directory: {dataset_root}")
    dataset_root = dataset_root.resolve()

    rows: list[dict[str, object]] = []

    all_images = (
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in all_images:
        split = image_path.parent.parent.name
        if split not in ("train", "test"):
            continue

        product_dir = image_path.parent.parent.parent
        defect_dir = image_path.parent

        is_anomaly = defect_dir.name != "good"
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

    # Sort rows to adhere to your test assertion (train before test, etc.):
    rows.sort(
        key=lambda row: (
            row["product"],
            0 if row["split"] == "train" else 1,
            row["defect_type"],
            row["path"],
        )
    )

    if not rows:
        logger.error("No MVTec images were found below: %s", dataset_root)
        raise ValueError(f"No MVTec images were found below: {dataset_root}")

    logger.info("Built manifest: %d images across %d categories", len(rows), len({r["product"] for r in rows}))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def create_mvtec_dataset(
    manifest: pd.DataFrame,
    preprocessing_steps: list[dict[str, Any]] | None = None,
    image_size: tuple[int, int] = (256, 256),
) -> MVTecImageDataset:
    """Create an MVTecImageDataset from a manifest and preprocessing steps.

    Args:
        manifest: The manifest DataFrame.
        preprocessing_steps: A list of preprocessing steps.
        image_size: The size of the images.

    Returns:
        An MVTecImageDataset.
    """
    pipeline = build_pipeline_from_configs(preprocessing_steps)
    adapter = PreprocessingTransformAdapter(pipeline)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            adapter,
            transforms.ToTensor(),
        ]
    )

    return MVTecImageDataset(frame=manifest, transform=transform)
