"""Dataset helpers shared by modelling experiments."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from app.core.logger import logger

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

FAIR_EVALUATION_PROTOCOL = "fair-eval-v1"
FAIR_EVALUATION_SPLIT_SEED = 42
FAIR_EVALUATION_VALIDATION_FRACTION = 0.15


def _ordered_path_digest(paths: list[str]) -> str:
    """Return a deterministic digest that preserves path membership and order."""
    payload = "\n".join(str(Path(path).expanduser().resolve()) for path in paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FairEvaluationSplit:
    """Shared fitting, validation, and test partitions for one MVTec category."""

    category: str
    fitting: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int
    validation_fraction: float
    fitting_digest: str
    validation_digest: str
    test_digest: str

    @property
    def fitting_paths(self) -> list[str]:
        """Return fitting paths in their protocol-defined order."""
        return [str(path) for path in self.fitting["path"]]

    @property
    def validation_paths(self) -> list[str]:
        """Return validation paths in their protocol-defined order."""
        return [str(path) for path in self.validation["path"]]

    @property
    def test_paths(self) -> list[str]:
        """Return official test paths in their protocol-defined order."""
        return [str(path) for path in self.test["path"]]

    def evidence(self) -> dict[str, Any]:
        """Return serialisable protocol evidence for hashes and metadata."""
        return {
            "protocol": FAIR_EVALUATION_PROTOCOL,
            "split_seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "train_normal": len(self.fitting),
            "val_normal": len(self.validation),
            "test_total": len(self.test),
            "fitting_path_digest": self.fitting_digest,
            "validation_path_digest": self.validation_digest,
            "test_path_digest": self.test_digest,
        }


def build_fair_evaluation_split(
    manifest: pd.DataFrame,
    category: str,
    *,
    validation_fraction: float = FAIR_EVALUATION_VALIDATION_FRACTION,
    seed: int = FAIR_EVALUATION_SPLIT_SEED,
) -> FairEvaluationSplit:
    """Build the deterministic shared baseline-evaluation split.

    Only official normal training rows may enter fitting or validation. Official
    test rows retain the manifest's existing deterministic order.
    """
    from sklearn.model_selection import train_test_split

    required = {"path", "product", "split", "is_anomaly"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    category_rows = manifest.loc[manifest["product"] == category].copy()
    if category_rows.empty:
        raise ValueError(f"No manifest rows found for category '{category}'")

    normal_train = category_rows.loc[
        (category_rows["split"] == "train") & (~category_rows["is_anomaly"].astype(bool))
    ].sort_values("path", kind="stable")
    official_test = category_rows.loc[category_rows["split"] == "test"]
    if normal_train.empty:
        raise ValueError(f"No official normal training rows found for category '{category}'")
    if official_test.empty:
        raise ValueError(f"No official test rows found for category '{category}'")

    fitting, validation = train_test_split(
        normal_train,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    fitting = fitting.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    official_test = official_test.reset_index(drop=True)

    fitting_paths = fitting["path"].astype(str).tolist()
    validation_paths = validation["path"].astype(str).tolist()
    test_paths = official_test["path"].astype(str).tolist()
    fitting_set = set(fitting_paths)
    validation_set = set(validation_paths)
    test_set = set(test_paths)
    if fitting_set & validation_set or fitting_set & test_set or validation_set & test_set:
        raise ValueError("Fair-evaluation partitions contain overlapping image paths")
    if fitting_set | validation_set != set(normal_train["path"].astype(str)):
        raise ValueError("Fitting and validation partitions do not exhaust normal training rows")

    return FairEvaluationSplit(
        category=category,
        fitting=fitting,
        validation=validation,
        test=official_test,
        seed=seed,
        validation_fraction=validation_fraction,
        fitting_digest=_ordered_path_digest(fitting_paths),
        validation_digest=_ordered_path_digest(validation_paths),
        test_digest=_ordered_path_digest(test_paths),
    )


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
    from app.pipelines.preprocessing import PreprocessingTransformAdapter, build_pipeline_from_configs

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
