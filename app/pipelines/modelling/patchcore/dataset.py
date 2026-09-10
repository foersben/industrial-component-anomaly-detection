"""Dataset partitioning and seeding utilities for the PatchCore pipeline."""

from copy import copy
from pathlib import Path
from typing import Any

from anomalib.data import MVTecAD
from lightning import seed_everything

from app.domain.data import (
    FairEvaluationSplit,
    build_fair_evaluation_split,
    build_mvtec_manifest,
)
from app.pipelines.preprocessing.adapter import (
    PreprocessedAnomalibDataset,
    PreprocessingTransformAdapter,
)


def _seed_patchcore_run(seed: int) -> None:
    """Seed every random source used by PatchCore and its data loaders.

    Args:
        seed: Random seed integer for reproducible execution.
    """
    seed_everything(seed, workers=True, verbose=False)


def _dataset_with_ordered_paths(dataset: Any, ordered_paths: list[str]) -> Any:
    """Copy an Anomalib dataset and restrict it to an exact ordered path list.

    Args:
        dataset: Anomalib dataset object exposing a `.samples` DataFrame.
        ordered_paths: Ordered list of file paths to select and order samples by.

    Returns:
        Restricted dataset copy matching the ordered paths.

    Raises:
        TypeError: If dataset does not expose a `.samples` DataFrame with `image_path`.
        ValueError: If any requested paths are missing from the dataset samples.
    """
    samples = getattr(dataset, "samples", None)
    if samples is None or "image_path" not in samples.columns:
        raise TypeError("PatchCore dataset must expose a samples frame with image_path")
    indexed = {
        str(Path(path).expanduser().resolve()): index for index, path in enumerate(samples["image_path"].astype(str))
    }
    requested = [str(Path(path).expanduser().resolve()) for path in ordered_paths]
    missing = [path for path in requested if path not in indexed]
    if missing:
        raise ValueError(f"PatchCore dataset is missing {len(missing)} protocol paths")
    restricted = copy(dataset)
    restricted.samples = samples.iloc[[indexed[path] for path in requested]].reset_index(drop=True).copy()
    return restricted


def _configure_patchcore_partitions(
    datamodule: MVTecAD,
    fair_split: FairEvaluationSplit,
    transform_adapter: PreprocessingTransformAdapter | None = None,
) -> None:
    """Replace Anomalib's implicit partitions with the shared fair evaluation split.

    Args:
        datamodule: Anomalib MVTecAD datamodule instance.
        fair_split: Strict fair evaluation split instance.
        transform_adapter: Optional preprocessing transform adapter.
    """
    datamodule.setup()
    source_train = datamodule.train_data
    source_test = datamodule.test_data
    train_data = _dataset_with_ordered_paths(source_train, fair_split.fitting_paths)
    validation_data = _dataset_with_ordered_paths(source_train, fair_split.validation_paths)
    test_data = _dataset_with_ordered_paths(source_test, fair_split.test_paths)
    if transform_adapter is not None:
        train_data = PreprocessedAnomalibDataset(train_data, transform_adapter)
        validation_data = PreprocessedAnomalibDataset(validation_data, transform_adapter)
        test_data = PreprocessedAnomalibDataset(test_data, transform_adapter)
    datamodule.train_data = train_data
    datamodule.val_data = validation_data
    datamodule.test_data = test_data


__all__ = [
    "build_fair_evaluation_split",
    "build_mvtec_manifest",
]
