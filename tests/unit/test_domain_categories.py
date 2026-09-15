"""Unit tests for domain category models and dataset discovery."""

from pathlib import Path

from app.domain.categories import (
    ANOMALY_DINO_MASKED_CATEGORIES,
    MVTEC_CATEGORIES,
    MVTEC_OBJECT_CATEGORIES,
    MVTEC_TEXTURE_CATEGORIES,
    OBJECT_CATEGORIES,
    TEXTURE_CATEGORIES,
    discover_dataset_categories,
)


def test_category_constants_invariants() -> None:
    """Verify official benchmark category counts and disjoint partition invariants."""
    assert len(MVTEC_OBJECT_CATEGORIES) == 10
    assert len(MVTEC_TEXTURE_CATEGORIES) == 5
    assert len(MVTEC_CATEGORIES) == 15
    assert len(OBJECT_CATEGORIES) == 10
    assert len(TEXTURE_CATEGORIES) == 5

    # Object and texture categories must be strictly disjoint
    assert OBJECT_CATEGORIES.isdisjoint(TEXTURE_CATEGORIES)

    # Union of objects and textures must equal the full benchmark set
    assert set(MVTEC_CATEGORIES) == OBJECT_CATEGORIES | TEXTURE_CATEGORIES

    # DINO masked categories must be a subset of rigid objects
    assert ANOMALY_DINO_MASKED_CATEGORIES.issubset(OBJECT_CATEGORIES)


def test_discover_dataset_categories_defaults() -> None:
    """Verify fallback behavior when path is None or does not exist."""
    assert discover_dataset_categories(None) == list(MVTEC_CATEGORIES)
    assert discover_dataset_categories("non_existent_path_xyz_123") == list(MVTEC_CATEGORIES)


def test_discover_dataset_categories_from_directory(tmp_path: Path) -> None:
    """Verify dynamic discovery of directory categories excluding hidden folders."""
    (tmp_path / "bottle").mkdir()
    (tmp_path / "capsule").mkdir()
    (tmp_path / ".hidden_folder").mkdir()
    (tmp_path / "_temp_folder").mkdir()
    (tmp_path / "regular_file.txt").write_text("hello", encoding="utf-8")

    discovered = discover_dataset_categories(tmp_path)
    assert discovered == ["bottle", "capsule"]


def test_discover_dataset_categories_empty_directory(tmp_path: Path) -> None:
    """Verify empty directory falls back to canonical benchmark categories."""
    assert discover_dataset_categories(tmp_path) == list(MVTEC_CATEGORIES)
