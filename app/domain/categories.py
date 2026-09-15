"""Domain models and canonical category definitions for the MVTec AD dataset.

Provides centralized category constants, semantic splits (objects vs textures),
and dynamic dataset discovery utilities.
"""

from pathlib import Path

MVTEC_OBJECT_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
"""Rigid industrial object categories in MVTec AD requiring orientation preservation."""

MVTEC_TEXTURE_CATEGORIES: tuple[str, ...] = (
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
)
"""Spatially invariant surface texture categories in MVTec AD."""

MVTEC_CATEGORIES: tuple[str, ...] = tuple(sorted(MVTEC_OBJECT_CATEGORIES + MVTEC_TEXTURE_CATEGORIES))
"""All 15 official canonical benchmark categories of the MVTec AD dataset."""

OBJECT_CATEGORIES: frozenset[str] = frozenset(MVTEC_OBJECT_CATEGORIES)
"""Set representation of rigid object categories for fast O(1) membership checks."""

TEXTURE_CATEGORIES: frozenset[str] = frozenset(MVTEC_TEXTURE_CATEGORIES)
"""Set representation of surface texture categories for fast O(1) membership checks."""

ANOMALY_DINO_MASKED_CATEGORIES: frozenset[str] = frozenset(
    {
        "capsule",
        "hazelnut",
        "pill",
        "screw",
        "toothbrush",
    }
)
"""Categories where background suppression/masking improves DINO representation."""


def discover_dataset_categories(data_root: str | Path | None = None) -> list[str]:
    """Discover available MVTec categories dynamically from a dataset root directory.

    If the specified directory exists and contains category subfolders, this function
    returns a sorted list of discovered folder names. If the directory does not exist,
    is unreadable, or contains no valid subfolders, it safely falls back to the
    canonical 15 MVTec AD benchmark categories.

    Args:
        data_root: Path or string pointing to the dataset root folder.

    Returns:
        Sorted list of discovered or fallback category names.
    """
    if data_root is None:
        return list(MVTEC_CATEGORIES)

    root_path = Path(data_root)
    if not root_path.is_dir():
        return list(MVTEC_CATEGORIES)

    discovered: list[str] = []
    try:
        for entry in root_path.iterdir():
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                discovered.append(entry.name)
    except OSError:
        return list(MVTEC_CATEGORIES)

    if discovered:
        return sorted(discovered)

    return list(MVTEC_CATEGORIES)
