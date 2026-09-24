"""Compact cached pixel precision-recall curves without invalidating evaluations."""

import argparse
from pathlib import Path

from app.pipelines.evaluation.metrics import migrate_pixel_metrics_archive


def main() -> None:
    """Compact every pixel metrics archive below a model registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", nargs="?", type=Path, default=Path("data/models"))
    args = parser.parse_args()

    paths = sorted(
        path for path in args.registry.rglob("pixel_metrics.npz") if ".legacy_migration_backups" not in path.parts
    )
    original_bytes = 0
    compacted_bytes = 0
    for path in paths:
        json_path, before, after = migrate_pixel_metrics_archive(path)
        original_bytes += before
        compacted_bytes += after
        print(f"{path} -> {json_path}: {before:,} -> {after:,} bytes")

    print(
        f"Compacted {len(paths)} archives: {original_bytes:,} -> {compacted_bytes:,} bytes "
        f"(reclaimed {original_bytes - compacted_bytes:,} bytes)"
    )


if __name__ == "__main__":
    main()
