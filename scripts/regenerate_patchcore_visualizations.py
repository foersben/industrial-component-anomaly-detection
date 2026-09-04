"""Regenerate PatchCore four-panel images from a fitted checkpoint.

This command performs validation and test inference only. It never calls model fitting.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from app.domain.data import build_fair_evaluation_split, build_mvtec_manifest
from app.pipelines.evaluation.scoring import compute_adaptive_threshold
from app.pipelines.modelling.baseline import (
    PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    _configure_patchcore_partitions,
    _RawScoreImageVisualizer,
    _tensor_to_numpy,
)
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.factory import build_pipeline_from_configs


def _load_preprocessing(metadata_path: Path | None, category: str) -> PreprocessingTransformAdapter | None:
    """Recreate any preprocessing recorded for the checkpoint."""
    if metadata_path is None:
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("category") != category:
        raise ValueError(
            f"Metadata category {metadata.get('category')!r} does not match requested category {category!r}"
        )
    pipeline = build_pipeline_from_configs(metadata.get("preprocessing_steps", []))
    return PreprocessingTransformAdapter(pipeline) if len(pipeline) > 0 else None


def regenerate_visualizations(
    category: str,
    checkpoint: Path,
    output_dir: Path,
    data_root: Path,
    metadata_path: Path | None = None,
    accelerator: str = "gpu",
    devices: int = 1,
) -> float:
    """Run inference from a fitted checkpoint and render calibrated predicted masks."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    manifest = build_mvtec_manifest(data_root)
    fair_split = build_fair_evaluation_split(manifest, category)
    transform_adapter = _load_preprocessing(metadata_path, category)

    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        val_split_mode="none",
    )
    _configure_patchcore_partitions(datamodule, fair_split, transform_adapter)

    visualizer = _RawScoreImageVisualizer(output_dir=output_dir)
    visualizer.render_enabled = False
    model = Patchcore.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        pre_trained=False,
        visualizer=visualizer,
    )
    engine = Engine(accelerator=accelerator, devices=devices, deterministic=True)

    validation_predictions = engine.predict(model=model, dataloaders=datamodule.val_dataloader())
    validation_pixel_scores: list[np.ndarray[Any, Any]] = []
    for batch in validation_predictions or []:
        maps = _tensor_to_numpy(getattr(batch, "anomaly_map", None))
        if maps is not None:
            validation_pixel_scores.append(maps)
    if not validation_pixel_scores:
        raise RuntimeError("Validation inference produced no anomaly maps")

    pixel_threshold = compute_adaptive_threshold(
        np.concatenate(validation_pixel_scores),
        method="quantile",
        quantile=PATCHCORE_PIXEL_THRESHOLD_QUANTILE,
    )
    visualizer.pixel_threshold = pixel_threshold
    visualizer.render_enabled = True
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.predict(model=model, dataloaders=datamodule.test_dataloader())

    threshold_path = output_dir / "pixel_threshold.txt"
    threshold_path.write_text(f"{pixel_threshold:.17g}\n", encoding="utf-8")
    return float(pixel_threshold)


def main() -> None:
    """Parse arguments and regenerate images without fitting the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="MVTec category, for example 'zipper'")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Saved PatchCore model.ckpt")
    parser.add_argument("--output-dir", required=True, type=Path, help="Separate directory for regenerated images")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--metadata", type=Path, help="Optional matching registry metadata.json")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "auto"])
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    threshold = regenerate_visualizations(
        category=args.category,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
        metadata_path=args.metadata,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    print(f"Regenerated {args.category} visualizations in {args.output_dir}")
    print(f"Normal-validation pixel threshold: {threshold:.8f}")


if __name__ == "__main__":
    main()
