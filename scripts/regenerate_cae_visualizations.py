"""Generate missing Keras CAE heatmap archives and four-panel image galleries."""

import gc
import json
from pathlib import Path
from typing import Any

from app.pipelines.modelling.keras_cae import run_keras_cae_pipeline


def _visualizations_complete(model_dir: Path, metadata: dict[str, Any]) -> bool:
    """Return whether both persisted CAE visualization formats are available."""
    heatmap_path = metadata.get("heatmap_overlays_path")
    panel_path = metadata.get("four_panel_images_path")
    if not isinstance(heatmap_path, str) or not (model_dir / heatmap_path).is_file():
        return False
    if not isinstance(panel_path, str):
        return False
    panel_dir = model_dir / panel_path
    return panel_dir.is_dir() and any(panel_dir.glob("*.png"))


def main() -> None:
    """Reevaluate existing weights only when visualization artifacts are incomplete."""
    registry = Path("data/models/keras_cae")
    targets: list[tuple[Path, dict[str, Any]]] = []
    for model_dir in sorted(registry.iterdir()):
        if not model_dir.is_dir() or model_dir.name == ".trash":
            continue
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        if not _visualizations_complete(model_dir, metadata):
            targets.append((model_dir, metadata))

    print(f"Generating missing visualizations for {len(targets)} cached model(s).", flush=True)
    skipped: list[tuple[str, str]] = []
    for index, (model_dir, metadata) in enumerate(targets, start=1):
        category = str(metadata["category"])
        print(f"[{index}/{len(targets)}] {model_dir.name} ({category})", flush=True)
        try:
            run_keras_cae_pipeline(
                data_root="data/raw/mvtec_ad",
                category=category,
                model_hash=model_dir.name,
                run_heatmap=True,
                reevaluate_cached=True,
                force_retrain=False,
            )
        except FileNotFoundError as exc:
            skipped.append((model_dir.name, str(exc)))
            print(f"  SKIPPED: {exc}", flush=True)
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            pass
        gc.collect()

    if skipped:
        print(f"Skipped {len(skipped)} incompatible cache(s):", flush=True)
        for model_hash, reason in skipped:
            print(f"  {model_hash}: {reason}", flush=True)


if __name__ == "__main__":
    main()
