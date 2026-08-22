import json
import logging
from pathlib import Path
from typing import Any

import optuna

from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline

logger = logging.getLogger(__name__)

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}


def objective(trial: optuna.Trial, category_name: str, data_root: str = "data/raw/mvtec_ad") -> float:
    """Optuna objective function for optimizing Keras CAE hyperparameters.

    Args:
        trial: Optuna trial object.
        category_name: MVTec category name to optimize.
        data_root: Path to the MVTec AD dataset.

    Returns:
        Pixel AUPIMO score to maximize.
    """
    # 1. Hyperparameter Search Space
    latent_channels = trial.suggest_categorical("latent_channels", [16, 32, 64, 128])
    apply_clahe = trial.suggest_categorical("apply_clahe", [True, False])
    apply_blur = trial.suggest_categorical("apply_blur", [True, False])

    blur_ksize = 5
    if apply_blur:
        blur_ksize = trial.suggest_categorical("blur_ksize", [3, 5, 7])

    # Type-Specific Logic for MVTec categories
    if category_name in TEXTURE_CATEGORIES:
        apply_foreground_mask = False
    else:
        apply_foreground_mask = trial.suggest_categorical("apply_foreground_mask", [True, False])

    # Build preprocessing steps
    preprocessing_steps = []
    if apply_foreground_mask:
        preprocessing_steps.append({"name": "foreground_mask", "params": {}})
    if apply_clahe:
        preprocessing_steps.append({"name": "clahe", "params": {}})
    if apply_blur:
        preprocessing_steps.append({"name": "gaussian_blur", "params": {"ksize": blur_ksize}})

    # We use fewer epochs and a smaller batch size to quickly prune bad trials
    epochs = 20
    batch_size = 16

    try:
        results = run_keras_cae_pipeline(
            data_root=data_root,
            category=category_name,
            latent_channels=latent_channels,
            preprocessing_steps=preprocessing_steps,
            epochs=epochs,
            batch_size=batch_size,
            force_retrain=True,  # Force retrain so it explores the space
            trial=trial,  # Pass trial down to trigger native pruning
        )
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error("Trial failed during execution: %s", e)
        raise optuna.exceptions.TrialPruned() from e

    # Extract the target metric to maximize
    pixel_aupimo = results.get("metrics", {}).get("pixel_aupimo")

    if pixel_aupimo is None:
        raise ValueError("Pixel AUPIMO score not found in results.")

    return float(pixel_aupimo)


def run_study(category_name: str, n_trials: int = 15, data_root: str = "data/raw/mvtec_ad") -> dict[str, Any]:
    """Run the Optuna study and save the best parameters."""
    study_name = f"keras_cae_{category_name}"

    # We want to MAXIMIZE Pixel AUPIMO
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    )

    study.optimize(lambda trial: objective(trial, category_name, data_root), n_trials=n_trials)

    best_params = study.best_params

    # Map to nested schema
    is_texture = category_name in TEXTURE_CATEGORIES
    use_mask = False if is_texture else best_params.get("apply_foreground_mask", False)

    cfg = {
        "target_metric": "pixel_aupimo",
        "score": study.best_value,
        "preprocessing": {
            "use_foreground_mask": use_mask,
            "use_clahe": best_params.get("apply_clahe", False),
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8],
            "use_gaussian_blur": best_params.get("apply_blur", False),
            "blur_ksize": best_params.get("blur_ksize", 5),
        },
        "model_hyperparameters": {
            "learning_rate": 0.001,
            "latent_dim": best_params.get("latent_channels", 32),
            "loss_weight_ssim": 0.84,
            "loss_weight_mse": 0.16,
        },
    }

    # Save to JSON registry
    registry_path = Path("data/hyperparameters/keras_cae_best.json")
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[category_name] = cfg

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)

    logger.info("Study finished. Best Pixel AUPIMO: %.4f", study.best_value)
    logger.info("Best Params: %s", cfg)

    return cfg


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Optuna study for Keras CAE")
    parser.add_argument("--category", type=str, required=True, help="MVTec category name")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--data-root", type=str, default="data/raw/mvtec_ad", help="Dataset root directory")

    args = parser.parse_args()
    run_study(args.category, n_trials=args.n_trials, data_root=args.data_root)
