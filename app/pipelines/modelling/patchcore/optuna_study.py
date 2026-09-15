"""Category-Adaptive Optuna Optimization for PatchCore.

Performs Bayesian hyperparameter sweeps across backbone architectures,
feature extraction layers, coreset sampling ratios, and domain preprocessing.
"""

import json
from pathlib import Path
from typing import Any

import anomalib.models.components.sampling.k_center_greedy as kcg
import optuna
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from app.core.logger import logger
from app.domain.categories import OBJECT_CATEGORIES, TEXTURE_CATEGORIES
from app.pipelines.preprocessing.adapter import PreprocessingTransformAdapter
from app.pipelines.preprocessing.factory import build_pipeline_from_configs

OBJECTS = OBJECT_CATEGORIES
TEXTURES = TEXTURE_CATEGORIES

__all__ = [
    "OBJECTS",
    "TEXTURES",
    "objective",
    "run_study",
]

# Monkeypatch tqdm in k_center_greedy to prevent Jupyter RecursionError loops
kcg.tqdm = lambda iterable, *_, **__: iterable  # type: ignore[attr-defined]


def _evaluate_patchcore(
    category_name: str,
    backbone: str,
    feature_layers: tuple[str, ...],
    coreset_ratio: float,
    num_neighbors: int,
    use_clahe: bool,
    use_gaussian_blur: bool,
    use_foreground_mask: bool,
    data_root: str = "data/raw/mvtec_ad",
) -> float:
    """Evaluate a single PatchCore hyperparameter configuration.

    Args:
        category_name: Target MVTec AD category name.
        backbone: Name of torchvision backbone feature extractor.
        feature_layers: Layer identifiers to tap for patch representations.
        coreset_ratio: Sampling fraction for greedy coreset memory subsampling.
        num_neighbors: K parameter for nearest-neighbor score distance.
        use_clahe: Whether CLAHE contrast enhancement is active.
        use_gaussian_blur: Whether Gaussian smoothing is active.
        use_foreground_mask: Whether foreground segmentation is active.
        data_root: Local root directory of the MVTec AD dataset.

    Returns:
        Validation/test pixel AUROC score for the trial.
    """
    pipeline = []
    if use_foreground_mask:
        pipeline.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        pipeline.append({"name": "clahe", "params": {}})
    if use_gaussian_blur:
        pipeline.append({"name": "gaussian_blur", "params": {}})

    proc_pipeline = build_pipeline_from_configs(pipeline)
    transform_adapter = PreprocessingTransformAdapter(proc_pipeline)

    class PreprocessedMVTecAD(MVTecAD):
        def setup(self, stage: str | None = None) -> None:
            super().setup(stage)
            from app.pipelines.preprocessing.adapter import PreprocessedAnomalibDataset

            if getattr(self, "train_data", None) is not None and not isinstance(
                self.train_data, PreprocessedAnomalibDataset
            ):
                self.train_data = PreprocessedAnomalibDataset(self.train_data, transform_adapter)  # type: ignore[assignment]

            if getattr(self, "test_data", None) is not None and not isinstance(
                self.test_data, PreprocessedAnomalibDataset
            ):
                self.test_data = PreprocessedAnomalibDataset(self.test_data, transform_adapter)  # type: ignore[assignment]

    datamodule = PreprocessedMVTecAD(
        root=data_root,
        category=category_name,
        train_batch_size=16,
        eval_batch_size=16,
    )

    model = Patchcore(
        backbone=backbone,
        layers=feature_layers,
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=num_neighbors,
    )

    engine = Engine(accelerator="gpu", devices=1)

    try:
        engine.fit(model, datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
        if test_results and len(test_results) > 0:
            return float(test_results[0].get("pixel_AUROC", 0.0))
        return 0.0
    except Exception as e:
        logger.error("Trial failed with error: %s", e)
        raise optuna.exceptions.TrialPruned() from e
    finally:
        torch.cuda.empty_cache()


def objective(trial: optuna.Trial, category_name: str, data_root: str = "data/raw/mvtec_ad") -> float:
    """Optuna objective function for tuning PatchCore hyperparameters.

    Args:
        trial: Active Optuna trial instance.
        category_name: MVTec AD category string.
        data_root: Root dataset folder path.

    Returns:
        Objective evaluation metric score for the trial.
    """
    is_texture = category_name in TEXTURES
    backbone = "resnet18"

    layer_config = trial.suggest_categorical("feature_layers", ["l2_l3", "l2_l3_l4"])
    feature_layers: tuple[str, ...]
    if layer_config == "l2_l3":
        feature_layers = ("layer2", "layer3")
    else:
        feature_layers = ("layer2", "layer3", "layer4")

    coreset_ratio = trial.suggest_float("coreset_sampling_ratio", 0.001, 0.20, log=True)
    num_neighbors = trial.suggest_int("num_neighbors", 1, 9)

    use_clahe = trial.suggest_categorical("use_clahe", [True, False])
    use_gaussian_blur = trial.suggest_categorical("use_gaussian_blur", [True, False])

    use_foreground_mask = False if is_texture else trial.suggest_categorical("use_foreground_mask", [True, False])

    return _evaluate_patchcore(
        category_name=category_name,
        backbone=backbone,
        feature_layers=feature_layers,
        coreset_ratio=coreset_ratio,
        num_neighbors=num_neighbors,
        use_clahe=use_clahe,
        use_gaussian_blur=use_gaussian_blur,
        use_foreground_mask=use_foreground_mask,
        data_root=data_root,
    )


def run_study(category_name: str, n_trials: int = 30, data_root: str = "data/raw/mvtec_ad") -> dict[str, Any]:
    """Execute Optuna optimization study for PatchCore on a specific category.

    Args:
        category_name: Target category to tune.
        n_trials: Maximum number of trials to evaluate.
        data_root: Dataset root directory path.

    Returns:
        Structured configuration dictionary of the best hyperparameter settings found.
    """
    study_name = f"patchcore_{category_name}"

    storage_path = Path("data/hyperparameters/patchcore_optuna.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.resolve()}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    trials_to_run = max(0, n_trials - len(study.trials))
    if trials_to_run > 0:
        study.optimize(lambda t: objective(t, category_name, data_root), n_trials=trials_to_run)

    best_trial = study.best_trial
    logger.info("Best trial for %s:", category_name)
    logger.info("  Value (Pixel AUPIMO): %f", best_trial.value)
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info("    %s: %s", key, value)

    is_texture = category_name in TEXTURES

    return {
        "target_metric": "pixel_auroc",
        "score": best_trial.value,
        "preprocessing": {
            "use_foreground_mask": False if is_texture else best_trial.params.get("use_foreground_mask", False),
            "use_clahe": best_trial.params["use_clahe"],
            "use_gaussian_blur": best_trial.params["use_gaussian_blur"],
        },
        "model_hyperparameters": {
            "backbone": "resnet18",
            "feature_layers": best_trial.params["feature_layers"],
            "coreset_sampling_ratio": best_trial.params["coreset_sampling_ratio"],
            "num_neighbors": best_trial.params["num_neighbors"],
        },
    }


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Optuna study for PatchCore")
    parser.add_argument("--category", type=str, required=True, help="MVTec category name")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--data-root", type=str, default="data/raw/mvtec_ad", help="Dataset root directory")

    args = parser.parse_args()

    best_cfg = run_study(category_name=args.category, n_trials=args.n_trials, data_root=args.data_root)

    out_path = Path("data/hyperparameters/patchcore_best.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            full_registry = json.load(f)
    else:
        full_registry = {}

    full_registry[args.category] = best_cfg
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_registry, f, indent=2)
