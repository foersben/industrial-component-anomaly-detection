"""Hyperparameter sweep orchestrator for Keras CAE and PatchCore models.

This script manages Optuna optimization studies across all 15 MVTec AD categories.
Each study explores preprocessing transforms, model architectures, and scoring
parameters to maximize the target validation metric (Pixel AUPIMO / AUROC).

Architecture & Memory Safety:
    Running repeated deep learning training trials (TensorFlow and PyTorch) inside a
    single persistent Python process causes GPU VRAM fragmentation and memory leaks.
    This orchestrator executes each optimization study or trial in an isolated
    subprocess, ensuring that the operating system reclaims all GPU and system memory
    upon completion.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep")

# All 15 official MVTec AD benchmark categories
ALL_CATEGORIES: list[str] = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def sweep_keras(
    categories: list[str] | None = None,
    n_trials: int = 30,
    data_root: str = "data/raw/mvtec_ad",
) -> None:
    """Execute hyperparameter optimization for the Keras Convolutional Autoencoder.

    Optimizes bottleneck capacity (latent channels), preprocessing filters (CLAHE,
    Gaussian Blur), and foreground masking across selected categories.

    Args:
        categories: List of categories to optimize. Defaults to all 15 categories.
        n_trials: Total number of Optuna trials to run per category (default: 30).
        data_root: Path to the MVTec AD dataset root.
    """
    target_categories = categories or ALL_CATEGORIES
    output_path = Path("data/hyperparameters/keras_cae_best.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    pending = [cat for cat in target_categories if cat not in registry]

    if not pending:
        logger.info("All target categories are already completed for Keras CAE.")
        return

    logger.info("Resuming Keras CAE sweep for %d pending categories: %s", len(pending), pending)

    env = os.environ.copy()
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    for idx, category in enumerate(pending, start=1):
        logger.info(
            "\n%s\n[%d/%d] Starting Keras CAE Optuna Study: %s\n%s", "=" * 60, idx, len(pending), category, "=" * 60
        )

        cmd = [
            sys.executable,
            "-m",
            "app.pipelines.modelling.keras_cae.optuna_study",
            "--category",
            category,
            "--n-trials",
            str(n_trials),
            "--data-root",
            data_root,
        ]

        try:
            subprocess.run(cmd, env=env, check=True)
            logger.info("Successfully completed study for category '%s'.", category)
        except subprocess.CalledProcessError as err:
            logger.error("Study failed for category '%s' with exit code %d.", category, err.returncode)
            logger.info("Stopping sweep to allow inspection.")
            sys.exit(1)

    logger.info("Keras CAE sweep completed successfully.")


def sweep_patchcore(
    categories: list[str] | None = None,
    n_trials: int = 30,
    data_root: str = "data/raw/mvtec_ad",
) -> None:
    """Execute hyperparameter optimization for PatchCore.

    Tunes coreset subsampling ratio, intermediate feature layers (layer2, layer3, layer4),
    k-NN neighborhood size, and preprocessing pipelines.

    Args:
        categories: List of categories to optimize. Defaults to all 15 categories.
        n_trials: Total number of Optuna trials to run per category (default: 30).
        data_root: Path to the MVTec AD dataset root.
    """
    target_categories = categories or ALL_CATEGORIES
    output_path = Path("data/hyperparameters/patchcore_best.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    pending = [cat for cat in target_categories if cat not in registry]

    if not pending:
        logger.info("All target categories are already completed for PatchCore.")
        return

    logger.info("Resuming PatchCore sweep for %d pending categories: %s", len(pending), pending)

    env = os.environ.copy()

    for cat_idx, category in enumerate(pending, start=1):
        logger.info(
            "\n%s\n[%d/%d] Starting PatchCore Optuna Study: %s (%d trials)\n%s",
            "=" * 60,
            cat_idx,
            len(pending),
            category,
            n_trials,
            "=" * 60,
        )

        for trial_target in range(1, n_trials + 1):
            logger.info("Category '%s' -> Trial %d/%d", category, trial_target, n_trials)

            cmd = [
                sys.executable,
                "-m",
                "app.pipelines.modelling.patchcore_optuna_study",
                "--category",
                category,
                "--n-trials",
                str(trial_target),
                "--data-root",
                data_root,
            ]

            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as err:
                logger.error(
                    "Trial %d for category '%s' failed with exit code %d.",
                    trial_target,
                    category,
                    err.returncode,
                )
                sys.exit(1)

        logger.info("Successfully completed PatchCore study for category '%s'.", category)

    logger.info("PatchCore sweep completed successfully.")


def _print_summary() -> None:
    """Print a clean terminal summary of current best configurations for both models."""
    keras_path = Path("data/hyperparameters/keras_cae_best.json")
    patchcore_path = Path("data/hyperparameters/patchcore_best.json")

    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)

    if keras_path.exists():
        with open(keras_path, encoding="utf-8") as f:
            k_data = json.load(f)
        print(f"\n[Keras CAE] - {len(k_data)}/{len(ALL_CATEGORIES)} Categories Complete")
        for cat, cfg in sorted(k_data.items()):
            score = cfg.get("score", 0.0)
            metric = cfg.get("target_metric", "metric")
            print(f"  * {cat:<14} -> {metric}: {score:.4f}")
    else:
        print("\n[Keras CAE] No completed study registry found.")

    if patchcore_path.exists():
        with open(patchcore_path, encoding="utf-8") as f:
            p_data = json.load(f)
        print(f"\n[PatchCore] - {len(p_data)}/{len(ALL_CATEGORIES)} Categories Complete")
        for cat, cfg in sorted(p_data.items()):
            score = cfg.get("score", 0.0)
            metric = cfg.get("target_metric", "metric")
            print(f"  * {cat:<14} -> {metric}: {score:.4f}")
    else:
        print("\n[PatchCore] No completed study registry found.")

    print("=" * 70 + "\n")


def main() -> None:
    """Parse command-line arguments and dispatch hyperparameter optimization sweeps."""
    parser = argparse.ArgumentParser(
        description="Run automated Optuna hyperparameter sweeps for Keras CAE and PatchCore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["keras", "patchcore", "all"],
        required=True,
        help="Target model architecture to sweep: 'keras', 'patchcore', or 'all'.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna optimization trials per category.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of categories to run (e.g. --categories bottle cable). Defaults to all 15.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw/mvtec_ad",
        help="Path to the MVTec AD dataset directory.",
    )

    args = parser.parse_args()

    if args.model in ("keras", "all"):
        sweep_keras(categories=args.categories, n_trials=args.trials, data_root=args.data_root)

    if args.model in ("patchcore", "all"):
        sweep_patchcore(categories=args.categories, n_trials=args.trials, data_root=args.data_root)

    _print_summary()


if __name__ == "__main__":
    main()
