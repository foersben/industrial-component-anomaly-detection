import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORIES = [
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


def sweep_keras() -> None:
    """Run the Keras CAE Optuna sweep."""
    keras_output = Path("data/hyperparameters/keras_cae_best.json")
    if keras_output.exists():
        with open(keras_output, encoding="utf-8") as f:
            keras_results = json.load(f)
    else:
        keras_results = {}

    remaining_categories = [c for c in CATEGORIES if c not in keras_results]

    if not remaining_categories:
        logger.info("All categories are already completed for Keras CAE!")
        return

    logger.info(f"Resuming sweep for remaining {len(remaining_categories)} categories: {remaining_categories}")

    env = os.environ.copy()
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    wrapper_code = """
import sys
from app.core.tf_device import preload_cuda_shared_libraries, configure_tensorflow
# 1. Force load the Pixi environment CUDA/cuDNN libraries into the global symbol table
preload_cuda_shared_libraries()
# 2. Configure TF (this uses the preloaded libraries)
configure_tensorflow()
# 3. Now we can safely run the Optuna study module
import runpy
sys.argv = {sys_argv}
runpy.run_module('app.pipelines.modelling.keras_cae.optuna_study', run_name='__main__')
"""

    for category in remaining_categories:
        logger.info(f"\n{'=' * 50}\nRunning Optuna study for: {category} in an isolated subprocess\n{'=' * 50}\n")

        args = ["--category", category, "--n-trials", "30"]
        sys_argv_str = str(["app.pipelines.modelling.keras_cae.optuna_study", *args])

        script = wrapper_code.replace("{sys_argv}", sys_argv_str)

        cmd = [sys.executable, "-c", script]

        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: Category {category} failed with return code {e.returncode}.")
            logger.info("Stopping the sweep to prevent further errors.")
            sys.exit(1)

    logger.info("\nKeras CAE Sweep complete!")


def sweep_patchcore() -> None:
    """Run the PatchCore Optuna sweep."""
    out_path = Path("data/hyperparameters/patchcore_best.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    remaining_categories = [c for c in CATEGORIES if c not in registry]

    if not remaining_categories:
        logger.info("All categories are already completed for PatchCore!")
        return

    logger.info(f"Resuming PatchCore sweep for remaining categories: {remaining_categories}")

    env = os.environ.copy()

    for cat in remaining_categories:
        logger.info(f"=== Starting Patchcore Optuna Sweep for category: {cat} ===")
        # Run 30 trials for this category, one by one in isolated subprocesses
        for trial_idx in range(30):
            logger.info(f"--- {cat} - Trial {trial_idx + 1}/30 ---")

            cmd = [
                sys.executable,
                "-m",
                "app.pipelines.modelling.patchcore_optuna_study",
                "--category",
                cat,
                "--n-trials",
                "1",
            ]

            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error: Trial {trial_idx + 1} for category {cat} failed with return code {e.returncode}.")
                sys.exit(1)

    logger.info("Finished Patchcore Optuna sweep for all categories!")


def main() -> None:
    """Run the sweep script."""
    parser = argparse.ArgumentParser(description="Run optuna sweeps for models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["keras", "patchcore"],
        required=True,
        help="Model to sweep (keras or patchcore)",
    )
    args = parser.parse_args()

    if args.model == "keras":
        sweep_keras()
    elif args.model == "patchcore":
        sweep_patchcore()


if __name__ == "__main__":
    main()
