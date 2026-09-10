import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from app.domain.categories import MVTEC_CATEGORIES

CATEGORIES = list(MVTEC_CATEGORIES)


def load_and_prepare_evaluation_data(metrics_path: Any) -> Any:
    """Load and prepare evaluation data."""
    from app.pipelines.evaluation.visualization import load_and_prepare_evaluation_data

    return load_and_prepare_evaluation_data(metrics_path)


def plot_tradeoff_curve(data: Any) -> Any:
    """Plot the tradeoff curve."""
    from app.pipelines.evaluation.visualization import plot_tradeoff_curve

    return plot_tradeoff_curve(data)


def plot_pr_curve(data: Any) -> Any:
    """Plot the precision-recall curve."""
    from app.pipelines.evaluation.visualization import plot_pr_curve

    return plot_pr_curve(data)


def _extract_preprocessing_steps_from_config(prep: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert boolean preprocessing flags from config to standard step list.

    Args:
        prep: Preprocessing flag dictionary from hyperparameter JSON.

    Returns:
        List of preprocessing step dictionaries.
    """
    steps: list[dict[str, Any]] = []
    if prep.get("use_foreground_mask", False):
        steps.append({"name": "foreground_mask", "params": {}})
    if prep.get("use_clahe", False):
        steps.append({"name": "clahe", "params": {}})
    if prep.get("use_gaussian_blur", False):
        steps.append({"name": "gaussian_blur", "params": {}})
    return steps


def _save_evaluation_curves(results: Any, out_dir: Path) -> None:
    """Generate and save tradeoff and precision-recall curves for image and pixel levels.

    Args:
        results: Evaluation results dictionary containing level metric metadata.
        out_dir: Target output directory for PNG plots.
    """
    for level in ["image_level", "pixel_level"]:
        metrics = results.get(level, {})
        metrics_path = metrics.get("metrics_path")
        if not (metrics_path and Path(metrics_path).exists()):
            continue
        try:
            data = load_and_prepare_evaluation_data(metrics_path)

            fig_tradeoff = plot_tradeoff_curve(data)
            fig_tradeoff.savefig(out_dir / f"{level}_tradeoff_curve.png")
            plt.close(fig_tradeoff)

            fig_pr = plot_pr_curve(data)
            fig_pr.savefig(out_dir / f"{level}_pr_curve.png")
            plt.close(fig_pr)
        except Exception as e:
            print(f"Error generating plots for {level}: {e}")


def _save_metrics_summary(results: Any, out_dir: Path) -> None:
    """Extract scalar evaluation metrics and persist summary CSV.

    Args:
        results: Evaluation results dictionary.
        out_dir: Target output directory for metrics_summary.csv.
    """
    summary: dict[str, Any] = {}
    for level in ["image_level", "pixel_level"]:
        level_metrics = results.get(level, {})
        for k, v in level_metrics.items():
            if isinstance(v, (int, float, str)) and k != "metrics_path":
                summary[f"{level}_{k}"] = v

    if "auroc" in results:
        summary["auroc"] = results.get("auroc", 0.0)
        summary["aupimo"] = results.get("aupimo", 0.0)
        summary["accuracy"] = results.get("accuracy", 0.0)
        summary["precision"] = results.get("precision", 0.0)
        summary["recall"] = results.get("recall", 0.0)
        summary["threshold"] = results.get("threshold", 0.0)

    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)


def _save_heatmap_images(heatmap_overlays: dict[Any, Any], heatmaps_pred_dir: Path, heatmaps_gt_dir: Path) -> None:
    """Save prediction and ground-truth overlay heatmap PNG images.

    Args:
        heatmap_overlays: Dictionary mapping image index to overlay arrays.
        heatmaps_pred_dir: Directory to save prediction heatmap PNGs.
        heatmaps_gt_dir: Directory to save ground-truth overlay PNGs.
    """
    for idx, overlay_data in heatmap_overlays.items():
        if not isinstance(overlay_data, dict):
            continue
        if "heatmap" in overlay_data:
            hm_arr = np.array(overlay_data["heatmap"], dtype=np.uint8)
            Image.fromarray(hm_arr).save(heatmaps_pred_dir / f"image_{idx}_prediction.png")

        if "gt_and_heatmap" in overlay_data:
            gt_hm_arr = np.array(overlay_data["gt_and_heatmap"], dtype=np.uint8)
            Image.fromarray(gt_hm_arr).save(heatmaps_gt_dir / f"image_{idx}_gt_overlay.png")


def save_plots_and_heatmaps(results: Any, out_dir: Any, heatmaps_pred_dir: Any, heatmaps_gt_dir: Any) -> None:
    """Save evaluation plots, metric summaries, and overlay heatmaps to disk.

    Args:
        results: Evaluation results dictionary.
        out_dir: Output directory path.
        heatmaps_pred_dir: Prediction heatmaps output directory.
        heatmaps_gt_dir: Ground-truth overlay output directory.
    """
    out_path = Path(out_dir)
    pred_path = Path(heatmaps_pred_dir)
    gt_path = Path(heatmaps_gt_dir)
    _save_evaluation_curves(results, out_path)
    _save_metrics_summary(results, out_path)
    _save_heatmap_images(results.get("heatmap_overlays", {}), pred_path, gt_path)


def _load_tuned_keras_hyperparams(category: str) -> tuple[int, list[dict[str, Any]]]:
    """Load tuned hyperparameters and preprocessing steps for Keras CAE.

    Args:
        category: Component category name.

    Returns:
        Tuple of (latent_channels, preprocessing_steps).
    """
    json_path = Path("data/hyperparameters/keras_cae_best.json")
    if not json_path.exists():
        return 32, []
    try:
        with open(json_path, encoding="utf-8") as f:
            hyperparams = json.load(f)
        config = hyperparams.get(category, {})
        prep = config.get("preprocessing", {})
        hp = config.get("model_hyperparameters", {})
        latent_channels = hp.get("latent_dim", hp.get("latent_channels", 32))
        return latent_channels, _extract_preprocessing_steps_from_config(prep)
    except Exception:
        return 32, []


def _save_keras_loss_curve(loss_history: Any, out_dir: Path, category: str) -> None:
    """Persist loss history table and visualization curve plot.

    Args:
        loss_history: Dictionary containing loss lists per split.
        out_dir: Directory where artifacts are saved.
        category: Component category name.
    """
    if not (loss_history and isinstance(loss_history, dict)):
        return
    clean_history = {k: v for k, v in loss_history.items() if isinstance(v, list) and len(v) > 0}
    if not clean_history:
        return
    df_loss = pd.DataFrame({k: pd.Series(v) for k, v in clean_history.items()})
    df_loss.to_csv(out_dir / "loss_table.csv", index=False)

    plt.figure(figsize=(10, 5))
    for col in df_loss.columns:
        plt.plot(df_loss[col].to_numpy(), label=col)
    plt.title(f"Training Loss Curve - {category}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "loss_diagram.png")
    plt.close()


def evaluate_keras(category: str, tuned: bool) -> None:
    """Evaluate the Keras CAE model.

    Args:
        category: MVTec category name to evaluate.
        tuned: Whether to use tuned hyperparameters.
    """
    from app.core.tf_device import configure_tensorflow

    configure_tensorflow(min_vram_mib=1024)
    import tensorflow as tf

    from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline

    print(
        f"\n{'=' * 50}\nEvaluating Keras CAE ({'Tuned' if tuned else 'Baseline'}) for category: {category}\n{'=' * 50}"
    )

    base_out_dir = Path("results/evaluation") / ("keras_cae" if tuned else "keras_cae_baseline")
    out_dir = base_out_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmaps_pred_dir = out_dir / "heatmaps" / "prediction"
    heatmaps_gt_dir = out_dir / "heatmaps" / "ground_truth_overlay"
    heatmaps_pred_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_gt_dir.mkdir(parents=True, exist_ok=True)

    latent_channels = 32
    epochs = 100
    preprocessing_steps: list[dict[str, Any]] = []

    if tuned:
        latent_channels, preprocessing_steps = _load_tuned_keras_hyperparams(category)

    results = run_keras_cae_pipeline(
        data_root="data/raw/mvtec_ad",
        category=category,
        img_size=128,
        crop_size=64,
        crop_stride=32,
        latent_channels=latent_channels,
        epochs=epochs,
        batch_size=16,
        mask_ratio=0.25,
        threshold_method="quantile",
        k_fraction=0.002,
        pipeline=preprocessing_steps,
        run_heatmap=True,
        force_retrain=True,
        model_hash=None,
    )

    _save_keras_loss_curve(results.get("loss_history"), out_dir, category)
    save_plots_and_heatmaps(results, out_dir, heatmaps_pred_dir, heatmaps_gt_dir)
    print(f"Completed evaluation for {category}. Outputs saved to {out_dir}")
    tf.keras.backend.clear_session()
    gc.collect()


def _load_tuned_patchcore_hyperparams(
    category: str,
) -> tuple[list[dict[str, Any]], str, tuple[str, ...], float, int]:
    """Load tuned hyperparameters and preprocessing steps for PatchCore.

    Args:
        category: Component category name.

    Returns:
        Tuple of (preprocessing_steps, backbone, feature_layers, coreset_ratio, num_neighbors).
    """
    json_path = Path("data/hyperparameters/patchcore_best.json")
    if not json_path.exists():
        return [], "resnet18", ("layer2", "layer3"), 0.1, 9
    try:
        with open(json_path, encoding="utf-8") as f:
            hyperparams = json.load(f)
        config = hyperparams.get(category, {})
        prep = config.get("preprocessing", {})
        hp = config.get("model_hyperparameters", {})

        steps = _extract_preprocessing_steps_from_config(prep)
        feature_layers_str = hp.get("feature_layers", "l2_l3")
        feature_layers = ("layer2", "layer3") if feature_layers_str == "l2_l3" else ("layer2", "layer3", "layer4")
        backbone = hp.get("backbone", "resnet18")
        coreset_ratio = hp.get("coreset_sampling_ratio", 0.1)
        num_neighbors = hp.get("num_neighbors", 9)
        return steps, backbone, feature_layers, coreset_ratio, num_neighbors
    except Exception:
        return [], "resnet18", ("layer2", "layer3"), 0.1, 9


def evaluate_patchcore(category: str, tuned: bool) -> None:
    """Evaluate the PatchCore model.

    Args:
        category: MVTec category name to evaluate.
        tuned: Whether to use tuned hyperparameters.
    """
    import torch

    from app.pipelines.modelling.patchcore import run_patchcore_pipeline

    print(
        f"\n{'=' * 50}\nEvaluating Patchcore ({'Tuned' if tuned else 'Baseline'}) for category: {category}\n{'=' * 50}"
    )

    base_out_dir = Path("results/evaluation") / ("patchcore_tuned" if tuned else "patchcore")
    out_dir = base_out_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmaps_pred_dir = out_dir / "heatmaps" / "prediction"
    heatmaps_gt_dir = out_dir / "heatmaps" / "ground_truth_overlay"
    heatmaps_pred_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_gt_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_steps: list[dict[str, Any]] = []
    backbone = "resnet18"
    feature_layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_ratio = 0.1
    num_neighbors = 9

    if tuned:
        preprocessing_steps, backbone, feature_layers, coreset_ratio, num_neighbors = _load_tuned_patchcore_hyperparams(
            category
        )

    results = run_patchcore_pipeline(
        data_root=Path("data/raw/mvtec_ad"),
        category=category,
        pipeline=preprocessing_steps,
        fpr_limit=1e-4,
        backbone=backbone,
        feature_layers=feature_layers,
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=num_neighbors,
        run_heatmap=True,
        force_retrain=True,
        model_hash=None,
    )

    pixel_metrics = results["pixel_level"]
    print(f"PatchCore {category} AUPIMO: {pixel_metrics['aupimo']:.6f}")
    print(
        "PatchCore anomaly maps: "
        f"min={pixel_metrics['anomaly_map_min']:.8f}, "
        f"max={pixel_metrics['anomaly_map_max']:.8f}, "
        f"range={pixel_metrics['anomaly_map_range']:.8f}"
    )

    save_plots_and_heatmaps(results, out_dir, heatmaps_pred_dir, heatmaps_gt_dir)
    print(f"Completed evaluation for {category}. Outputs saved to {out_dir}")
    torch.cuda.empty_cache()
    gc.collect()


def orchestrator(model: str, tuned: bool) -> None:
    """Orchestrate evaluation across categories."""
    base_out_dir = Path("results/evaluation")
    if model == "keras":
        out_dir = base_out_dir / ("keras_cae" if tuned else "keras_cae_baseline")
    else:
        out_dir = base_out_dir / ("patchcore_tuned" if tuned else "patchcore")

    categories_to_run = []

    if tuned:
        json_path = Path(
            f"data/hyperparameters/{model}_cae_best.json"
            if model == "keras"
            else "data/hyperparameters/patchcore_best.json"
        )
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return
        with open(json_path, encoding="utf-8") as f:
            hyperparams = json.load(f)
        categories_to_run = list(hyperparams.keys())
    else:
        categories_to_run = CATEGORIES

    for category in categories_to_run:
        if (out_dir / category / "metrics_summary.csv").exists():
            print(f"Skipping '{category}' as metrics_summary.csv already exists (evaluated).")
            continue

        print(f"\n[ORCHESTRATOR] Spawning isolated process for '{category}'...")
        cmd = [sys.executable, __file__, "--model", model, "--category", category]
        if tuned:
            cmd.append("--tuned")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ORCHESTRATOR] Process for '{category}' failed with exit code {e.returncode}.")
            sys.exit(e.returncode)


def main() -> None:
    """Run the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["keras", "patchcore", "baseline"],
        required=True,
        help="Model to evaluate (baseline is an alias for patchcore)",
    )
    parser.add_argument("--tuned", action="store_true", help="Evaluate the tuned hyperparameters")
    parser.add_argument("--category", type=str, help="Specific category to evaluate (used internally for isolation)")
    args = parser.parse_args()

    model = "patchcore" if args.model == "baseline" else args.model

    if args.category:
        if model == "keras":
            evaluate_keras(args.category, args.tuned)
        elif model == "patchcore":
            evaluate_patchcore(args.category, args.tuned)
    else:
        orchestrator(model, args.tuned)


if __name__ == "__main__":
    main()
