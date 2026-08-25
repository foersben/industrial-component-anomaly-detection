import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def load_and_prepare_evaluation_data(metrics_path):
    from app.pipelines.evaluation.visualization import load_and_prepare_evaluation_data
    return load_and_prepare_evaluation_data(metrics_path)

def plot_tradeoff_curve(data):
    from app.pipelines.evaluation.visualization import plot_tradeoff_curve
    return plot_tradeoff_curve(data)

def plot_pr_curve(data):
    from app.pipelines.evaluation.visualization import plot_pr_curve
    return plot_pr_curve(data)

def save_plots_and_heatmaps(results, out_dir, heatmaps_pred_dir, heatmaps_gt_dir):
    for level in ["image_level", "pixel_level"]:
        metrics = results.get(level, {})
        metrics_path = metrics.get("metrics_path")
        if metrics_path and Path(metrics_path).exists():
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
    
    summary = {}
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

    heatmap_overlays = results.get("heatmap_overlays", {})
    for idx, overlay_data in heatmap_overlays.items():
        if isinstance(overlay_data, dict):
            if "heatmap" in overlay_data:
                hm_arr = np.array(overlay_data["heatmap"], dtype=np.uint8)
                Image.fromarray(hm_arr).save(heatmaps_pred_dir / f"image_{idx}_prediction.png")
            
            if "gt_and_heatmap" in overlay_data:
                gt_hm_arr = np.array(overlay_data["gt_and_heatmap"], dtype=np.uint8)
                Image.fromarray(gt_hm_arr).save(heatmaps_gt_dir / f"image_{idx}_gt_overlay.png")

def evaluate_keras(category, tuned):
    from app.core.tf_device import configure_tensorflow
    configure_tensorflow(min_vram_mib=1024)
    import tensorflow as tf
    from app.pipelines.modelling.keras_cae.cae_pipeline import run_keras_cae_pipeline

    print(f"\n{'='*50}\nEvaluating Keras CAE ({'Tuned' if tuned else 'Baseline'}) for category: {category}\n{'='*50}")
    
    base_out_dir = Path("results/evaluation") / ("keras_cae" if tuned else "keras_cae_baseline")
    out_dir = base_out_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    
    heatmaps_pred_dir = out_dir / "heatmaps" / "prediction"
    heatmaps_gt_dir = out_dir / "heatmaps" / "ground_truth_overlay"
    heatmaps_pred_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_gt_dir.mkdir(parents=True, exist_ok=True)

    latent_channels = 32
    epochs = 100
    preprocessing_steps = []

    if tuned:
        json_path = Path("data/hyperparameters/keras_cae_best.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                hyperparams = json.load(f)
            config = hyperparams.get(category, {})
            prep = config.get("preprocessing", {})
            hp = config.get("model_hyperparameters", {})
            latent_channels = hp.get("latent_dim", hp.get("latent_channels", 32))
            
            if prep.get("use_foreground_mask", False):
                preprocessing_steps.append({"name": "foreground_mask", "params": {}})
            if prep.get("use_clahe", False):
                preprocessing_steps.append({"name": "clahe", "params": {}})
            if prep.get("use_gaussian_blur", False):
                preprocessing_steps.append({"name": "gaussian_blur", "params": {}})

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
        preprocessing_steps=preprocessing_steps,
        run_heatmap=True,
        force_retrain=True,
        model_hash=None
    )

    loss_history = results.get("loss_history")
    if loss_history and isinstance(loss_history, dict):
        clean_history = {k: v for k, v in loss_history.items() if isinstance(v, list) and len(v) > 0}
        if clean_history:
            df_loss = pd.DataFrame({k: pd.Series(v) for k, v in clean_history.items()})
            df_loss.to_csv(out_dir / "loss_table.csv", index=False)

            plt.figure(figsize=(10, 5))
            for col in df_loss.columns:
                plt.plot(df_loss[col], label=col)
            plt.title(f"Training Loss Curve - {category}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(out_dir / "loss_diagram.png")
            plt.close()

    save_plots_and_heatmaps(results, out_dir, heatmaps_pred_dir, heatmaps_gt_dir)
    print(f"Completed evaluation for {category}. Outputs saved to {out_dir}")
    tf.keras.backend.clear_session()
    gc.collect()


def evaluate_patchcore(category, tuned):
    import torch
    from app.pipelines.modelling.baseline import run_baseline

    print(f"\n{'='*50}\nEvaluating Patchcore ({'Tuned' if tuned else 'Baseline'}) for category: {category}\n{'='*50}")
    
    base_out_dir = Path("results/evaluation") / ("patchcore_tuned" if tuned else "patchcore")
    out_dir = base_out_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    
    heatmaps_pred_dir = out_dir / "heatmaps" / "prediction"
    heatmaps_gt_dir = out_dir / "heatmaps" / "ground_truth_overlay"
    heatmaps_pred_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_gt_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_steps = []
    backbone = "resnet18"
    feature_layers = ("layer2", "layer3")
    coreset_ratio = 0.1
    num_neighbors = 9

    if tuned:
        json_path = Path("data/hyperparameters/patchcore_best.json")
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                hyperparams = json.load(f)
            config = hyperparams.get(category, {})
            prep = config.get("preprocessing", {})
            hp = config.get("model_hyperparameters", {})
            
            if prep.get("use_foreground_mask", False):
                preprocessing_steps.append({"name": "foreground_mask", "params": {}})
            if prep.get("use_clahe", False):
                preprocessing_steps.append({"name": "clahe", "params": {}})
            if prep.get("use_gaussian_blur", False):
                preprocessing_steps.append({"name": "gaussian_blur", "params": {}})

            feature_layers_str = hp.get("feature_layers", "l2_l3")
            if feature_layers_str == "l2_l3":
                feature_layers = ("layer2", "layer3")
            else:
                feature_layers = ("layer2", "layer3", "layer4")

            backbone = hp.get("backbone", "resnet18")
            coreset_ratio = hp.get("coreset_sampling_ratio", 0.1)
            num_neighbors = hp.get("num_neighbors", 9)

    results = run_baseline(
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
        model_hash=None
    )

    save_plots_and_heatmaps(results, out_dir, heatmaps_pred_dir, heatmaps_gt_dir)
    print(f"Completed evaluation for {category}. Outputs saved to {out_dir}")
    torch.cuda.empty_cache()
    gc.collect()

def orchestrator(model, tuned):
    base_out_dir = Path("results/evaluation")
    if model == "keras":
        out_dir = base_out_dir / ("keras_cae" if tuned else "keras_cae_baseline")
    else:
        out_dir = base_out_dir / ("patchcore_tuned" if tuned else "patchcore")

    categories_to_run = []
    
    if tuned:
        json_path = Path(f"data/hyperparameters/{model}_cae_best.json" if model == "keras" else "data/hyperparameters/patchcore_best.json")
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models.")
    parser.add_argument("--model", type=str, choices=["keras", "patchcore"], required=True, help="Model to evaluate")
    parser.add_argument("--tuned", action="store_true", help="Evaluate the tuned hyperparameters")
    parser.add_argument("--category", type=str, help="Specific category to evaluate (used internally for isolation)")
    args = parser.parse_args()

    if args.category:
        if args.model == "keras":
            evaluate_keras(args.category, args.tuned)
        elif args.model == "patchcore":
            evaluate_patchcore(args.category, args.tuned)
    else:
        orchestrator(args.model, args.tuned)

if __name__ == "__main__":
    main()
