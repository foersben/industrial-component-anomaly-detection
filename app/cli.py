"""A command line interface for running pipelines.

This module provides a CLI for running different anomaly detection pipelines.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# Suppress timm deprecation warnings emitted during third-party library imports
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")


def preprocess_sys_argv() -> None:
    """Preprocess sys.argv to convert key=value positional args to --key value flags."""
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            key, val = arg.split("=", 1)
            flag = f"--{key.replace('_', '-')}"
            new_argv.extend([flag, val])
        else:
            new_argv.append(arg)
    sys.argv = new_argv


def _setup_dummy_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Configure arguments for the dummy classifier subcommand.

    Args:
        subparsers: Subparsers action to add the dummy parser to.
    """
    dummy_parser = subparsers.add_parser(
        "dummy", help="Run dummy classifier evaluation to demonstrate accuracy paradox"
    )
    dummy_parser.add_argument(
        "--mode", default="theoretical", help="Evaluation mode: theoretical or real (default: theoretical)"
    )
    dummy_parser.add_argument("--pixels", type=int, default=1000000, help="Total number of simulated pixels")
    dummy_parser.add_argument("--anomaly-ratio", type=float, default=0.015, help="Ratio of anomalous pixels")
    dummy_parser.add_argument(
        "--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root"
    )
    dummy_parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category")


def _setup_patchcore_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Configure arguments for the PatchCore anomaly detection subcommand.

    Args:
        subparsers: Subparsers action to add the patchcore parser to.
    """
    parser = subparsers.add_parser("patchcore", help="Run PatchCore pipeline on MVTec AD dataset")
    parser.add_argument("--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category")
    parser.add_argument(
        "--fpr-limit", type=float, default=1e-4, help="Max False Positive Rate limit for AUPIMO threshold"
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18", help="Feature extractor backbone (default: resnet18)"
    )
    parser.add_argument(
        "--coreset-sampling-ratio", type=float, default=0.1, help="Coreset subsampling ratio (default: 0.1)"
    )
    parser.add_argument(
        "--num-neighbors", type=int, default=9, help="Number of nearest neighbors for scoring (default: 9)"
    )
    parser.add_argument("--heatmap", action="store_true", help="Render anomaly prediction heatmaps")
    parser.add_argument("--force-retrain", action="store_true", help="Bypass cache and force model refit")
    parser.add_argument(
        "--preprocessing-config",
        "--preprocessing-json",
        type=str,
        default=None,
        help="JSON string or file path containing preprocessing steps configuration",
    )
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing step")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit parameter (default: 2.0)")
    parser.add_argument("--gaussian-blur", action="store_true", help="Enable Gaussian Blur preprocessing step")
    parser.add_argument("--blur-kernel-size", type=int, default=5, help="Gaussian Blur kernel size (default: 5)")


def _setup_cae_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Configure arguments for the Keras CAE pipeline subcommand.

    Args:
        subparsers: Subparsers action to add the cae parser to.
    """
    parser = subparsers.add_parser("cae", help="Run Keras Convolutional Autoencoder pipeline on MVTec AD dataset")
    parser.add_argument("--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category")
    parser.add_argument("--img-size", type=int, default=256, help="Image size (default: 256)")
    parser.add_argument("--crop-size", type=int, default=64, help="Crop window size (default: 64)")
    parser.add_argument("--crop-stride", type=int, default=32, help="Crop sliding stride (default: 32)")
    parser.add_argument("--latent-channels", type=int, default=32, help="Bottleneck latent channels (default: 32)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--mask-ratio", type=float, default=0.25, help="MIM mask ratio (default: 0.25)")
    parser.add_argument("--mask-patch-size", type=int, default=8, help="MIM mask patch size (default: 8)")
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="quantile",
        choices=("quantile", "mahalanobis"),
        help="Adaptive threshold calibration method (default: quantile)",
    )
    parser.add_argument("--k-fraction", type=float, default=0.002, help="Top-K anomaly score fraction (default: 0.002)")
    parser.add_argument("--heatmap", action="store_true", help="Render reconstruction error heatmaps")
    parser.add_argument("--force-retrain", action="store_true", help="Bypass cache and force model retrain")
    parser.add_argument(
        "--preprocessing-config",
        "--preprocessing-json",
        type=str,
        default=None,
        help="JSON string or file path containing preprocessing steps configuration",
    )
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing step")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit parameter (default: 2.0)")
    parser.add_argument("--gaussian-blur", action="store_true", help="Enable Gaussian Blur preprocessing step")
    parser.add_argument("--blur-kernel-size", type=int, default=5, help="Gaussian Blur kernel size (default: 5)")


def _setup_dinov2_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Configure the minimal frozen DINOv2 baseline subcommand."""
    parser = subparsers.add_parser("dinov2", help="Run frozen DINOv2 patch nearest-neighbour baseline")
    parser.add_argument("--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category or 'all'")
    parser.add_argument(
        "--fpr-limit",
        type=float,
        default=1e-4,
        help="Fair-protocol AUPIMO upper FPR bound (must remain 1e-4)",
    )
    parser.add_argument("--num-neighbors", type=int, default=1, help="Normal patch neighbors per query patch")
    parser.add_argument(
        "--variant",
        choices=("baseline", "enhanced"),
        default="baseline",
        help="Stock scorer or multi-layer position/density-aware scorer",
    )
    parser.add_argument(
        "--masking",
        choices=("off", "on", "published"),
        default="published",
        help="PCA foreground masking policy (default: published category policy)",
    )
    parser.add_argument("--heatmap", action="store_true", help="Render anomalous test-image heatmaps")


def _setup_dinov3_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Configure the frozen DINOv3 baseline subcommand."""
    parser = subparsers.add_parser("dinov3", help="Run frozen DINOv3 patch nearest-neighbour baseline")
    parser.add_argument("--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category or 'all'")
    parser.add_argument(
        "--fpr-limit",
        type=float,
        default=1e-4,
        help="Fair-protocol AUPIMO upper FPR bound (must remain 1e-4)",
    )
    parser.add_argument("--num-neighbors", type=int, default=1, help="Normal patch neighbors per query patch")
    parser.add_argument("--heatmap", action="store_true", help="Render anomalous test-image heatmaps")


def _parse_preprocessing_steps(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    """Extract preprocessing steps from a JSON file/string or CLI flags.

    Args:
        args: Command line arguments.

    Returns:
        List of preprocessing steps.
    """
    preprocessing_steps: list[dict[str, Any]] = []

    if args.preprocessing_config:
        config_str_or_path = args.preprocessing_config.strip()
        path = Path(config_str_or_path)

        try:
            if path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
            else:
                loaded = json.loads(config_str_or_path)

            if isinstance(loaded, list):
                return loaded
        except json.JSONDecodeError as e:
            print(f"Error parsing preprocessing config: {e}")
            sys.exit(1)

    # Fallback to individual CLI flags
    if args.clahe:
        preprocessing_steps.append({"name": "clahe", "params": {"clip_limit": args.clahe_clip_limit}})
    if args.gaussian_blur:
        preprocessing_steps.append({"name": "gaussian_blur", "params": {"kernel_size": args.blur_kernel_size}})

    return preprocessing_steps if preprocessing_steps else None


def _handle_dummy_command(args: argparse.Namespace) -> None:
    """Execute the dummy evaluation subcommand.

    Args:
        args: Command line arguments.
    """
    from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

    mode = args.mode.lower()
    if mode == "real":
        run_real_data_dummy(data_root=args.data_root, category=args.category)
    elif mode == "theoretical":
        run_dummy_evaluation(total_pixels=args.pixels, anomaly_ratio=args.anomaly_ratio)
    else:
        print(f"Error: Invalid mode '{args.mode}'. Choose from 'theoretical' or 'real'.")
        sys.exit(1)


def _handle_patchcore_command(args: argparse.Namespace) -> None:
    """Execute the PatchCore pipeline subcommand.

    Args:
        args: Command line arguments.
    """
    from app.pipelines.modelling.patchcore import run_patchcore_pipeline

    preprocessing_steps = _parse_preprocessing_steps(args)

    run_patchcore_pipeline(
        data_root=args.data_root,
        category=args.category,
        fpr_limit=args.fpr_limit,
        backbone=getattr(args, "backbone", "resnet18"),
        coreset_sampling_ratio=getattr(args, "coreset_sampling_ratio", 0.1),
        num_neighbors=getattr(args, "num_neighbors", 9),
        run_heatmap=getattr(args, "heatmap", False),
        force_retrain=getattr(args, "force_retrain", False),
        preprocessing_steps=preprocessing_steps,
    )


def _handle_cae_command(args: argparse.Namespace) -> None:
    """Execute the Keras Convolutional Autoencoder pipeline subcommand.

    Args:
        args: Command line arguments.
    """
    from app.pipelines.modelling.keras_cae import run_keras_cae_pipeline

    preprocessing_steps = _parse_preprocessing_steps(args)

    run_keras_cae_pipeline(
        data_root=args.data_root,
        category=args.category,
        img_size=args.img_size,
        crop_size=args.crop_size,
        crop_stride=args.crop_stride,
        latent_channels=args.latent_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        threshold_method=args.threshold_method,
        k_fraction=args.k_fraction,
        preprocessing_steps=preprocessing_steps,
        run_heatmap=getattr(args, "heatmap", False),
        force_retrain=getattr(args, "force_retrain", False),
    )


def _handle_dinov2_command(args: argparse.Namespace) -> None:
    """Execute the frozen DINOv2 baseline subcommand."""
    from app.pipelines.modelling.dinov2_baseline import run_dinov2_baseline

    run_dinov2_baseline(
        data_root=args.data_root,
        category=args.category,
        fpr_limit=args.fpr_limit,
        num_neighbors=args.num_neighbors,
        masking=args.masking,
        run_heatmap=args.heatmap,
        variant=args.variant,
    )


def _handle_dinov3_command(args: argparse.Namespace) -> None:
    """Execute the frozen DINOv3 baseline subcommand."""
    from app.pipelines.modelling.dinov3_baseline import run_dinov3_baseline

    run_dinov3_baseline(
        data_root=args.data_root,
        category=args.category,
        fpr_limit=args.fpr_limit,
        num_neighbors=args.num_neighbors,
        run_heatmap=args.heatmap,
    )


def main() -> None:
    """Run the main CLI."""
    preprocess_sys_argv()

    parser = argparse.ArgumentParser(description="Industrial Component Anomaly Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _setup_dummy_parser(subparsers)
    _setup_patchcore_parser(subparsers)
    _setup_cae_parser(subparsers)
    _setup_dinov2_parser(subparsers)
    _setup_dinov3_parser(subparsers)

    args = parser.parse_args()

    if args.command == "dummy":
        _handle_dummy_command(args)
    elif args.command == "patchcore":
        _handle_patchcore_command(args)
    elif args.command == "cae":
        _handle_cae_command(args)
    elif args.command == "dinov2":
        _handle_dinov2_command(args)
    elif args.command == "dinov3":
        _handle_dinov3_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
