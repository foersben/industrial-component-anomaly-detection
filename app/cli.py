"""A command line interface for running pipelines.

This module provides a CLI for running different anomaly detection pipelines.
"""

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


def _setup_baseline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Configure arguments for the Patchcore baseline subcommand.

    Args:
        subparsers: Subparsers action to add the baseline parser to.
    """
    baseline_parser = subparsers.add_parser("baseline", help="Run Patchcore baseline on MVTec AD dataset")
    baseline_parser.add_argument(
        "--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root"
    )
    baseline_parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category")
    baseline_parser.add_argument(
        "--fpr-limit", type=float, default=1e-4, help="Max False Positive Rate limit for AUPIMO threshold"
    )
    baseline_parser.add_argument(
        "--preprocessing-config",
        "--preprocessing-json",
        type=str,
        default=None,
        help="JSON string or file path containing preprocessing steps configuration",
    )
    baseline_parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing step")
    baseline_parser.add_argument(
        "--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit parameter (default: 2.0)"
    )
    baseline_parser.add_argument("--gaussian-blur", action="store_true", help="Enable Gaussian Blur preprocessing step")
    baseline_parser.add_argument(
        "--blur-kernel-size", type=int, default=5, help="Gaussian Blur kernel size (default: 5)"
    )


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


def _handle_baseline_command(args: argparse.Namespace) -> None:
    """Execute the Patchcore baseline subcommand.

    Args:
        args: Command line arguments.
    """
    from app.pipelines.modelling.baseline import run_baseline

    preprocessing_steps = _parse_preprocessing_steps(args)

    run_baseline(
        data_root=args.data_root,
        category=args.category,
        fpr_limit=args.fpr_limit,
        preprocessing_steps=preprocessing_steps,
    )


def main() -> None:
    """Run the main CLI."""
    preprocess_sys_argv()

    parser = argparse.ArgumentParser(description="Industrial Component Anomaly Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _setup_dummy_parser(subparsers)
    _setup_baseline_parser(subparsers)

    args = parser.parse_args()

    if args.command == "dummy":
        _handle_dummy_command(args)
    elif args.command == "baseline":
        _handle_baseline_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
