"""A command line interface for running pipelines.

This module provides a CLI for running different anomaly detection pipelines.
"""

import argparse
import sys
import warnings

# Suppress timm deprecation warnings emitted during third-party library imports
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")


def preprocess_sys_argv() -> None:
    """Preprocess sys.argv to convert key=value positional args to --key value flags.

    Some sort of stupidity management.
    """
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            key, val = arg.split("=", 1)
            flag = f"--{key.replace('_', '-')}"
            new_argv.extend([flag, val])
        else:
            new_argv.append(arg)
    sys.argv = new_argv


def main() -> None:
    """Run the main CLI."""
    preprocess_sys_argv()

    parser = argparse.ArgumentParser(description="Industrial Component Anomaly Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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

    baseline_parser = subparsers.add_parser("baseline", help="Run Patchcore baseline on MVTec AD dataset")
    baseline_parser.add_argument(
        "--data-root", type=str, default="data/raw/mvtec_ad", help="Path to MVTec AD dataset root"
    )
    baseline_parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category")
    baseline_parser.add_argument(
        "--fpr-limit", type=float, default=1e-4, help="Max False Positive Rate limit for AUPIMO threshold"
    )

    args = parser.parse_args()

    if args.command == "dummy":
        from app.pipelines.modelling.dummy_classifier import run_dummy_evaluation, run_real_data_dummy

        mode = args.mode.lower()
        if mode == "real":
            run_real_data_dummy(data_root=args.data_root, category=args.category)
        elif mode == "theoretical":
            run_dummy_evaluation(total_pixels=args.pixels, anomaly_ratio=args.anomaly_ratio)
        else:
            print(f"Error: Invalid mode '{args.mode}'. Choose from 'theoretical' or 'real'.")
            sys.exit(1)
    elif args.command == "baseline":
        from app.pipelines.modelling.baseline import run_baseline

        run_baseline(data_root=args.data_root, category=args.category, fpr_limit=args.fpr_limit)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
