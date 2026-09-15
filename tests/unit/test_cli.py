"""Unit tests for the command-line interface (CLI) argument parsing and execution dispatch."""

import sys
from pathlib import Path
from unittest.mock import patch

from app.cli import main, preprocess_sys_argv


def test_preprocess_sys_argv() -> None:
    """Verify that positional key=value CLI arguments are converted into standard --key value flags."""
    test_args = ["main.py", "patchcore", "category=bottle", "clahe_clip_limit=3.0"]

    with patch.object(sys, "argv", test_args):
        preprocess_sys_argv()

        assert sys.argv == ["main.py", "patchcore", "--category", "bottle", "--clahe-clip-limit", "3.0"]


def test_cli_patchcore_preprocessing_flags() -> None:
    """Verify that CLI patchcore subcommand correctly translates individual preprocessing flags."""
    test_args = [
        "main.py",
        "patchcore",
        "--category",
        "bottle",
        "--clahe",
        "--clahe-clip-limit",
        "3.0",
        "--gaussian-blur",
        "--blur-kernel-size",
        "5",
    ]

    with (
        patch.object(sys, "argv", test_args),
        patch("app.pipelines.modelling.patchcore.run_patchcore_pipeline") as mock_run,
    ):
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["category"] == "bottle"

        expected_steps = [
            {"name": "clahe", "params": {"clip_limit": 3.0}},
            {"name": "gaussian_blur", "params": {"kernel_size": 5}},
        ]

        assert kwargs["pipeline"] == expected_steps


def test_cli_patchcore_preprocessing_json_string() -> None:
    """Verify that CLI patchcore subcommand parses inline JSON strings provided to --preprocessing-config."""
    json_str = '[{"name": "clahe", "params": {"clip_limit": 2.5}}]'
    test_args = ["main.py", "patchcore", "--category", "bottle", "--preprocessing-config", json_str]

    with (
        patch.object(sys, "argv", test_args),
        patch("app.pipelines.modelling.patchcore.run_patchcore_pipeline") as mock_run,
    ):
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["pipeline"] == [{"name": "clahe", "params": {"clip_limit": 2.5}}]


def test_cli_patchcore_preprocessing_json_file(tmp_path: Path) -> None:
    """Verify that CLI patchcore subcommand reads and parses config files passed to --preprocessing-config.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    config_file = tmp_path / "config.json"
    config_file.write_text('[{"name": "gaussian_blur", "params": {"kernel_size": 3}}]', encoding="utf-8")
    test_args = ["main.py", "patchcore", "--category", "bottle", "--preprocessing-config", str(config_file)]

    with (
        patch.object(sys, "argv", test_args),
        patch("app.pipelines.modelling.patchcore.run_patchcore_pipeline") as mock_run,
    ):
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["pipeline"] == [{"name": "gaussian_blur", "params": {"kernel_size": 3}}]


def test_cli_dinov2_exposes_only_justified_baseline_options() -> None:
    """DINOv2 CLI dispatches the pre-registered masking policy and k-NN size."""
    test_args = ["main.py", "dinov2", "--category", "capsule", "--masking", "published", "--num-neighbors", "3"]

    with (
        patch.object(sys, "argv", test_args),
        patch("app.pipelines.modelling.dino.run_dinov2_baseline") as mock_run,
    ):
        main()

    mock_run.assert_called_once_with(
        data_root="data/raw/mvtec_ad",
        category="capsule",
        fpr_limit=1e-4,
        num_neighbors=3,
        pipeline=None,
        masking="published",
        run_heatmap=False,
        variant="baseline",
    )


def test_cli_dinov2_accepts_enhanced_variant() -> None:
    """The CLI exposes the pre-registered enhanced DINOv2 scorer."""
    with (
        patch.object(sys, "argv", ["main.py", "dinov2", "--category", "bottle", "--variant", "enhanced"]),
        patch("app.pipelines.modelling.dino.run_dinov2_baseline") as mock_run,
    ):
        main()

    assert mock_run.call_args.kwargs["variant"] == "enhanced"


def test_cli_dinov2_accepts_all_categories() -> None:
    """The CLI forwards the all-category selector to the model dispatcher."""
    with (
        patch.object(sys, "argv", ["main.py", "dinov2", "--category", "all"]),
        patch("app.pipelines.modelling.dino.run_dinov2_baseline") as mock_run,
    ):
        main()

    assert mock_run.call_args.kwargs["category"] == "all"


def test_cli_dinov3_accepts_all_categories() -> None:
    """The CLI exposes the DINOv3 all-category baseline."""
    with (
        patch.object(sys, "argv", ["main.py", "dinov3", "--category", "all"]),
        patch("app.pipelines.modelling.dino.run_dinov3_baseline") as mock_run,
    ):
        main()

    mock_run.assert_called_once_with(
        data_root="data/raw/mvtec_ad",
        category="all",
        fpr_limit=1e-4,
        num_neighbors=1,
        run_heatmap=False,
    )


def test_cli_patchcore_subcommand() -> None:
    """The CLI correctly dispatches the patchcore subcommand."""
    test_args = [
        "main.py",
        "patchcore",
        "--category",
        "bottle",
        "--backbone",
        "resnet18",
        "--coreset-sampling-ratio",
        "0.2",
        "--heatmap",
    ]

    with (
        patch.object(sys, "argv", test_args),
        patch("app.pipelines.modelling.patchcore.run_patchcore_pipeline") as mock_run,
    ):
        main()

    mock_run.assert_called_once_with(
        data_root="data/raw/mvtec_ad",
        category="bottle",
        fpr_limit=1e-4,
        backbone="resnet18",
        coreset_sampling_ratio=0.2,
        num_neighbors=9,
        run_heatmap=True,
        force_retrain=False,
        pipeline=None,
    )


def test_cli_cae_subcommand() -> None:
    """The CLI correctly dispatches the cae subcommand."""
    test_args = [
        "main.py",
        "cae",
        "--category",
        "bottle",
        "--epochs",
        "10",
        "--batch-size",
        "32",
        "--heatmap",
    ]

    with (
        patch.object(sys, "argv", test_args),
        patch("app.pipelines.modelling.keras_cae.run_keras_cae_pipeline") as mock_run,
    ):
        main()

    mock_run.assert_called_once_with(
        data_root="data/raw/mvtec_ad",
        category="bottle",
        img_size=256,
        crop_size=64,
        crop_stride=32,
        latent_channels=32,
        epochs=10,
        batch_size=32,
        mask_ratio=0.25,
        mask_patch_size=8,
        threshold_method="quantile",
        k_fraction=0.002,
        pipeline=None,
        run_heatmap=True,
        force_retrain=False,
    )
