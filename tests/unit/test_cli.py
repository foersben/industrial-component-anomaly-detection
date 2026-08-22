"""Unit tests for the command-line interface (CLI) argument parsing and execution dispatch."""

import sys
from pathlib import Path
from unittest.mock import patch

from app.cli import main, preprocess_sys_argv


def test_preprocess_sys_argv() -> None:
    """Verify that positional key=value CLI arguments are converted into standard --key value flags."""
    test_args = ["main.py", "baseline", "category=bottle", "clahe_clip_limit=3.0"]

    with patch.object(sys, "argv", test_args):
        preprocess_sys_argv()

        assert sys.argv == ["main.py", "baseline", "--category", "bottle", "--clahe-clip-limit", "3.0"]


def test_cli_baseline_preprocessing_flags() -> None:
    """Verify that CLI baseline subcommand correctly translates individual preprocessing flags."""
    test_args = [
        "main.py",
        "baseline",
        "--category",
        "bottle",
        "--clahe",
        "--clahe-clip-limit",
        "3.0",
        "--gaussian-blur",
        "--blur-kernel-size",
        "5",
    ]

    with patch.object(sys, "argv", test_args), patch("app.pipelines.modelling.baseline.run_baseline") as mock_run:
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["category"] == "bottle"

        expected_steps = [
            {"name": "clahe", "params": {"clip_limit": 3.0}},
            {"name": "gaussian_blur", "params": {"kernel_size": 5}},
        ]

        assert kwargs["preprocessing_steps"] == expected_steps


def test_cli_baseline_preprocessing_json_string() -> None:
    """Verify that CLI baseline subcommand parses inline JSON strings provided to --preprocessing-config."""
    json_str = '[{"name": "clahe", "params": {"clip_limit": 2.5}}]'
    test_args = ["main.py", "baseline", "--category", "bottle", "--preprocessing-config", json_str]

    with patch.object(sys, "argv", test_args), patch("app.pipelines.modelling.baseline.run_baseline") as mock_run:
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["preprocessing_steps"] == [{"name": "clahe", "params": {"clip_limit": 2.5}}]


def test_cli_baseline_preprocessing_json_file(tmp_path: Path) -> None:
    """Verify that CLI baseline subcommand reads and parses config files passed to --preprocessing-config.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    config_file = tmp_path / "config.json"
    config_file.write_text('[{"name": "gaussian_blur", "params": {"kernel_size": 3}}]', encoding="utf-8")
    test_args = ["main.py", "baseline", "--category", "bottle", "--preprocessing-config", str(config_file)]

    with patch.object(sys, "argv", test_args), patch("app.pipelines.modelling.baseline.run_baseline") as mock_run:
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["preprocessing_steps"] == [{"name": "gaussian_blur", "params": {"kernel_size": 3}}]
