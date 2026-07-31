"""Unit tests for the CLI module with preprocessing options."""

import sys
from unittest.mock import patch

from app.cli import main, preprocess_sys_argv


def test_preprocess_sys_argv() -> None:
    """Test converting positional key=value args to --key value flags."""
    test_args = ["main.py", "baseline", "category=bottle", "clahe_clip_limit=3.0"]

    with patch.object(sys, "argv", test_args):
        preprocess_sys_argv()

        assert sys.argv == ["main.py", "baseline", "--category", "bottle", "--clahe-clip-limit", "3.0"]


def test_cli_baseline_preprocessing_flags() -> None:
    """Test CLI baseline command with individual preprocessing flags."""
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


def test_cli_baseline_preprocessing_json() -> None:
    """Test CLI baseline command with --preprocessing-json argument."""
    json_str = '[{"name": "clahe", "params": {"clip_limit": 2.5}}]'
    test_args = ["main.py", "baseline", "--category", "bottle", "--preprocessing-json", json_str]

    with patch.object(sys, "argv", test_args), patch("app.pipelines.modelling.baseline.run_baseline") as mock_run:
        main()
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["preprocessing_steps"] == [{"name": "clahe", "params": {"clip_limit": 2.5}}]
