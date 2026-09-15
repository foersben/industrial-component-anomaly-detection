"""Tests for persisted fair-evaluation summaries."""

from pathlib import Path

import pandas as pd

from scripts.evaluate import save_plots_and_heatmaps


def test_metrics_summary_persists_image_confusion_counts(tmp_path: Path) -> None:
    """The shared script serializer writes all image confusion evidence."""
    output_dir = tmp_path / "evaluation"
    prediction_dir = output_dir / "heatmaps" / "prediction"
    ground_truth_dir = output_dir / "heatmaps" / "ground_truth"
    prediction_dir.mkdir(parents=True)
    ground_truth_dir.mkdir(parents=True)
    results = {
        "image_level": {
            "auroc": 0.9,
            "true_positives": 7,
            "false_positives": 2,
            "false_negatives": 3,
            "true_negatives": 8,
            "precision": 7 / 9,
            "recall": 0.7,
            "f1_score": 0.736842,
        },
        "pixel_level": {"auroc": 0.8, "aupimo": 0.6},
    }

    save_plots_and_heatmaps(results, output_dir, prediction_dir, ground_truth_dir)

    summary = pd.read_csv(output_dir / "metrics_summary.csv").iloc[0]
    assert summary["image_level_true_positives"] == 7
    assert summary["image_level_false_positives"] == 2
    assert summary["image_level_false_negatives"] == 3
    assert summary["image_level_true_negatives"] == 8
