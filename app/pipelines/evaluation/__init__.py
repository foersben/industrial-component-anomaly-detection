"""Evaluation subpackage."""

from app.pipelines.evaluation.cae_metrics import compute_aupimo, compute_image_auroc, evaluate_cae
from app.pipelines.evaluation.heatmaps import compute_error_heatmap, overlay_ground_truth, overlay_heatmap
from app.pipelines.evaluation.metrics import compute_and_save_pr_metrics, save_evaluation_metrics
from app.pipelines.evaluation.scoring import compute_adaptive_threshold, compute_pixel_error_map, top_k_pooling
from app.pipelines.evaluation.visualization import render_evaluation_curves

__all__ = [
    "compute_adaptive_threshold",
    "compute_and_save_pr_metrics",
    "compute_aupimo",
    "compute_error_heatmap",
    "compute_image_auroc",
    "compute_pixel_error_map",
    "evaluate_cae",
    "overlay_ground_truth",
    "overlay_heatmap",
    "render_evaluation_curves",
    "save_evaluation_metrics",
    "top_k_pooling",
]
