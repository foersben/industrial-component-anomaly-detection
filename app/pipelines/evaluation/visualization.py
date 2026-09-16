"""Evaluation visualization functions for Streamlit."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.figure import Figure
from sklearn.metrics import auc


class ProcessedEvaluationData(NamedTuple):
    """Data container for processed evaluation metrics.

    Attributes:
        precisions: Array of precision values.
        recalls: Array of recall values.
        thresholds: Array of threshold values.
        sorted_recalls: Array of recall values sorted in ascending order.
        sorted_precisions: Array of precision values sorted in ascending order.
        t_crossover: Threshold value at which precision and recall are approximately equal.
        integrated_aupr: Area Under the Precision-Recall Curve.
        eval_level_label: Label indicating the evaluation level (image or pixel).
    """

    precisions: np.ndarray[Any, Any]
    recalls: np.ndarray[Any, Any]
    thresholds: np.ndarray[Any, Any]
    sorted_recalls: np.ndarray[Any, Any]
    sorted_precisions: np.ndarray[Any, Any]
    t_crossover: float
    integrated_aupr: float
    eval_level_label: str


MAX_PLOT_POINTS = 5_000


def _downsample_curve(
    x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Bound chart rendering cost without changing full-resolution metric calculations."""
    if len(x) <= MAX_PLOT_POINTS:
        return x, y
    indices = np.linspace(0, len(x) - 1, num=MAX_PLOT_POINTS, dtype=np.int64)
    return x[indices], y[indices]


@st.cache_data(show_spinner=False)
def _load_and_prepare_evaluation_data_cached(data_path: str, modified_ns: int) -> ProcessedEvaluationData:
    """Load processed metrics once for an unchanged on-disk artifact."""
    del modified_ns
    with np.load(data_path, allow_pickle=True) as data:
        precisions = data["precision"]
        recalls = data["recall"]
        thresholds = data["thresholds"]
        raw_level = str(data["level"]) if "level" in data else ""

    # Align shapes if metrics returned boundary values
    if len(precisions) == len(thresholds) + 1:
        precisions = precisions[:-1]
        recalls = recalls[:-1]

    # Calculate Optimal Breakpoint
    diff = np.abs(precisions - recalls)
    t_crossover = float(thresholds[np.argmin(diff)])

    # Arrays from precision_recall_curve are ordered by decision threshold,
    # making recall monotonically decreasing. This preserves the curve's continuous line.
    sorted_recalls = recalls
    sorted_precisions = precisions

    # Pad curve down to Recall = 0.0 for accurate AUPR integration
    if sorted_recalls[-1] > 0.0:
        sorted_recalls = np.append(sorted_recalls, 0.0)
        sorted_precisions = np.append(sorted_precisions, sorted_precisions[-1])

    integrated_aupr = float(auc(sorted_recalls, sorted_precisions))
    eval_level_label = (
        "Image-Level (Classification)" if "image" in raw_level or "image" in data_path else "Pixel-Level (Localization)"
    )

    return ProcessedEvaluationData(
        precisions=precisions,
        recalls=recalls,
        thresholds=thresholds,
        sorted_recalls=sorted_recalls,
        sorted_precisions=sorted_precisions,
        t_crossover=t_crossover,
        integrated_aupr=integrated_aupr,
        eval_level_label=eval_level_label,
    )


def load_and_prepare_evaluation_data(data_path: str) -> ProcessedEvaluationData:
    """Loads .npz data, aligns array shapes, and computes PR/AUPIMO metrics.

    Args:
        data_path: Path to the .npz file containing precision, recall, and thresholds.

    Returns:
        Data container with processed evaluation metrics.
    """
    metrics_path = Path(data_path)
    return _load_and_prepare_evaluation_data_cached(str(metrics_path), metrics_path.stat().st_mtime_ns)


def plot_tradeoff_curve(data: ProcessedEvaluationData) -> Figure:
    """Generate the precision and recall threshold-tradeoff figure.

    Args:
        data: Processed evaluation data.

    Returns:
        The matplotlib figure.
    """
    thresholds, precisions = _downsample_curve(data.thresholds, data.precisions)
    _, recalls = _downsample_curve(data.thresholds, data.recalls)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(thresholds, precisions, label="Precision", color="#1f77b4", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color="#ff7f0e", linewidth=2)
    ax.axvline(
        x=data.t_crossover,
        color="gray",
        linestyle="--",
        label=f"Breakpoint (~{data.t_crossover:.2f})",
    )

    ax.set_title(f"Threshold Tradeoff Curve ({data.eval_level_label})")
    ax.set_xlabel("Binarization Threshold")
    ax.set_ylabel("Metric Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_pr_curve(data: ProcessedEvaluationData) -> Figure:
    """Generate the precision-recall figure.

    Args:
        data: Processed evaluation data.

    Returns:
        The matplotlib figure.
    """
    recalls, precisions = _downsample_curve(data.sorted_recalls, data.sorted_precisions)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    # Full PR Curve
    ax.plot(
        recalls,
        precisions,
        label=f"PR Curve (AUPR={data.integrated_aupr:.4f})",
        color="#8E44AD",
        lw=2.5,
    )
    ax.fill_between(recalls, precisions, alpha=0.2, color="#8E44AD")

    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({data.eval_level_label})")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    return fig


def _save_figure_once(figure: Figure, output_path: Path) -> None:
    """Persist a figure atomically unless an existing cache is already available."""
    if output_path.is_file():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.stem}-", suffix=".png", delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
        figure.savefig(temp_path, bbox_inches="tight")
        if not output_path.exists():
            temp_path.replace(output_path)
            temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _resolve_or_create_curve_images(data_path: str, data: ProcessedEvaluationData) -> tuple[Path, Path]:
    """Return persistent chart images, generating only cache misses."""
    metrics_path = Path(data_path)
    tradeoff_path = metrics_path.with_name(f"{metrics_path.stem}_threshold_tradeoff.png")
    pr_path = metrics_path.with_name(f"{metrics_path.stem}_precision_recall.png")

    if not tradeoff_path.is_file():
        tradeoff_figure = plot_tradeoff_curve(data)
        try:
            _save_figure_once(tradeoff_figure, tradeoff_path)
        finally:
            plt.close(tradeoff_figure)

    if not pr_path.is_file():
        pr_figure = plot_pr_curve(data)
        try:
            _save_figure_once(pr_figure, pr_path)
        finally:
            plt.close(pr_figure)

    return tradeoff_path, pr_path


def render_evaluation_curves(data_path: str) -> None:
    """Orchestrates data loading, chart generation, and rendering in Streamlit.

    Args:
        data_path: Path to the .npz file containing precision, recall, and thresholds.

    Raises:
        IOError: If data cannot be loaded from the specified path.
    """
    try:
        data = load_and_prepare_evaluation_data(data_path)
    except Exception as e:
        st.error(f"Failed to load data from {data_path}: {e}")
        return

    # 1. Streamlit Metrics Header
    st.subheader(f"Model Evaluation Metrics — {data.eval_level_label}")

    cols = st.columns(2)
    cols[0].metric("Optimal Breakpoint (Prec ~= Rec)", f"{data.t_crossover:.4f}")
    cols[1].metric("PR-AUC (AUPR)", f"{data.integrated_aupr:.4f}")

    tradeoff_path, pr_path = _resolve_or_create_curve_images(data_path, data)

    # 2. Streamlit Charts Side-by-Side
    col_chart1, col_chart2 = st.columns(2)
    col_chart1.image(str(tradeoff_path), width="stretch")
    col_chart2.image(str(pr_path), width="stretch")
