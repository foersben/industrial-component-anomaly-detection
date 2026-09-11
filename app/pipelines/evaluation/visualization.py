"""Evaluation visualization functions for Streamlit."""

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


def load_and_prepare_evaluation_data(data_path: str) -> ProcessedEvaluationData:
    """Loads .npz data, aligns array shapes, and computes PR/AUPIMO metrics.

    Args:
        data_path: Path to the .npz file containing precision, recall, and thresholds.

    Returns:
        Data container with processed evaluation metrics.
    """
    data = np.load(data_path, allow_pickle=True)

    precisions = data["precision"]
    recalls = data["recall"]
    thresholds = data["thresholds"]
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

    # Detect evaluation level
    raw_level = str(data["level"]) if "level" in data else ""
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


def plot_tradeoff_curve(data: ProcessedEvaluationData) -> Figure:
    """Generate the precision and recall threshold-tradeoff figure.

    Args:
        data: Processed evaluation data.

    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(data.thresholds, data.precisions, label="Precision", color="#1f77b4", linewidth=2)
    ax.plot(data.thresholds, data.recalls, label="Recall", color="#ff7f0e", linewidth=2)
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
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    # Full PR Curve
    ax.plot(
        data.sorted_recalls,
        data.sorted_precisions,
        label=f"PR Curve (AUPR={data.integrated_aupr:.4f})",
        color="#8E44AD",
        lw=2.5,
    )
    ax.fill_between(data.sorted_recalls, data.sorted_precisions, alpha=0.2, color="#8E44AD")

    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({data.eval_level_label})")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    return fig


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

    # 2. Streamlit Charts Side-by-Side
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig1 = plot_tradeoff_curve(data)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_chart2:
        fig2 = plot_pr_curve(data)
        st.pyplot(fig2)
        plt.close(fig2)
