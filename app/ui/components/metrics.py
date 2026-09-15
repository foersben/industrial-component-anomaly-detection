"""Metrics display components for Streamlit UI."""

from pathlib import Path
from typing import Any

import streamlit as st

from app.pipelines.evaluation.visualization import render_evaluation_curves


def _display_metrics_row(metrics: dict[str, Any]) -> None:
    """Helper to render 3-column metric cards (Accuracy, Precision, Recall).

    Args:
        metrics: The metrics to display.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    col2.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.2f}")


def _display_level_metrics(title: str, metrics: dict[str, Any], level_type: str = "image") -> None:
    """Helper to render level-specific metrics (Image or Pixel).

    Args:
        title: The title of the metrics.
        metrics: The metrics to display.
        level_type: Type of level ('image' or 'pixel').
    """
    st.subheader(title)
    if level_type == "pixel":
        m1, m2, m3 = st.columns(3)
        m1.metric("Pixel AUROC", f"{metrics.get('auroc', 0.0):.4f}")
        aupimo_score = metrics.get("aupimo_score", metrics.get("aupimo", 0.0))
        m2.metric("AUPIMO Score", f"{aupimo_score:.4f}")
        m3.metric("Pixel F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Image AUROC", f"{metrics.get('auroc', 0.0):.4f}")
        m2.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
        m3.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")

    st.caption(f"Saved: `{metrics.get('metrics_path', '')}`")


def _render_step_badge(name: str) -> None:
    """Render an individual preprocessing step markdown badge.

    Args:
        name: Name of the preprocessing filter.
    """
    badges = {
        "foreground_mask": "🟢 **Foreground Mask** *(Otsu + Canny)*",
        "clahe": "🟢 **CLAHE** *(Contrast Equalization)*",
        "gaussian_blur": "🟢 **Gaussian Blur** *(Denoising)*",
    }
    st.markdown(badges.get(name, f"🟢 **`{name}`**"))


def _render_overview_preprocessing_column(results: dict[str, Any]) -> None:
    """Render active preprocessing configuration badge list.

    Args:
        results: Pipeline results dictionary.
    """
    st.markdown("#### 🔧 Preprocessing")
    prep_steps = results.get("preprocessing_steps")
    if prep_steps is None and isinstance(results.get("metadata"), dict):
        prep_steps = results["metadata"].get("preprocessing_steps")

    if prep_steps is None or not isinstance(prep_steps, list):
        st.info("⚠️ *Preprocessing configuration was not recorded with this legacy model run.*")
        return

    if not prep_steps:
        st.markdown("⚪ **None** *(Raw unmodified images)*")
        return

    for s in prep_steps:
        name = str(s.get("name", "Unknown Step"))
        _render_step_badge(name)


def _render_overview_cae_hyperparams(hp: dict[str, Any], results: dict[str, Any]) -> None:
    """Render Keras CAE hyperparameter details.

    Args:
        hp: Hyperparameters dictionary.
        results: Top-level results dictionary for fallbacks.
    """
    st.markdown(f"- **Category:** `{hp.get('category', results.get('category', 'N/A'))}`")
    st.markdown(f"- **Epochs:** `{hp.get('epochs', results.get('epochs', 'N/A'))}`")
    st.markdown(f"- **Batch Size:** `{hp.get('batch_size', 'N/A')}`")
    img_sz = hp.get("img_size", 128)
    st.markdown(f"- **Image Size:** `{img_sz}x{img_sz}`")
    crop_sz = hp.get("crop_size", 64)
    crop_str = hp.get("crop_stride", 32)
    st.markdown(f"- **Crop Size / Stride:** `{crop_sz}x{crop_sz}` *(stride: {crop_str})*")
    latent = hp.get("latent_channels", hp.get("latent_dim", "N/A"))
    st.markdown(f"- **Latent Channels:** `{latent}`")
    mask_r = hp.get("mask_ratio", 0.25)
    try:
        mask_pct = float(mask_r) * 100
    except (ValueError, TypeError):
        mask_pct = 25.0
    st.markdown(f"- **Masking (MIM):** `{mask_pct:.0f}%` *(patch: {hp.get('mask_patch_size', 8)})*")
    st.markdown(f"- **Thresholding:** `{hp.get('threshold_method', 'quantile_95')}`")


def _render_overview_patchcore_hyperparams(hp: dict[str, Any], results: dict[str, Any]) -> None:
    """Render PatchCore or DINO hyperparameter details.

    Args:
        hp: Hyperparameters dictionary.
        results: Top-level results dictionary for fallbacks.
    """
    st.markdown(f"- **Category:** `{results.get('category', 'N/A')}`")
    encoder = hp.get("encoder_name", hp.get("backbone", "resnet18"))
    st.markdown(f"- **Backbone / Encoder:** `{encoder}`")
    if "coreset_sampling_ratio" in hp:
        st.markdown(f"- **Coreset Ratio:** `{hp.get('coreset_sampling_ratio', 0.1)}`")
    if "num_neighbors" in hp:
        st.markdown(f"- **Nearest Neighbors:** `{hp.get('num_neighbors', 1)}`")
    st.markdown(f"- **FPR Limit:** `{hp.get('fpr_limit', 1e-4)}`")
    batch_sz = hp.get("train_batch_size", hp.get("batch_size", 16))
    st.markdown(f"- **Batch Size:** `{batch_sz}`")


def _render_overview_hyperparameters_column(results: dict[str, Any], model_type: str) -> None:
    """Render the hyperparameters overview column.

    Args:
        results: Pipeline results dictionary.
        model_type: Model type ('cae' or 'patchcore').
    """
    st.markdown("#### ⚙️ Hyperparameters")
    if model_type == "cae":
        hp = results.get("hyperparameters") or (
            results.get("metadata") if isinstance(results.get("metadata"), dict) else None
        )
        if hp and isinstance(hp, dict):
            _render_overview_cae_hyperparams(hp, results)
        else:
            st.info("⚠️ *Hyperparameter details were not recorded with this legacy model run.*")
    else:
        hp = results.get("hyperparameters", {})
        if hp and isinstance(hp, dict) and len(hp) > 0:
            _render_overview_patchcore_hyperparams(hp, results)
        else:
            st.info("⚠️ *Hyperparameters were not recorded with this legacy model run.*")


def _render_overview_dataset_split_column(results: dict[str, Any], model_type: str) -> None:
    """Render the dataset partition split overview column.

    Args:
        results: Pipeline results dictionary.
        model_type: Model type ('cae' or 'patchcore').
    """
    st.markdown("#### 📊 Dataset Partition Split")
    split = results.get("dataset_split") or (
        results.get("metadata", {}).get("dataset_split") if isinstance(results.get("metadata"), dict) else None
    )
    if not (split and isinstance(split, dict) and len(split) > 0):
        st.info("⚠️ *Dataset partition sample counts were not recorded with this legacy model run.*")
        return

    st.markdown(f"- **Train (Normal):** `{split.get('train_normal', 'N/A')}`")
    if model_type == "cae":
        st.markdown(f"- **Validation (Normal, 15%):** `{split.get('val_normal', 'N/A')}`")
        test_tot = split.get("test_total", "N/A")
        test_norm = split.get("test_normal")
        test_anom = split.get("test_anomalous")
        if test_norm is not None and test_anom is not None:
            st.markdown(f"- **Test Total:** `{test_tot}` *({test_norm} normal, {test_anom} anomalous)*")
        else:
            st.markdown(f"- **Test Total:** `{test_tot}`")
    else:
        st.markdown(f"- **Test Total:** `{split.get('test_total', 'N/A')}`")


def _render_model_run_overview(results: dict[str, Any], model_type: str = "cae") -> None:
    """Render a comprehensive overview of active preprocessing, hyperparameters, and dataset split.

    Args:
        results: Results dictionary returned from the pipeline or model evaluation.
        model_type: Type of model evaluated ('cae' or 'patchcore').
    """
    with st.expander("📋 Model Run Overview (Preprocessing, Hyperparameters & Dataset Split)", expanded=True):
        col_prep, col_hp, col_split = st.columns(3)
        with col_prep:
            _render_overview_preprocessing_column(results)
        with col_hp:
            _render_overview_hyperparameters_column(results, model_type)
        with col_split:
            _render_overview_dataset_split_column(results, model_type)
    st.divider()


def _render_evaluation_summary(results: dict[str, Any], model_type: str = "cae") -> None:
    """Render structured Image-Level and Pixel-Level evaluation metric cards and dynamic info boxes.

    Args:
        results: Dictionary containing image_level and pixel_level evaluation metrics.
        model_type: Type of model evaluated ('cae' or 'patchcore').
    """
    _render_model_run_overview(results, model_type=model_type)

    if "image_level" in results and "pixel_level" in results:
        col_img, col_pix = st.columns(2)
        img_metrics = results.get("image_level", {})
        pix_metrics = results.get("pixel_level", {})

        with col_img:
            _display_level_metrics("Image-Level (Classification)", img_metrics, level_type="image")
            image_threshold = float(img_metrics.get("threshold", results.get("threshold", 0.0)))
            image_precision = float(img_metrics.get("precision", results.get("precision", 0.0)))
            image_recall = float(img_metrics.get("recall", results.get("recall", 0.0)))

            st.info(
                f"**Image-Level Classification**\n"
                f"* **Threshold:** The model uses an anomaly score threshold of **{image_threshold:.4f}**, "
                f"which represents the 95th percentile of normal validation images.\n"
                f"* **Precision:** At this threshold, the model achieves a Precision of "
                f"**{image_precision * 100:.1f}%**. This means out of all components flagged as defective, "
                f"{image_precision * 100:.1f}% are truly defective (minimal false alarms/wasted parts).\n"
                f"* **Recall:** The model achieves a Recall of **{image_recall * 100:.1f}%**, meaning it successfully "
                f"catches {image_recall * 100:.1f}% of all actual defective components on the line."
            )

        with col_pix:
            _display_level_metrics("Pixel-Level (Localization)", pix_metrics, level_type="pixel")
            aupimo_score = float(pix_metrics.get("aupimo_score", pix_metrics.get("aupimo", results.get("aupimo", 0.0))))
            fpr_lower_bound = float(pix_metrics.get("fpr_lower_bound", 1e-05))
            fpr_upper_bound = float(pix_metrics.get("fpr_upper_bound", 1e-04))

            st.info(
                f"**Pixel-Level Localization (AUPIMO)**\n"
                f"AUPIMO (Area Under the Per-Image Overlap) evaluates how well the model localizes "
                f"defects across a strictly controlled False Positive Rate (FPR) range "
                f"(from {fpr_lower_bound} to {fpr_upper_bound}).\n\n"
                f"* **Overall Score:** **{aupimo_score:.4f}** is the normalized area under the per-anomalous-image "
                f"recall curves across that logarithmic shared-FPR interval. Higher is better."
            )

        st.divider()

        if "metrics_path" in img_metrics:
            render_evaluation_curves(img_metrics["metrics_path"])

        st.divider()

        if "metrics_path" in pix_metrics:
            render_evaluation_curves(pix_metrics["metrics_path"])
    else:
        st.subheader("Evaluation Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUROC (Image)", f"{results.get('auroc', 0.0):.4f}")
        m2.metric("AUPIMO Score", f"{results.get('aupimo', 0.0):.4f}")
        m3.metric("Accuracy", f"{results.get('accuracy', 0.0) * 100:.2f}%")
        m4.metric("Precision", f"{results.get('precision', 0.0):.4f}")
        m5.metric("Recall", f"{results.get('recall', 0.0):.4f}")


def _find_metric_files() -> list[str]:
    """Search for saved metric .npz files in default results directories.

    Returns:
        A list of metric file paths.
    """
    found_files: list[str] = []
    for search_dir in [Path("data/models"), Path("results"), Path("data/external")]:
        if search_dir.exists():
            found_files.extend(
                [str(p) for p in search_dir.rglob("*.npz") if not p.name.startswith(".") and ".trash" not in p.parts]
            )
    return sorted(found_files)
