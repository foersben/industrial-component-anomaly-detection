"""Streamlit UI app."""

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from app.pipelines.evaluation.visualization import render_evaluation_curves

BACKEND_URL = "http://127.0.0.1:8000"

# Suppress timm deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*timm.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*timm.*")


def make_api_request(endpoint: str, payload: dict[str, Any], timeout: int | None = 10) -> Any:
    """Helper to handle repetitive POST requests and error catching.

    Args:
        endpoint: The API endpoint to call.
        payload: The payload to send to the API.
        timeout: The timeout for the request.

    Returns:
        The response from the API.
    """
    try:
        res = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ReadTimeout:
        st.error("Request timed out. This process might take longer.")
    except requests.exceptions.ConnectionError:
        st.error("Backend unreachable. Ensure FastAPI server is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
    return None


def _display_metrics_row(metrics: dict[str, Any]) -> None:
    """Helper to render 3-column metric cards (Accuracy, Precision, Recall).

    Args:
        metrics: The metrics to display.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
    col2.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.2f}")


def _display_level_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Helper to render level-specific metrics (Image or Pixel).

    Args:
        title: The title of the metrics.
        metrics: The metrics to display.
    """
    st.subheader(title)
    if "auroc" in metrics:
        st.metric("AUROC", f"{metrics.get('auroc', 0.0):.4f}")

    if "f1_score" in metrics:
        st.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
    elif "aupimo" in metrics:
        st.metric("AUPIMO", f"{metrics.get('aupimo', 0.0):.4f}")

    st.caption(f"Saved: `{metrics.get('metrics_path', '')}`")


def _find_metric_files() -> list[str]:
    """Search for saved metric .npz files in default results directories.

    Returns:
        A list of metric file paths.
    """
    found_files: list[str] = []
    for search_dir in [Path("results"), Path("data/external")]:
        if search_dir.exists():
            found_files.extend([str(p) for p in search_dir.rglob("*.npz") if not p.name.startswith(".")])
    return sorted(found_files)


def _handle_theoretical_dummy_eval() -> None:
    """Render and execute the theoretical dummy evaluation form."""
    col1, col2 = st.columns(2)
    pixels = col1.number_input("Total Pixels", min_value=1000, value=1000000, step=100000)
    ratio = col2.slider("Anomaly Ratio", min_value=0.001, max_value=0.200, value=0.015, step=0.005)

    if not st.button("Run Theoretical Evaluation"):
        return

    payload = {"mode": "theoretical", "pixels": pixels, "anomaly_ratio": ratio}
    data = make_api_request("/api/pipelines/dummy", payload)
    if data:
        st.success(data.get("message", "Success"))
        st.metric("Dummy Accuracy", f"{data.get('accuracy', 0) * 100:.2f}%")


def _handle_real_dummy_eval() -> None:
    """Render and execute the real dataset dummy evaluation form."""
    data_root = st.text_input("Dataset Root Path", value="data/raw/mvtec_ad")
    category = st.text_input("Category", value="bottle")

    if not st.button("Run Real Dataset Evaluation"):
        return

    payload = {"mode": "real", "data_root": data_root, "category": category}
    data = make_api_request("/api/pipelines/dummy", payload, timeout=60)
    if not data:
        return

    st.success(data.get("message", "Success"))
    results = data.get("results", {})

    if "error" in results:
        st.error(results["error"])
        return

    st.text_area("Evaluation Summary Output", value=results.get("summary", ""), height=160)
    _display_metrics_row(results)


def render_dummy_evaluation_tab() -> None:
    """Render the dummy classifier evaluation tab."""
    st.header("Dummy Classifier (Accuracy Paradox)")
    st.markdown("Evaluate a dummy classifier predicting all normal pixels on synthetic or real data.")

    mode = st.radio("Evaluation Mode", options=["theoretical", "real"], horizontal=True)

    if mode == "theoretical":
        _handle_theoretical_dummy_eval()
    else:
        _handle_real_dummy_eval()


def render_baseline_patchcore_tab() -> None:
    """Render the Patchcore anomaly detection evaluation tab."""
    st.header("Patchcore Anomaly Detection & Evaluation")
    st.markdown("Run Patchcore model training and evaluation on MVTec AD dataset (Image & Pixel level).")

    data_root = st.text_input("Dataset Root Directory", value="data/raw/mvtec_ad", key="b_root")
    category = st.text_input("Category Name", value="bottle", key="b_cat")

    st.subheader("Model Configuration")
    c1, c2 = st.columns(2)
    backbone = c1.selectbox("Backbone", ["resnet18", "wide_resnet50_2"])
    coreset_ratio = c2.slider("Coreset Sampling Ratio", min_value=0.01, max_value=0.2, value=0.1, step=0.01)

    st.subheader("Preprocessing Options")
    use_mask = st.checkbox(
        "Apply Otsu+Canny Foreground Masking (zeros out background)", value=False, key="patchcore_mask"
    )
    use_clahe = st.checkbox("Apply CLAHE", value=False, key="patchcore_clahe")
    use_gaussian = st.checkbox("Apply Gaussian Blur", value=False, key="patchcore_gaussian")

    preprocessing_steps = []
    if use_mask:
        preprocessing_steps.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        preprocessing_steps.append({"name": "clahe", "params": {}})
    if use_gaussian:
        preprocessing_steps.append({"name": "gaussian_blur", "params": {}})

    run_heatmap = st.checkbox("Compute Anomaly Heatmaps for anomalous images", value=False)

    if not st.button("Run Patchcore Evaluation Pipeline"):
        return

    with st.spinner("Fitting model and evaluating Image & Pixel level metrics..."):
        payload = {
            "data_root": data_root,
            "category": category,
            "preprocessing_steps": preprocessing_steps,
            "backbone": backbone,
            "coreset_sampling_ratio": coreset_ratio,
            "run_heatmap": run_heatmap,
        }
        data = make_api_request("/api/pipelines/baseline", payload, timeout=300)

        if not data:
            return

        st.success(data.get("message", "Success"))
        results = data.get("results", {})

        if isinstance(results, dict):
            col_img, col_pix = st.columns(2)
            img_metrics = results.get("image_level", {})
            pix_metrics = results.get("pixel_level", {})

            with col_img:
                _display_level_metrics("Image-Level (Classification)", img_metrics)
                prec = img_metrics.get("precision", 0.0) * 100
                rec = img_metrics.get("recall", 0.0) * 100
                if prec > 0 or rec > 0:
                    st.info(
                        "Using a strict threshold calculated only on normal test images, the model successfully "
                        f"flagged **{rec:.1f}%** of the actual defects. When it fired an alarm, "
                        f"it was correct **{prec:.1f}%** of the time."
                    )
            with col_pix:
                _display_level_metrics("Pixel-Level (Localization)", pix_metrics)
                aupimo_thresh = pix_metrics.get("t_aupimo_min", 0.0)
                aupimo_score = pix_metrics.get("aupimo", 0.0)
                if aupimo_thresh > 0:
                    st.info(
                        f"The strict industrial False Positive Rate (1e-5) threshold limit was calculated as "
                        f"**{aupimo_thresh:.4f}**. The model must exceed this high threshold to flag a pixel "
                        f"without violating the FPR constraint.\n\n"
                        f"At this threshold, the model finds **{aupimo_score * 100:.2f}%** "
                        f"of the actual anomalous pixels, guaranteeing highly reliable defect localization "
                        f"with fewer than 1 false alarm per 100,000 normal pixels."
                    )

            st.divider()

            if "metrics_path" in img_metrics:
                render_evaluation_curves(img_metrics["metrics_path"])

            st.divider()

            if "metrics_path" in pix_metrics:
                render_evaluation_curves(pix_metrics["metrics_path"])

            _render_heatmap_explorer(results)
        else:
            st.text_area("Results Summary", value=str(results), height=180)


def render_autoencoder_tab() -> None:
    """Render the Convolutional Autoencoder baseline evaluation tab."""
    st.header("Convolutional Autoencoder Anomaly Detection")
    st.markdown(
        "Train a minimal Convolutional autoencoder on normal samples and "
        "evaluate reconstruction error on test anomalies."
    )

    col1, col2 = st.columns(2)
    data_root = col1.text_input("Dataset Root Directory", value="data/raw/mvtec_ad", key="ae_root")
    category = col2.text_input("Category Name", value="bottle", key="ae_cat")

    col_e, col_b, col_l, col_s = st.columns(4)
    epochs = col_e.number_input("Epochs", min_value=1, max_value=50, value=5, step=1)
    batch_size = col_b.number_input("Batch Size", min_value=1, max_value=64, value=16, step=4)
    latent_dim = col_l.number_input("Latent Dim", min_value=8, max_value=256, value=64, step=8)
    img_size = col_s.number_input("Image Size", min_value=32, max_value=128, value=64, step=16)

    if not st.button("Run Autoencoder Training & Evaluation"):
        return

    with st.spinner("Training Autoencoder and evaluating on multi-class anomalies..."):
        payload = {
            "data_root": data_root,
            "category": category,
            "epochs": epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "img_size": img_size,
        }
        data = make_api_request("/api/pipelines/autoencoder", payload, timeout=180)

        if not data:
            return

        st.success(data.get("message", "Success"))
        results = data.get("results", {})

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("AUROC", f"{results.get('auroc', 0.0):.4f}")
        col_m2.metric("Accuracy", f"{results.get('accuracy', 0.0) * 100:.2f}%")
        col_m3.metric("Precision", f"{results.get('precision', 0.0):.2f}")
        col_m4.metric("Recall", f"{results.get('recall', 0.0):.2f}")

        st.caption(
            f"Decision Threshold (95th percentile normal): `{results.get('threshold', 0.0):.6f}` | "
            f"Final Training Loss: `{results.get('final_train_loss', 0.0):.6f}`"
        )

        if report := results.get("report"):
            st.subheader("Classification Report")
            st.code(report, language="text")


def render_keras_cae_tab() -> None:
    """Render the state-of-the-art Keras CAE anomaly detection tab.

    This tab exposes the full research-grade pipeline:
    ELU activations, Masked Image Modeling, SSIM+MSE loss, AdamW optimiser,
    Top-K pooling, adaptive thresholds, AUPIMO pixel evaluation, and optional Reconstruction Error Heatmaps.
    """
    st.header("State-of-the-Art Keras CAE")
    st.markdown(
        "Trains a from-scratch Convolutional Autoencoder with **ELU activations**, "
        "**Masked Image Modeling**, combined **SSIM+MSE loss**, and **AdamW** optimiser. "
        "Evaluates using **Top-K pooling** and **AUPIMO** at industrially strict FPR bounds."
    )

    # ── Model Registry Table ──────────────────────────────────────────────────────
    st.subheader("Model Registry (Cached Models)")
    registry_path = Path("data/models/keras_cae")
    cached_models = []
    if registry_path.exists():
        for meta_file in registry_path.rglob("metadata.json"):
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                    cached_models.append(
                        {
                            "Category": meta.get("category"),
                            "Hash": meta.get("hash"),
                            "Img Size": meta.get("img_size"),
                            "Latent": meta.get("latent_dim"),
                            "Epochs": meta.get("epochs"),
                            "Batch": meta.get("batch_size"),
                            "Mask Ratio": meta.get("mask_ratio"),
                            "Created": meta.get("timestamp", "")[:19].replace("T", " "),
                        }
                    )
            except Exception:
                pass

    if cached_models:
        df_models = pd.DataFrame(cached_models)
        st.dataframe(df_models, width="stretch", hide_index=True)
        st.info(
            "If you run the pipeline with hyperparameters matching a cached model, "
            "it will instantly load without retraining."
        )
    else:
        st.caption("No cached models found in registry.")

    st.divider()

    col1, col2 = st.columns(2)
    data_root = col1.text_input("Dataset Root Directory", value="data/raw/mvtec_ad", key="kcae_root")
    category = col2.text_input("Category Name", value="bottle", key="kcae_cat")

    st.subheader("Training Hyperparameters")
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", min_value=1, max_value=100, value=20, step=5)
    latent_channels = c2.number_input("Latent Channels", min_value=8, max_value=256, value=32, step=8)
    img_size = c3.number_input("Image Size", min_value=64, max_value=256, value=128, step=16)
    batch_size = c4.number_input("Batch Size", min_value=4, max_value=64, value=16, step=4)

    with st.expander("Advanced Hyperparameters"):
        ac1, ac2, ac3 = st.columns(3)
        mask_ratio = ac1.slider("Mask Ratio (MIM)", 0.0, 0.75, 0.25, 0.05)
        threshold_method = ac2.selectbox("Threshold Method", ["quantile", "mahalanobis"])
        k_fraction = ac3.number_input(
            "Top-K Fraction", min_value=0.001, max_value=0.050, value=0.002, step=0.001, format="%.3f"
        )

    st.subheader("Preprocessing Options")
    use_seg = st.checkbox("Apply Otsu+Canny Foreground Masking (BGRP-G)", value=True, key="kcae_mask")
    use_clahe = st.checkbox("Apply CLAHE", value=False, key="kcae_clahe")
    use_gaussian = st.checkbox("Apply Gaussian Blur", value=False, key="kcae_gaussian")

    preprocessing_steps = []
    if use_seg:
        preprocessing_steps.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        preprocessing_steps.append({"name": "clahe", "params": {}})
    if use_gaussian:
        preprocessing_steps.append({"name": "gaussian_blur", "params": {}})

    st.subheader("Execution")
    force_retrain = st.checkbox("Force Retrain Model (bypass cache even if hyperparameters match)", value=False)

    if not st.button("Run Keras CAE Pipeline"):
        return

    with st.spinner("Training Keras CAE and evaluating... (may take several minutes)"):
        payload = {
            "data_root": data_root,
            "category": category,
            "img_size": img_size,
            "latent_channels": latent_channels,
            "epochs": epochs,
            "batch_size": batch_size,
            "mask_ratio": mask_ratio,
            "threshold_method": threshold_method,
            "k_fraction": k_fraction,
            "preprocessing_steps": preprocessing_steps,
            "run_heatmap": True,  # Automatically compute heatmaps
            "force_retrain": force_retrain,
        }
        data = make_api_request("/api/pipelines/keras_cae", payload, timeout=None)

    if not data:
        return

    st.success("Pipeline completed successfully!")
    results = data.get("results", {})

    # ── Metrics ──────────────────────────────────────────────────────────────────
    if "image_level" in results and "pixel_level" in results:
        col_img, col_pix = st.columns(2)
        img_metrics = results.get("image_level", {})
        pix_metrics = results.get("pixel_level", {})

        with col_img:
            _display_level_metrics("Image-Level (Classification)", img_metrics)
            total = results.get("total_test_images", 0)
            prec = results.get("precision", 0.0) * 100
            rec = results.get("recall", 0.0) * 100
            st.info(
                f"Out of {total} test components, the model successfully flagged **{rec:.1f}%** of the actual defects. "
                f"When it fired an alarm, it was correct **{prec:.1f}%** of the time."
            )

        with col_pix:
            _display_level_metrics("Pixel-Level (Localization)", pix_metrics)
            aupimo_thresh = pix_metrics.get("t_aupimo_min", 0.0)
            aupimo_score = pix_metrics.get("aupimo", 0.0)
            if aupimo_thresh > 0:
                st.info(
                    f"The strict industrial False Positive Rate (1e-5) threshold limit was calculated as "
                    f"**{aupimo_thresh:.4f}**. The model must exceed this high threshold to flag a pixel "
                    f"without violating the FPR constraint.\n\n"
                    f"At this threshold, the model finds **{aupimo_score * 100:.2f}%** "
                    f"of the actual anomalous pixels, guaranteeing highly reliable defect localization "
                    f"with fewer than 1 false alarm per 100,000 normal pixels."
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
        m2.metric("AUPIMO (Pixel)", f"{results.get('aupimo', 0.0):.4f}")
        m3.metric("Accuracy", f"{results.get('accuracy', 0.0) * 100:.2f}%")
        m4.metric("Precision", f"{results.get('precision', 0.0):.4f}")
        m5.metric("Recall", f"{results.get('recall', 0.0):.4f}")

    st.caption(
        f"Adaptive Threshold: `{results.get('threshold', 0.0):.6f}` | "
        f"Final Training Loss: `{results.get('final_train_loss', 0.0):.6f}`"
    )

    # ── Loss Curve ───────────────────────────────────────────────────────────────
    if loss_history := results.get("loss_history"):
        if isinstance(loss_history, dict):
            clean_history = {k: v for k, v in loss_history.items() if isinstance(v, list) and len(v) > 0}
            if clean_history:
                st.subheader("Training Loss Curve")
                df_loss = pd.DataFrame({k: pd.Series(v) for k, v in clean_history.items()})
                st.line_chart(df_loss)

    # ── Heatmap Explorer ───────────────────────────────────────────────
    _render_heatmap_explorer(results)


def _render_heatmap_explorer(results: dict[str, Any]) -> None:
    """Render the Reconstruction Error Heatmap explorer below the evaluation results.

    Shows a gallery of error heatmap overlays for every detected anomalous image.
    The user can trigger computation by clicking a single button; results are
    shown in a responsive grid so all anomalous images are visible at once.
    """
    anomalous_indices: list[int] = results.get("anomalous_indices", [])
    if not anomalous_indices:
        return

    st.divider()
    with st.expander(
        f"Anomaly Heatmap Explorer — {len(anomalous_indices)} anomalous image(s) found",
        expanded=True,
    ):
        st.markdown("""
        The model processes test images and generates a pixel-wise **Anomaly Map**. High values indicate that the model
        believes those specific pixels are defective based on what it learned from normal components.

        **Red / warm** — high anomaly score → likely a defect
        **Blue / cool** — low anomaly score → looks normal to the model
        """)

        # Check if we have Heatmap results from the pipeline run
        existing_overlays: dict[Any, Any] = results.get("heatmap_overlays", {})

        if existing_overlays:
            _render_heatmap_gallery(existing_overlays, anomalous_indices)
        else:
            st.info("No heatmaps were computed.")


def _render_heatmap_gallery(overlays: dict[Any, Any], anomalous_indices: list[int]) -> None:
    """Render a grid of Error heatmap overlays for all anomalous images.

    Displays two rectangular areas (grids) side-by-side:
    - Left Grid: Original image + Model Prediction Heatmap
    - Right Grid: Ground Truth Mask + Model Prediction Heatmap
    """
    import numpy as np

    st.success(f"Heatmaps computed for {len(overlays)} image(s).")

    indices = [i for i in anomalous_indices if str(i) in overlays or i in overlays]
    if not indices:
        return

    main_col1, main_col2 = st.columns(2)
    cols_per_row = 2

    with main_col1:
        st.subheader("Prediction Heatmap")
        for row_start in range(0, len(indices), cols_per_row):
            row_indices = indices[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_indices))
            for col, idx in zip(cols, row_indices, strict=False):
                overlay_data = overlays.get(idx) or overlays.get(str(idx))
                if isinstance(overlay_data, dict) and "heatmap" in overlay_data:
                    hm_arr = np.array(overlay_data["heatmap"], dtype=np.uint8)
                    col.image(hm_arr, caption=f"Image #{idx}", width="stretch")
                elif isinstance(overlay_data, list):
                    # Fallback
                    col.image(np.array(overlay_data, dtype=np.uint8), caption=f"Image #{idx}")

    with main_col2:
        st.subheader("Ground Truth + Heatmap")
        for row_start in range(0, len(indices), cols_per_row):
            row_indices = indices[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_indices))
            for col, idx in zip(cols, row_indices, strict=False):
                overlay_data = overlays.get(idx) or overlays.get(str(idx))
                if isinstance(overlay_data, dict):
                    if "gt_and_heatmap" in overlay_data:
                        gt_hm_arr = np.array(overlay_data["gt_and_heatmap"], dtype=np.uint8)
                        col.image(gt_hm_arr, caption=f"Image #{idx}", width="stretch")
                    elif "gt_overlay" in overlay_data:
                        # Fallback for previous run format
                        gt_hm_arr = np.array(overlay_data["gt_overlay"], dtype=np.uint8)
                        col.image(gt_hm_arr, caption=f"Image #{idx}", width="stretch")

    st.caption(
        "Heatmaps derived directly from the per-pixel Mean Squared Error between the original "
        "image and the autoencoder's reconstruction, slightly smoothed with a Gaussian filter."
    )


def render_evaluation_guide_tab() -> None:
    """Render the evaluation guide for beginners."""
    st.header("Evaluation Guide & Metrics Interpretation")
    st.markdown("""
    When evaluating an anomaly detection model in an industrial setting, classical metrics like **Accuracy**
    can be incredibly misleading. This guide explains exactly how to read the metrics generated by the pipelines.

    ---

    ### 1. The Accuracy Paradox
    In industrial manufacturing, anomalies are rare (e.g., 99% of components are good, 1% are defective).
    If a "dummy" model blindly guesses that **every** component is good, it will achieve **99% Accuracy**.
    - **Why it's bad:** The model entirely missed the 1% of defective parts, rendering it useless for quality control.
    - **Takeaway:** Never trust Accuracy alone when evaluating an anomaly detection model.

    ---

    ### 2. Precision and Recall
    These metrics give you a clearer picture than Accuracy, especially when dealing with rare defects.
    - **Precision:** Out of all the components the model *flagged* as defective, how many were *actually* defective?
      - *High Precision = Very few false alarms.*
    - **Recall:** Out of all the *actually* defective components, how many did the model successfully *flag*?
      - *High Recall = Very few escaped defects (missed anomalies).*

    In industry, we usually prioritize **Recall** (we cannot let a broken part reach a customer).
    However, if Precision is too low, the operator will be overwhelmed with false alarms and might turn the system off.

    ---

    ### 3. AUROC (Area Under the Receiver Operating Characteristic Curve)
    *Used for: Image-Level Evaluation (Is the entire component defective or not?)*

    The AUROC score measures how well the model separates the "Good" scores from the "Defective" scores
    across *all possible threshold values*.

    **⚠️ The Imbalance Trap:**
    In real factories, 99.9% of components are good, and 0.1% are bad. In such extreme class imbalances, AUROC
    is notoriously deceptive. A model can achieve a 0.95 AUROC but still have a terrible Precision (firing false
    alarms constantly) because the massive number of True Negatives drowns out the False Positives.

    **Why do we still use it?**
    1. **Benchmarking:** Academic test sets (like MVTec AD) artificially balance the test images (e.g.,
       20 good, 60 bad), making Image-Level AUROC acceptable for literature comparison.
    2. **Top-K Pooling:** We do not use the maximum pixel error as the image score. Max-pooling fails because
       a single dead camera pixel causes a false positive. We use **Top-K Pooling** (averaging the worst ~30
       pixels), which ensures the image score reflects a *cluster* of defects, making the Image-Level AUROC
       much more robust to noise.

    *For real-world deployment, rely on **F1-Score** or **PR-AUC** (Precision-Recall AUC) instead of AUROC.*

    ### 4. AUPIMO (Area Under Per-Image Overlap)
    *Used for: Pixel-Level Evaluation (Where exactly is the defect on the component?)*

    Standard pixel evaluation metrics (like PRO-score) evaluate the model up to a 30% False Positive Rate (FPR).
    In an industrial factory making 10,000 parts a day, a 30% false alarm rate means throwing away
    3,000 perfect components. This is unacceptable.

    **AUPIMO** measures how well the model isolates defective pixels under extremely strict,
    realistic industrial constraints (FPR between 1e-5 and 1e-4).
    - **FPR 1e-5:** Equals 1 false alarm per 100,000 normal parts.
    - **What a Good Score Looks Like:** Because the FPR restriction is so brutal, AUPIMO scores are
      naturally lower than AUROC. A score above `0.20` is considered excellent, and anything above `0.05`
      is generally highly usable for precise localization.
    """)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
    st.title("Industrial Component Anomaly Detection Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Evaluation Guide",
            "Dummy Classifier Evaluation",
            "Convolutional Autoencoder Evaluation",
            "Patchcore Evaluation (Image & Pixel Level)",
            "Keras CAE (State-of-the-Art)",
        ]
    )

    with tab1:
        render_evaluation_guide_tab()
    with tab2:
        render_dummy_evaluation_tab()
    with tab3:
        render_autoencoder_tab()
    with tab4:
        render_baseline_patchcore_tab()
    with tab5:
        render_keras_cae_tab()


if __name__ == "__main__":
    main()
