"""Streamlit UI app."""

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from app.pipelines.evaluation.visualization import render_evaluation_curves
from app.pipelines.modelling.keras_cae.cae_pipeline import (
    delete_cached_model,
    list_trashed_models,
    purge_trash,
    restore_cached_model,
)

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


def _render_model_run_overview(results: dict[str, Any], model_type: str = "cae") -> None:
    """Render a comprehensive overview of active preprocessing, hyperparameters, and dataset split.

    Args:
        results: Results dictionary returned from the pipeline or model evaluation.
        model_type: Type of model evaluated ('cae' or 'patchcore').
    """
    with st.expander("📋 Model Run Overview (Preprocessing, Hyperparameters & Dataset Split)", expanded=True):
        col_prep, col_hp, col_split = st.columns(3)

        # ── 1. Preprocessing Configuration ──
        with col_prep:
            st.markdown("#### 🔧 Preprocessing")
            prep_steps = results.get("preprocessing_steps")
            if prep_steps is None and "metadata" in results and isinstance(results["metadata"], dict):
                prep_steps = results["metadata"].get("preprocessing_steps")

            if prep_steps is not None and isinstance(prep_steps, list):
                if len(prep_steps) == 0:
                    st.markdown("⚪ **None** *(Raw unmodified images)*")
                else:
                    for s in prep_steps:
                        name = str(s.get("name", "Unknown Step"))
                        if name == "foreground_mask":
                            st.markdown("🟢 **Foreground Mask** *(Otsu + Canny)*")
                        elif name == "clahe":
                            st.markdown("🟢 **CLAHE** *(Contrast Equalization)*")
                        elif name == "gaussian_blur":
                            st.markdown("🟢 **Gaussian Blur** *(Denoising)*")
                        else:
                            st.markdown(f"🟢 **`{name}`**")
            else:
                st.info("⚠️ *Preprocessing configuration was not recorded with this legacy model run.*")

        # ── 2. Hyperparameters ──
        with col_hp:
            st.markdown("#### ⚙️ Hyperparameters")
            if model_type == "cae":
                hp = results.get("hyperparameters") or (
                    results.get("metadata") if isinstance(results.get("metadata"), dict) else None
                )
                if hp and isinstance(hp, dict):
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
                else:
                    st.info("⚠️ *Hyperparameter details were not recorded with this legacy model run.*")
            else:
                hp = results.get("hyperparameters", {})
                if hp and isinstance(hp, dict) and len(hp) > 0:
                    st.markdown(f"- **Category:** `{results.get('category', 'N/A')}`")
                    st.markdown(f"- **Backbone:** `{hp.get('backbone', 'resnet18')}`")
                    st.markdown(f"- **Coreset Ratio:** `{hp.get('coreset_sampling_ratio', 0.1)}`")
                    st.markdown(f"- **FPR Limit:** `{hp.get('fpr_limit', 1e-4)}`")
                    st.markdown(f"- **Batch Size:** `{hp.get('train_batch_size', 16)}`")
                else:
                    st.info("⚠️ *Hyperparameters were not recorded with this legacy model run.*")

        # ── 3. Dataset Split ──
        with col_split:
            st.markdown("#### 📊 Dataset Partition Split")
            split = results.get("dataset_split") or (
                results.get("metadata", {}).get("dataset_split") if isinstance(results.get("metadata"), dict) else None
            )
            if split and isinstance(split, dict) and len(split) > 0:
                if model_type == "cae":
                    st.markdown(f"- **Train (Normal):** `{split.get('train_normal', 'N/A')}`")
                    st.markdown(f"- **Validation (Normal, 15%):** `{split.get('val_normal', 'N/A')}`")
                    test_tot = split.get("test_total", "N/A")
                    test_norm = split.get("test_normal")
                    test_anom = split.get("test_anomalous")
                    if test_norm is not None and test_anom is not None:
                        st.markdown(f"- **Test Total:** `{test_tot}` *({test_norm} normal, {test_anom} anomalous)*")
                    else:
                        st.markdown(f"- **Test Total:** `{test_tot}`")
                else:
                    st.markdown(f"- **Train (Normal):** `{split.get('train_normal', 'N/A')}`")
                    st.markdown(f"- **Test Total:** `{split.get('test_total', 'N/A')}`")
            else:
                st.info("⚠️ *Dataset partition sample counts were not recorded with this legacy model run.*")
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
            threshold_limit = float(pix_metrics.get("threshold_limit", pix_metrics.get("t_aupimo_min", 0.0)))
            tpr_at_limit = float(
                pix_metrics.get("tpr_at_limit", pix_metrics.get("aupimo_recall", pix_metrics.get("aupimo", 0.0)))
            )
            fpr_denom = fpr_lower_bound if fpr_lower_bound > 0 else 1e-5

            st.info(
                f"**Pixel-Level Localization (AUPIMO)**\n"
                f"AUPIMO (Area Under the Per-Image Measurement Overlap) evaluates how well the model localizes "
                f"defects across a strictly controlled False Positive Rate (FPR) range "
                f"(from {fpr_lower_bound} to {fpr_upper_bound}).\n\n"
                f"* **Overall Score:** An AUPIMO score of **{aupimo_score:.4f}** means that, on average across "
                f"this strict trust range, the model successfully highlights **{aupimo_score * 100:.1f}%** "
                f"of the actual defective pixel area.\n"
                f"* **Industrial Threshold Limit:** To guarantee highly reliable defect localization with fewer than "
                f"1 false alarm per {int(1 / fpr_denom):,} normal pixels, the model calculates a strict "
                f"binarization threshold of **{threshold_limit:.4f}**.\n"
                f"* **Reliability:** If deployed at this strict threshold, the model catches "
                f"**{tpr_at_limit * 100:.2f}%** of the actual anomalous pixels, meaning any pixel flagged "
                f"is guaranteed to be a defect with >99.99% confidence."
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


def _handle_theoretical_dummy_eval() -> None:
    """Render and execute the theoretical dummy evaluation form."""
    col1, col2 = st.columns(2)
    pixels = col1.number_input("Total Pixels", value=1_000_000, step=100_000)
    ratio = col2.slider("Synthetic Anomaly Ratio", min_value=0.001, max_value=0.05, value=0.015, step=0.001)

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
    """Render the Patchcore anomaly detection evaluation tab with registry and caching support."""
    st.header("Patchcore Anomaly Detection & Evaluation")
    st.markdown(
        "Run Patchcore feature-memory-bank anomaly detection and evaluation on MVTec AD dataset "
        "(Image & Pixel level) with automated caching, model versioning, and soft-delete recovery."
    )

    # ── Model Registry Table ──────────────────────────────────────────────────────
    st.subheader("Model Registry (Cached Patchcore Models)")
    registry_path = Path("data/models/patchcore")
    cached_models: list[dict[str, Any]] = []
    if registry_path.exists():
        for meta_file in registry_path.rglob("metadata.json"):
            if ".trash" in meta_file.parts:
                continue
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                    ts_str = meta.get("timestamp", "")
                    created_display = ts_str[:19].replace("T", " ") if ts_str else "Unknown"
                    prep_list = meta.get("preprocessing_steps", [])
                    prep_names = [s.get("name", "") for s in prep_list] if isinstance(prep_list, list) else []
                    prep_display = ", ".join(prep_names) if prep_names else "None"
                    cached_models.append(
                        {
                            "Category": meta.get("category", "unknown"),
                            "Hash": meta.get("hash", meta_file.parent.name),
                            "Backbone": meta.get("backbone", "resnet18"),
                            "Coreset Ratio": meta.get("coreset_sampling_ratio", 0.1),
                            "Preprocessing": prep_display,
                            "Created": created_display,
                            "_raw_timestamp": ts_str,
                            "_raw_preprocessing_steps": prep_list,
                        }
                    )
            except Exception:
                pass

    selected_model_hash: str | None = None
    selected_model_meta: dict[str, Any] | None = None
    selected_model_hashes: list[str] = []
    load_selected_clicked = False

    if cached_models:
        cached_models.sort(key=lambda x: str(x.get("_raw_timestamp", "")), reverse=True)
        display_models = [{k: v for k, v in m.items() if not k.startswith("_")} for m in cached_models]
        df_models = pd.DataFrame(display_models)

        selection = st.dataframe(
            df_models,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="patchcore_registry_selection",
        )

        selected_rows: list[int] = []
        if selection is not None:
            if isinstance(selection, dict):
                selected_rows = selection.get("selection", {}).get("rows", [])
            else:
                sel_attr = getattr(selection, "selection", None)
                if isinstance(sel_attr, dict):
                    selected_rows = sel_attr.get("rows", [])
                elif hasattr(sel_attr, "rows"):
                    selected_rows = getattr(sel_attr, "rows", [])

        selected_model_metas = [cached_models[r] for r in selected_rows if 0 <= r < len(cached_models)]
        selected_model_hashes = [str(m.get("Hash")) for m in selected_model_metas]

        if selected_model_hashes:
            if len(selected_model_hashes) == 1:
                selected_model_meta = selected_model_metas[0]
                selected_model_hash = selected_model_hashes[0]

                if st.session_state.get("_last_patchcore_selected_hash") != selected_model_hash:
                    st.session_state["_last_patchcore_selected_hash"] = selected_model_hash
                    st.session_state["b_cat"] = str(selected_model_meta.get("Category", "bottle"))
                    st.session_state["b_backbone"] = str(selected_model_meta.get("Backbone", "resnet18"))
                    st.session_state["b_coreset_ratio"] = float(selected_model_meta.get("Coreset Ratio", 0.1))

                    raw_prep = selected_model_meta.get("_raw_preprocessing_steps", [])
                    if isinstance(raw_prep, list):
                        st.session_state["patchcore_mask"] = any(s.get("name") == "foreground_mask" for s in raw_prep)
                        st.session_state["patchcore_clahe"] = any(s.get("name") == "clahe" for s in raw_prep)
                        st.session_state["patchcore_gaussian"] = any(s.get("name") == "gaussian_blur" for s in raw_prep)

                st.success(
                    f"Selected cached Patchcore model: **`{selected_model_hash}`** ("
                    f"Category: `{selected_model_meta.get('Category')}`, "
                    f"Backbone: `{selected_model_meta.get('Backbone')}`, "
                    f"Coreset Ratio: `{selected_model_meta.get('Coreset Ratio')}`, "
                    f"Preprocessing: `{selected_model_meta.get('Preprocessing')}`, "
                    f"Created: `{selected_model_meta.get('Created')}`)"
                )
                col_load, col_del, _ = st.columns([2, 1, 3])
                load_selected_clicked = col_load.button(
                    f"⚡ Load & Evaluate Model `{selected_model_hash}`",
                    type="primary",
                    key="btn_load_patchcore_selected",
                )
                with col_del.popover("🗑️ Delete Model", help=f"Move Patchcore model {selected_model_hash} to Trash"):
                    st.warning(f"Move Patchcore model `{selected_model_hash}` to Trash (can be restored)?")
                    if st.button("Move to Trash", type="primary", key="btn_confirm_delete_patchcore_single"):
                        if delete_cached_model(selected_model_hash, registry_base=registry_path, soft_delete=True):
                            st.session_state.pop("_last_patchcore_selected_hash", None)
                            st.success(f"Patchcore model `{selected_model_hash}` moved to Trash (reversible).")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete Patchcore model `{selected_model_hash}`.")
            else:
                st.warning(f"Selected **{len(selected_model_hashes)} models**: `{', '.join(selected_model_hashes)}`")
                col_del_multi, _ = st.columns([2, 4])
                with col_del_multi.popover(
                    f"🗑️ Delete {len(selected_model_hashes)} Models",
                    help=f"Move {len(selected_model_hashes)} selected Patchcore models to Trash",
                ):
                    st.warning(f"Move **{len(selected_model_hashes)}** selected Patchcore models to Trash?")
                    st.markdown("\n".join(f"- `{h}`" for h in selected_model_hashes))
                    if st.button(
                        f"Move to Trash ({len(selected_model_hashes)} models)",
                        type="primary",
                        key="btn_confirm_delete_patchcore_multi",
                    ):
                        deleted_cnt = 0
                        for h in selected_model_hashes:
                            if delete_cached_model(h, registry_base=registry_path, soft_delete=True):
                                deleted_cnt += 1
                        st.session_state.pop("_last_patchcore_selected_hash", None)
                        st.success(f"Moved {deleted_cnt} Patchcore model(s) to Trash (reversible).")
                        st.rerun()
        else:
            st.info(
                "💡 **Interactive Patchcore Registry:** Click on any row above to select, load, or delete that "
                "cached model. The pipeline always loads the newest matching cached model automatically when available."
            )
    else:
        st.caption("No cached Patchcore models found in registry.")

    # ── Trash & Restoration Section ───────────────────────────────────────────────
    trashed_models = list_trashed_models(registry_base=registry_path)
    if trashed_models:
        with st.expander(f"🗑️ Trash / Recently Deleted ({len(trashed_models)} models)", expanded=False):
            st.caption("Soft-deleted Patchcore models are safely preserved here and can be restored at any time.")
            trashed_display = []
            for tm in trashed_models:
                ts_raw = tm.get("timestamp", "")
                trashed_display.append(
                    {
                        "Category": tm.get("category", "unknown"),
                        "Hash": tm.get("hash", "unknown"),
                        "Backbone": tm.get("backbone", "resnet18"),
                        "Coreset Ratio": tm.get("coreset_sampling_ratio", 0.1),
                        "Created": ts_raw[:19].replace("T", " ") if ts_raw else "Unknown",
                    }
                )
            df_trashed = pd.DataFrame(trashed_display)
            trash_selection = st.dataframe(
                df_trashed,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="patchcore_trash_selection",
            )

            trashed_sel_rows: list[int] = []
            if trash_selection is not None:
                if isinstance(trash_selection, dict):
                    trashed_sel_rows = trash_selection.get("selection", {}).get("rows", [])
                else:
                    sel_attr = getattr(trash_selection, "selection", None)
                    if isinstance(sel_attr, dict):
                        trashed_sel_rows = sel_attr.get("rows", [])
                    elif hasattr(sel_attr, "rows"):
                        trashed_sel_rows = getattr(sel_attr, "rows", [])

            trashed_selected_hashes = [
                str(trashed_models[r].get("hash")) for r in trashed_sel_rows if 0 <= r < len(trashed_models)
            ]

            col_rest, col_purge, _ = st.columns([2, 2, 4])
            if trashed_selected_hashes:
                if col_rest.button(
                    f"♻️ Restore Selected ({len(trashed_selected_hashes)})",
                    type="primary",
                    key="btn_restore_patchcore_selected",
                ):
                    restored_cnt = 0
                    for th in trashed_selected_hashes:
                        if restore_cached_model(th, registry_base=registry_path):
                            restored_cnt += 1
                    st.success(f"Restored {restored_cnt} Patchcore model(s) back to registry!")
                    st.rerun()
            else:
                if col_rest.button("♻️ Restore All Trashed", key="btn_restore_patchcore_all"):
                    restored_cnt = 0
                    for tm in trashed_models:
                        th_val = str(tm.get("hash", ""))
                        if th_val and restore_cached_model(th_val, registry_base=registry_path):
                            restored_cnt += 1
                    st.success(f"Restored all {restored_cnt} Patchcore model(s) back to registry!")
                    st.rerun()

            with col_purge.popover(
                "⚠️ Empty Trash (Permanent)",
                help="Permanently delete all Patchcore models in Trash",
            ):
                st.error("Are you sure you want to permanently delete these models? This cannot be undone.")
                if st.button("Yes, Empty Trash", type="primary", key="btn_purge_patchcore_trash"):
                    cnt = purge_trash(registry_base=registry_path)
                    st.success(f"Permanently purged {cnt} Patchcore model(s) from disk.")
                    st.rerun()

    st.divider()

    st.session_state.setdefault("b_root", "data/raw/mvtec_ad")
    data_root = st.text_input("Dataset Root Directory", key="b_root")

    st.session_state.setdefault("b_cat", "bottle")
    st.session_state.setdefault("b_backbone", "resnet18")
    st.session_state.setdefault("b_feature_layers", "l2_l3")
    st.session_state.setdefault("b_coreset_ratio", 0.1)
    st.session_state.setdefault("b_num_neighbors", 9)
    st.session_state.setdefault("patchcore_mask", False)
    st.session_state.setdefault("patchcore_clahe", False)
    st.session_state.setdefault("patchcore_gaussian", False)

    def load_patchcore_optuna_defaults() -> None:
        selected_cat = st.session_state.b_cat
        registry_path = Path("data/hyperparameters/patchcore_best.json")
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)

            if selected_cat in registry:
                cfg = registry[selected_cat]
                if "preprocessing" not in cfg:
                    prep = cfg
                    hp = cfg
                else:
                    prep = cfg.get("preprocessing", {})
                    hp = cfg.get("model_hyperparameters", {})

                if "backbone" in hp:
                    st.session_state.b_backbone = hp["backbone"]
                if "feature_layers" in hp:
                    layers = hp["feature_layers"]
                    st.session_state.b_feature_layers = "l2_l3_l4" if "layer4" in layers else "l2_l3"
                if "coreset_sampling_ratio" in hp:
                    st.session_state.b_coreset_ratio = float(hp["coreset_sampling_ratio"])
                if "num_neighbors" in hp:
                    st.session_state.b_num_neighbors = int(hp["num_neighbors"])

                if "use_clahe" in prep:
                    st.session_state.patchcore_clahe = prep["use_clahe"]
                if "use_gaussian_blur" in prep:
                    st.session_state.patchcore_gaussian = prep["use_gaussian_blur"]
                if "use_foreground_mask" in prep:
                    st.session_state.patchcore_mask = prep["use_foreground_mask"]

    mvtec_categories = [
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
    ]
    category = st.selectbox(
        "Category Name", options=mvtec_categories, key="b_cat", on_change=load_patchcore_optuna_defaults
    )

    st.subheader("Model Configuration")
    c1, c2, c3, c4 = st.columns(4)
    backbone = c1.selectbox("Backbone", ["resnet18", "wide_resnet50_2"], key="b_backbone")
    feature_layers_str = c2.selectbox("Feature Layers", ["l2_l3", "l2_l3_l4"], key="b_feature_layers")
    feature_layers = ["layer2", "layer3", "layer4"] if feature_layers_str == "l2_l3_l4" else ["layer2", "layer3"]
    coreset_ratio = c3.slider(
        "Coreset Sampling Ratio", min_value=0.001, max_value=0.2, step=0.005, format="%.3f", key="b_coreset_ratio"
    )
    num_neighbors = c4.slider("Nearest Neighbors", min_value=1, max_value=20, step=1, key="b_num_neighbors")

    st.subheader("Preprocessing Options")
    use_mask = st.checkbox("Apply Otsu+Canny Foreground Masking (zeros out background)", key="patchcore_mask")
    st.session_state.setdefault("patchcore_clahe", False)
    use_clahe = st.checkbox("Apply CLAHE", key="patchcore_clahe")
    st.session_state.setdefault("patchcore_gaussian", False)
    use_gaussian = st.checkbox("Apply Gaussian Blur", key="patchcore_gaussian")

    preprocessing_steps = []
    if use_mask:
        preprocessing_steps.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        preprocessing_steps.append({"name": "clahe", "params": {}})
    if use_gaussian:
        preprocessing_steps.append({"name": "gaussian_blur", "params": {}})

    st.session_state.setdefault("b_heatmap", False)
    run_heatmap = st.checkbox("Compute Anomaly Heatmaps for anomalous images", key="b_heatmap")

    force_retrain = st.checkbox(
        "Force Retrain (Ignore Cache)",
        value=False,
        key="patchcore_force_retrain",
        help="Check this to force re-running model training even if an identical cached run exists in the registry.",
    )

    col_btn, _ = st.columns([2, 4])
    run_clicked = col_btn.button("Run Patchcore Evaluation Pipeline", key="btn_run_patchcore")

    if not (run_clicked or load_selected_clicked):
        return

    active_hash = selected_model_hash if load_selected_clicked else None
    active_force_retrain = False if load_selected_clicked else force_retrain
    active_prep = (
        selected_model_meta.get("_raw_preprocessing_steps", preprocessing_steps)
        if load_selected_clicked and selected_model_meta
        else preprocessing_steps
    )

    spinner_msg = (
        f"Loading cached Patchcore model `{active_hash}` and evaluating..."
        if load_selected_clicked
        else "Fitting Patchcore model and evaluating Image & Pixel level metrics..."
    )

    with st.spinner(spinner_msg):
        payload = {
            "data_root": data_root,
            "category": category,
            "backbone": backbone,
            "feature_layers": feature_layers,
            "coreset_sampling_ratio": coreset_ratio,
            "num_neighbors": num_neighbors,
            "preprocessing_steps": active_prep,
            "run_heatmap": run_heatmap,
            "force_retrain": active_force_retrain,
            "model_hash": active_hash,
        }
        data = make_api_request("/api/pipelines/baseline", payload, timeout=300)

        if not data:
            return

        st.success(data.get("message", "Success"))
        results = data.get("results", {})

        if isinstance(results, dict):
            _render_evaluation_summary(results, model_type="patchcore")

            # Additional Parity for PatchCore
            pixel_metrics = results.get("pixel_level", {})
            metrics_path = pixel_metrics.get("metrics_path")
            if metrics_path:
                render_evaluation_curves(metrics_path)

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
    st.session_state.setdefault("ae_root", "data/raw/mvtec_ad")
    data_root = col1.text_input("Dataset Root Directory", key="ae_root")

    mvtec_categories = [
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
    ]
    st.session_state.setdefault("ae_cat", "bottle")
    category = col2.selectbox("Category Name", options=mvtec_categories, key="ae_cat")

    col_e, col_b, col_l, col_s = st.columns(4)
    st.session_state.setdefault("ae_epochs", 5)
    epochs = col_e.number_input("Epochs", min_value=1, max_value=50, step=1, key="ae_epochs")
    st.session_state.setdefault("ae_batch", 16)
    batch_size = col_b.number_input("Batch Size", min_value=1, max_value=64, step=4, key="ae_batch")
    st.session_state.setdefault("ae_latent", 64)
    latent_dim = col_l.number_input("Latent Dim", min_value=8, max_value=256, step=8, key="ae_latent")
    st.session_state.setdefault("ae_size", 64)
    img_size = col_s.number_input("Image Size", min_value=32, max_value=128, step=16, key="ae_size")

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
    cached_models: list[dict[str, Any]] = []
    if registry_path.exists():
        for meta_file in registry_path.rglob("metadata.json"):
            if ".trash" in meta_file.parts:
                continue
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                    ts_str = meta.get("timestamp", "")
                    created_display = ts_str[:19].replace("T", " ") if ts_str else "Unknown"
                    prep_list = meta.get("preprocessing_steps", [])
                    prep_names = [s.get("name", "") for s in prep_list] if isinstance(prep_list, list) else []
                    prep_display = ", ".join(prep_names) if prep_names else "None"
                    cached_models.append(
                        {
                            "Category": meta.get("category", "unknown"),
                            "Hash": meta.get("hash", meta_file.parent.name),
                            "Img Size": meta.get("img_size", 128),
                            "Latent": meta.get("latent_channels", meta.get("latent_dim", 32)),
                            "Epochs": meta.get("epochs", 20),
                            "Batch": meta.get("batch_size", 16),
                            "Mask Ratio": meta.get("mask_ratio", 0.25),
                            "Preprocessing": prep_display,
                            "Created": created_display,
                            "_raw_timestamp": ts_str,
                            "_raw_preprocessing_steps": prep_list,
                        }
                    )
            except Exception:
                pass

    selected_model_hash: str | None = None
    selected_model_meta: dict[str, Any] | None = None
    selected_model_hashes: list[str] = []
    load_selected_clicked = False

    if cached_models:
        # Sort newest first
        cached_models.sort(key=lambda x: str(x.get("_raw_timestamp", "")), reverse=True)
        display_models = [{k: v for k, v in m.items() if not k.startswith("_")} for m in cached_models]
        df_models = pd.DataFrame(display_models)

        selection = st.dataframe(
            df_models,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="kcae_registry_selection",
        )

        selected_rows: list[int] = []
        if selection is not None:
            if isinstance(selection, dict):
                selected_rows = selection.get("selection", {}).get("rows", [])
            else:
                sel_attr = getattr(selection, "selection", None)
                if isinstance(sel_attr, dict):
                    selected_rows = sel_attr.get("rows", [])
                elif hasattr(sel_attr, "rows"):
                    selected_rows = getattr(sel_attr, "rows", [])

        selected_model_metas = [cached_models[r] for r in selected_rows if 0 <= r < len(cached_models)]
        selected_model_hashes = [str(m.get("Hash")) for m in selected_model_metas]

        if selected_model_hashes:
            if len(selected_model_hashes) == 1:
                selected_model_meta = selected_model_metas[0]
                selected_model_hash = selected_model_hashes[0]

                # Sync inputs with selected model if selection changed
                if st.session_state.get("_last_kcae_selected_hash") != selected_model_hash:
                    st.session_state["_last_kcae_selected_hash"] = selected_model_hash
                    st.session_state["kcae_cat"] = str(selected_model_meta.get("Category", "bottle"))
                    st.session_state["kcae_epochs"] = int(selected_model_meta.get("Epochs", 20))
                    st.session_state["kcae_latent"] = int(selected_model_meta.get("Latent", 32))
                    st.session_state["kcae_img_size"] = int(selected_model_meta.get("Img Size", 128))
                    st.session_state["kcae_batch"] = int(selected_model_meta.get("Batch", 16))
                    st.session_state["kcae_mask_ratio"] = float(selected_model_meta.get("Mask Ratio", 0.25))

                    # Sync preprocessing checkboxes with the cached model's preprocessing configuration
                    raw_prep = selected_model_meta.get("_raw_preprocessing_steps", [])
                    if isinstance(raw_prep, list):
                        st.session_state["kcae_mask"] = any(s.get("name") == "foreground_mask" for s in raw_prep)
                        st.session_state["kcae_clahe"] = any(s.get("name") == "clahe" for s in raw_prep)
                        st.session_state["kcae_gaussian"] = any(s.get("name") == "gaussian_blur" for s in raw_prep)

                st.success(
                    f"Selected cached model: **`{selected_model_hash}`** ("
                    f"Category: `{selected_model_meta.get('Category')}`, "
                    f"Img Size: `{selected_model_meta.get('Img Size')}`, "
                    f"Latent: `{selected_model_meta.get('Latent')}`, "
                    f"Epochs: `{selected_model_meta.get('Epochs')}`, "
                    f"Preprocessing: `{selected_model_meta.get('Preprocessing')}`, "
                    f"Created: `{selected_model_meta.get('Created')}`)"
                )
                col_load, col_del, _ = st.columns([2, 1, 3])
                load_selected_clicked = col_load.button(
                    f"⚡ Load & Evaluate Model `{selected_model_hash}`",
                    type="primary",
                    key="btn_load_kcae_selected",
                )
                with col_del.popover("🗑️ Delete Model", help=f"Move model {selected_model_hash} to Trash"):
                    st.warning(f"Move model `{selected_model_hash}` to Trash (can be restored)?")
                    if st.button("Move to Trash", type="primary", key="btn_confirm_delete_single"):
                        if delete_cached_model(selected_model_hash, registry_base=registry_path, soft_delete=True):
                            st.session_state.pop("_last_kcae_selected_hash", None)
                            st.success(f"Model `{selected_model_hash}` moved to Trash (reversible).")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete model `{selected_model_hash}`.")
            else:
                st.warning(f"Selected **{len(selected_model_hashes)} models**: `{', '.join(selected_model_hashes)}`")
                col_del_multi, _ = st.columns([2, 4])
                with col_del_multi.popover(
                    f"🗑️ Delete {len(selected_model_hashes)} Models",
                    help=f"Move {len(selected_model_hashes)} selected models to Trash",
                ):
                    st.warning(f"Move **{len(selected_model_hashes)}** selected models to Trash?")
                    st.markdown("\n".join(f"- `{h}`" for h in selected_model_hashes))
                    if st.button(
                        f"Move to Trash ({len(selected_model_hashes)} models)",
                        type="primary",
                        key="btn_confirm_delete_multi",
                    ):
                        deleted_cnt = 0
                        for h in selected_model_hashes:
                            if delete_cached_model(h, registry_base=registry_path, soft_delete=True):
                                deleted_cnt += 1
                        st.session_state.pop("_last_kcae_selected_hash", None)
                        st.success(f"Moved {deleted_cnt} model(s) to Trash (reversible).")
                        st.rerun()
        else:
            st.info(
                "💡 **Interactive Model Registry:** Click on any row above to select, load, or delete that "
                "cached model. The pipeline always loads the newest matching cached model automatically when available."
            )
    else:
        st.caption("No cached models found in registry.")

    # ── Trash & Restoration Section ───────────────────────────────────────────────
    trashed_models = list_trashed_models(registry_base=registry_path)
    if trashed_models:
        with st.expander(f"🗑️ Trash / Recently Deleted ({len(trashed_models)} models)", expanded=False):
            st.caption("Soft-deleted models are safely preserved here and can be restored at any time.")
            trashed_display = []
            for tm in trashed_models:
                ts_raw = tm.get("timestamp", "")
                trashed_display.append(
                    {
                        "Category": tm.get("category", "unknown"),
                        "Hash": tm.get("hash", "unknown"),
                        "Img Size": tm.get("img_size", 128),
                        "Latent": tm.get("latent_channels", tm.get("latent_dim", 32)),
                        "Epochs": tm.get("epochs", 20),
                        "Created": ts_raw[:19].replace("T", " ") if ts_raw else "Unknown",
                    }
                )
            df_trashed = pd.DataFrame(trashed_display)
            trash_selection = st.dataframe(
                df_trashed,
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="kcae_trash_selection",
            )

            trashed_sel_rows: list[int] = []
            if trash_selection is not None:
                if isinstance(trash_selection, dict):
                    trashed_sel_rows = trash_selection.get("selection", {}).get("rows", [])
                else:
                    sel_attr = getattr(trash_selection, "selection", None)
                    if isinstance(sel_attr, dict):
                        trashed_sel_rows = sel_attr.get("rows", [])
                    elif hasattr(sel_attr, "rows"):
                        trashed_sel_rows = getattr(sel_attr, "rows", [])

            trashed_selected_hashes = [
                str(trashed_models[r].get("hash")) for r in trashed_sel_rows if 0 <= r < len(trashed_models)
            ]

            col_rest, col_purge, _ = st.columns([2, 2, 4])
            if trashed_selected_hashes:
                if col_rest.button(
                    f"♻️ Restore Selected ({len(trashed_selected_hashes)})",
                    type="primary",
                    key="btn_restore_selected",
                ):
                    restored_cnt = 0
                    for th in trashed_selected_hashes:
                        if restore_cached_model(th, registry_base=registry_path):
                            restored_cnt += 1
                    st.success(f"Restored {restored_cnt} model(s) back to registry!")
                    st.rerun()
            else:
                if col_rest.button("♻️ Restore All Trashed", key="btn_restore_all"):
                    restored_cnt = 0
                    for tm in trashed_models:
                        th_val = str(tm.get("hash", ""))
                        if th_val and restore_cached_model(th_val, registry_base=registry_path):
                            restored_cnt += 1
                    st.success(f"Restored all {restored_cnt} model(s) back to registry!")
                    st.rerun()

            with col_purge.popover(
                "⚠️ Empty Trash (Permanent)", help="Permanently delete all models in trash from disk"
            ):
                st.warning("This will PERMANENTLY erase all models currently in the trash directory.")
                if st.button("Confirm Empty Trash", type="primary", key="btn_confirm_purge_trash"):
                    purged = purge_trash(registry_base=registry_path)
                    st.success(f"Permanently erased {purged} model(s).")
                    st.rerun()

    st.divider()

    # Initialize session state defaults
    st.session_state.setdefault("kcae_root", "data/raw/mvtec_ad")
    st.session_state.setdefault("kcae_cat", "bottle")
    st.session_state.setdefault("kcae_epochs", 20)
    st.session_state.setdefault("kcae_latent", 32)
    st.session_state.setdefault("kcae_img_size", 128)
    st.session_state.setdefault("kcae_batch", 16)
    st.session_state.setdefault("kcae_mask_ratio", 0.25)
    st.session_state.setdefault("kcae_mask", True)
    st.session_state.setdefault("kcae_clahe", False)
    st.session_state.setdefault("kcae_gaussian", False)

    def load_optuna_defaults() -> None:
        selected_cat = st.session_state.kcae_cat
        registry_path = Path("data/hyperparameters/keras_cae_best.json")
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)

            if selected_cat in registry:
                cfg = registry[selected_cat]
                # Backward compatibility for flat schema just in case
                if "preprocessing" not in cfg:
                    prep = cfg
                    hp = cfg
                else:
                    prep = cfg.get("preprocessing", {})
                    hp = cfg.get("model_hyperparameters", {})

                if "latent_dim" in hp:
                    st.session_state.kcae_latent = hp["latent_dim"]
                elif "latent_channels" in hp:
                    st.session_state.kcae_latent = hp["latent_channels"]

                if "use_clahe" in prep:
                    st.session_state.kcae_clahe = prep["use_clahe"]
                elif "apply_clahe" in prep:
                    st.session_state.kcae_clahe = prep["apply_clahe"]

                if "use_gaussian_blur" in prep:
                    st.session_state.kcae_gaussian = prep["use_gaussian_blur"]
                elif "apply_blur" in prep:
                    st.session_state.kcae_gaussian = prep["apply_blur"]

                if "use_foreground_mask" in prep:
                    st.session_state.kcae_mask = prep["use_foreground_mask"]
                elif "apply_foreground_mask" in prep:
                    st.session_state.kcae_mask = prep["apply_foreground_mask"]

    col1, col2 = st.columns(2)
    data_root = col1.text_input("Dataset Root Directory", key="kcae_root")

    mvtec_categories = [
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
    ]
    category = col2.selectbox("Category Name", options=mvtec_categories, key="kcae_cat", on_change=load_optuna_defaults)

    st.subheader("Training Hyperparameters")
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", min_value=1, max_value=100, step=5, key="kcae_epochs")
    latent_channels = c2.number_input("Latent Channels", min_value=8, max_value=256, step=8, key="kcae_latent")
    img_size = c3.number_input("Image Size", min_value=64, max_value=256, step=16, key="kcae_img_size")
    batch_size = c4.number_input("Batch Size", min_value=4, max_value=64, step=4, key="kcae_batch")

    with st.expander("Advanced Hyperparameters"):
        ac1, ac2, ac3 = st.columns(3)
        mask_ratio = ac1.slider("Mask Ratio (MIM)", 0.0, 0.75, step=0.05, key="kcae_mask_ratio")
        threshold_method = ac2.selectbox("Threshold Method", ["quantile", "mahalanobis"])
        k_fraction = ac3.number_input(
            "Top-K Fraction", min_value=0.001, max_value=0.050, value=0.002, step=0.001, format="%.3f"
        )

    st.subheader("Preprocessing Options")
    use_seg = st.checkbox("Apply Otsu+Canny Foreground Masking (BGRP-G)", key="kcae_mask")
    use_clahe = st.checkbox("Apply CLAHE", key="kcae_clahe")
    use_gaussian = st.checkbox("Apply Gaussian Blur", key="kcae_gaussian")

    preprocessing_steps = []
    if use_seg:
        preprocessing_steps.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        preprocessing_steps.append({"name": "clahe", "params": {}})
    if use_gaussian:
        preprocessing_steps.append({"name": "gaussian_blur", "params": {}})

    st.subheader("Execution")
    force_retrain = st.checkbox("Force Retrain Model (bypass cache even if hyperparameters match)", value=False)

    run_pipeline_clicked = st.button("Run Keras CAE Pipeline")

    if not (load_selected_clicked or run_pipeline_clicked):
        return

    active_hash = selected_model_hash if load_selected_clicked else None
    active_force_retrain = False if load_selected_clicked else force_retrain
    active_prep = (
        selected_model_meta.get("_raw_preprocessing_steps", preprocessing_steps)
        if load_selected_clicked and selected_model_meta
        else preprocessing_steps
    )

    with st.spinner("Executing Keras CAE pipeline and evaluating..."):
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
            "preprocessing_steps": active_prep,
            "run_heatmap": True,  # Automatically compute heatmaps
            "force_retrain": active_force_retrain,
            "model_hash": active_hash,
        }
        data = make_api_request("/api/pipelines/keras_cae", payload, timeout=None)

    if not data:
        return

    st.success("Pipeline completed successfully!")
    results = data.get("results", {})

    # ── Metrics ──────────────────────────────────────────────────────────────────
    _render_evaluation_summary(results)

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
