"""Keras Convolutional Autoencoder (CAE) evaluation tab for Streamlit UI."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.pipelines.modelling.keras_cae.cae_pipeline import (
    delete_cached_model,
    list_trashed_models,
    purge_trash,
    restore_cached_model,
)
from app.ui.components.api_client import make_api_request
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


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

                if st.session_state.get("_last_kcae_selected_hash") != selected_model_hash:
                    st.session_state["_last_kcae_selected_hash"] = selected_model_hash
                    st.session_state["kcae_cat"] = str(selected_model_meta.get("Category", "bottle"))
                    st.session_state["kcae_epochs"] = int(selected_model_meta.get("Epochs", 20))
                    st.session_state["kcae_latent"] = int(selected_model_meta.get("Latent", 32))
                    st.session_state["kcae_img_size"] = int(selected_model_meta.get("Img Size", 128))
                    st.session_state["kcae_batch"] = int(selected_model_meta.get("Batch", 16))
                    st.session_state["kcae_mask_ratio"] = float(selected_model_meta.get("Mask Ratio", 0.25))

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
        reg_path = Path("data/hyperparameters/keras_cae_best.json")
        if reg_path.exists():
            with open(reg_path, encoding="utf-8") as f:
                registry = json.load(f)

            if selected_cat in registry:
                cfg = registry[selected_cat]
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
            "run_heatmap": True,
            "force_retrain": active_force_retrain,
            "model_hash": active_hash,
        }
        data = make_api_request("/api/pipelines/keras_cae", payload, timeout=None)

    if not data:
        return

    st.success("Pipeline completed successfully!")
    results = data.get("results", {})

    _render_evaluation_summary(results)

    st.caption(
        f"Adaptive Threshold: `{results.get('threshold', 0.0):.6f}` | "
        f"Final Training Loss: `{results.get('final_train_loss', 0.0):.6f}`"
    )

    if loss_history := results.get("loss_history"):
        if isinstance(loss_history, dict):
            clean_history = {k: v for k, v in loss_history.items() if isinstance(v, list) and len(v) > 0}
            if clean_history:
                st.subheader("Training Loss Curve")
                df_loss = pd.DataFrame({k: pd.Series(v) for k, v in clean_history.items()})
                st.line_chart(df_loss)

    _render_heatmap_explorer(results)
