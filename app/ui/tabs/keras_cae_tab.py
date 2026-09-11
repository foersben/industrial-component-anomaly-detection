"""Keras Convolutional Autoencoder (CAE) evaluation tab for Streamlit UI."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.domain.categories import discover_dataset_categories
from app.pipelines.modelling.keras_cae import (
    delete_cached_model,
    list_trashed_models,
    purge_trash,
    restore_cached_model,
    run_keras_cae_pipeline,
)
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


def _parse_cae_metadata_file(meta_file: Path) -> dict[str, Any] | None:
    """Parse a single Keras CAE metadata.json file safely.

    Args:
        meta_file: Path to metadata.json file.

    Returns:
        Structured dictionary if successfully read, else None.
    """
    try:
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        ts_str = meta.get("timestamp", "")
        created_display = ts_str[:19].replace("T", " ") if ts_str else "Unknown"
        prep_list = meta.get("preprocessing_steps", [])
        prep_names = [s.get("name", "") for s in prep_list] if isinstance(prep_list, list) else []
        prep_display = ", ".join(prep_names) if prep_names else "None"
        return {
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
    except Exception:
        return None


def _load_cached_cae_models_list(registry_path: Path) -> list[dict[str, Any]]:
    """Scan and parse metadata for all non-trashed cached Keras CAE models.

    Args:
        registry_path: Root path to the model registry directory.

    Returns:
        List of metadata dictionaries sorted descending by timestamp.
    """
    if not registry_path.exists():
        return []
    cached_models: list[dict[str, Any]] = []
    for meta_file in registry_path.rglob("metadata.json"):
        if ".trash" in meta_file.parts:
            continue
        parsed = _parse_cae_metadata_file(meta_file)
        if parsed is not None:
            cached_models.append(parsed)
    cached_models.sort(key=lambda x: str(x.get("_raw_timestamp", "")), reverse=True)
    return cached_models


def _extract_cae_selected_rows(selection: Any) -> list[int]:
    """Extract selected row indices from a Streamlit dataframe selection event.

    Args:
        selection: Object or dict returned by st.dataframe on_select.

    Returns:
        List of integer row indices selected by the user.
    """
    if selection is None:
        return []
    if isinstance(selection, dict):
        return list(selection.get("selection", {}).get("rows", []))
    sel_attr = getattr(selection, "selection", None)
    if isinstance(sel_attr, dict):
        return list(sel_attr.get("rows", []))
    if hasattr(sel_attr, "rows"):
        return list(getattr(sel_attr, "rows", []))
    return []


def _sync_cae_session_state(selected_meta: dict[str, Any]) -> None:
    """Synchronize session state parameters when a new CAE model is selected.

    Args:
        selected_meta: Metadata dictionary of the selected model.
    """
    selected_model_hash = str(selected_meta.get("Hash"))
    if st.session_state.get("_last_kcae_selected_hash") == selected_model_hash:
        return

    st.session_state["_last_kcae_selected_hash"] = selected_model_hash
    st.session_state["kcae_cat"] = str(selected_meta.get("Category", "bottle"))
    st.session_state["kcae_epochs"] = int(selected_meta.get("Epochs", 20))
    st.session_state["kcae_latent"] = int(selected_meta.get("Latent", 32))
    st.session_state["kcae_img_size"] = int(selected_meta.get("Img Size", 128))
    st.session_state["kcae_batch"] = int(selected_meta.get("Batch", 16))
    st.session_state["kcae_mask_ratio"] = float(selected_meta.get("Mask Ratio", 0.25))

    raw_prep = selected_meta.get("_raw_preprocessing_steps", [])
    if isinstance(raw_prep, list):
        st.session_state["kcae_mask"] = any(s.get("name") == "foreground_mask" for s in raw_prep)
        st.session_state["kcae_clahe"] = any(s.get("name") == "clahe" for s in raw_prep)
        st.session_state["kcae_gaussian"] = any(s.get("name") == "gaussian_blur" for s in raw_prep)


def _handle_cae_single_selection(
    selected_meta: dict[str, Any],
    registry_path: Path,
) -> bool:
    """Handle single model selection, state sync, and action buttons in registry.

    Args:
        selected_meta: Metadata dictionary for the selected model.
        registry_path: Registry root directory path.

    Returns:
        True if the user clicked the Load & Evaluate button, False otherwise.
    """
    _sync_cae_session_state(selected_meta)
    selected_model_hash = str(selected_meta.get("Hash"))
    st.success(
        f"Selected cached model: **`{selected_model_hash}`** ("
        f"Category: `{selected_meta.get('Category')}`, "
        f"Img Size: `{selected_meta.get('Img Size')}`, "
        f"Latent: `{selected_meta.get('Latent')}`, "
        f"Epochs: `{selected_meta.get('Epochs')}`, "
        f"Preprocessing: `{selected_meta.get('Preprocessing')}`, "
        f"Created: `{selected_meta.get('Created')}`)"
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
    return load_selected_clicked


def _handle_cae_multi_deletion(selected_hashes: list[str], registry_path: Path) -> None:
    """Handle batch deletion confirmation for multiple selected Keras CAE models.

    Args:
        selected_hashes: List of model hash strings to delete.
        registry_path: Registry root directory path.
    """
    st.warning(f"Selected **{len(selected_hashes)} models**: `{', '.join(selected_hashes)}`")
    col_del_multi, _ = st.columns([2, 4])
    with col_del_multi.popover(
        f"🗑️ Delete {len(selected_hashes)} Models",
        help=f"Move {len(selected_hashes)} selected models to Trash",
    ):
        st.warning(f"Move **{len(selected_hashes)}** selected models to Trash?")
        st.markdown("\n".join(f"- `{h}`" for h in selected_hashes))
        if st.button(
            f"Move to Trash ({len(selected_hashes)} models)",
            type="primary",
            key="btn_confirm_delete_multi",
        ):
            deleted_cnt = 0
            for h in selected_hashes:
                if delete_cached_model(h, registry_base=registry_path, soft_delete=True):
                    deleted_cnt += 1
            st.session_state.pop("_last_kcae_selected_hash", None)
            st.success(f"Moved {deleted_cnt} model(s) to Trash (reversible).")
            st.rerun()


def _render_cae_registry_section(
    registry_path: Path,
) -> tuple[str | None, dict[str, Any] | None, bool]:
    """Render the cached model registry table and selection controls.

    Args:
        registry_path: Root path to the model registry directory.

    Returns:
        Tuple of (selected_model_hash, selected_model_meta, load_selected_clicked).
    """
    st.subheader("Model Registry (Cached Models)")
    cached_models = _load_cached_cae_models_list(registry_path)
    if not cached_models:
        st.caption("No cached models found in registry.")
        return None, None, False

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

    selected_rows = _extract_cae_selected_rows(selection)
    selected_model_metas = [cached_models[r] for r in selected_rows if 0 <= r < len(cached_models)]
    selected_model_hashes = [str(m.get("Hash")) for m in selected_model_metas]

    if not selected_model_hashes:
        st.info(
            "💡 **Interactive Model Registry:** Click on any row above to select, load, or delete that "
            "cached model. The pipeline always loads the newest matching cached model automatically when available."
        )
        return None, None, False

    if len(selected_model_hashes) == 1:
        meta = selected_model_metas[0]
        load_clicked = _handle_cae_single_selection(meta, registry_path)
        return selected_model_hashes[0], meta, load_clicked

    _handle_cae_multi_deletion(selected_model_hashes, registry_path)
    return None, None, False


def _restore_cae_models(
    trashed_selected_hashes: list[str],
    trashed_models: list[dict[str, Any]],
    registry_path: Path,
    col_rest: Any,
) -> None:
    """Handle restoration of selected or all soft-deleted models.

    Args:
        trashed_selected_hashes: Hashes selected via the trash table.
        trashed_models: Full list of trashed model metadata dicts.
        registry_path: Root registry path.
        col_rest: Streamlit column to place the restore button.
    """
    if trashed_selected_hashes:
        if col_rest.button(
            f"♻️ Restore Selected ({len(trashed_selected_hashes)})",
            type="primary",
            key="btn_restore_selected",
        ):
            restored_cnt = sum(
                1 for th in trashed_selected_hashes if restore_cached_model(th, registry_base=registry_path)
            )
            st.success(f"Restored {restored_cnt} model(s) back to registry!")
            st.rerun()
    else:
        if col_rest.button("♻️ Restore All Trashed", key="btn_restore_all"):
            restored_cnt = sum(
                1
                for tm in trashed_models
                if (th_val := str(tm.get("hash", ""))) and restore_cached_model(th_val, registry_base=registry_path)
            )
            st.success(f"Restored all {restored_cnt} model(s) back to registry!")
            st.rerun()


def _render_cae_trash_section(registry_path: Path) -> None:
    """Render the trash / soft-deleted models section and restoration tools.

    Args:
        registry_path: Root path to the model registry directory.
    """
    trashed_models = list_trashed_models(registry_base=registry_path)
    if not trashed_models:
        return

    with st.expander(f"🗑️ Trash / Recently Deleted ({len(trashed_models)} models)", expanded=False):
        st.caption("Soft-deleted models are safely preserved here and can be restored at any time.")
        trashed_display = [
            {
                "Category": tm.get("category", "unknown"),
                "Hash": tm.get("hash", "unknown"),
                "Img Size": tm.get("img_size", 128),
                "Latent": tm.get("latent_channels", tm.get("latent_dim", 32)),
                "Epochs": tm.get("epochs", 20),
                "Created": ts_raw[:19].replace("T", " ") if (ts_raw := tm.get("timestamp", "")) else "Unknown",
            }
            for tm in trashed_models
        ]
        trash_selection = st.dataframe(
            pd.DataFrame(trashed_display),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="kcae_trash_selection",
        )

        trashed_sel_rows = _extract_cae_selected_rows(trash_selection)
        trashed_selected_hashes = [
            str(trashed_models[r].get("hash")) for r in trashed_sel_rows if 0 <= r < len(trashed_models)
        ]

        col_rest, col_purge, _ = st.columns([2, 2, 4])
        _restore_cae_models(trashed_selected_hashes, trashed_models, registry_path, col_rest)

        with col_purge.popover("⚠️ Empty Trash (Permanent)", help="Permanently delete all models in trash from disk"):
            st.warning("This will PERMANENTLY erase all models currently in the trash directory.")
            if st.button("Confirm Empty Trash", type="primary", key="btn_confirm_purge_trash"):
                purged = purge_trash(registry_base=registry_path)
                st.success(f"Permanently erased {purged} model(s).")
                st.rerun()


def _apply_optuna_cae_config(cfg: dict[str, Any]) -> None:
    """Apply Optuna best hyperparameters dictionary to Streamlit session state.

    Args:
        cfg: Category hyperparameter dictionary from Optuna JSON.
    """
    prep = cfg if "preprocessing" not in cfg else cfg.get("preprocessing", {})
    hp = cfg if "preprocessing" not in cfg else cfg.get("model_hyperparameters", {})

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


def _load_cae_optuna_defaults() -> None:
    """Load tuned hyperparameters for the currently selected category from disk."""
    selected_cat = st.session_state.kcae_cat
    reg_path = Path("data/hyperparameters/keras_cae_best.json")
    if not reg_path.exists():
        return
    try:
        with open(reg_path, encoding="utf-8") as f:
            registry = json.load(f)
        if selected_cat in registry:
            _apply_optuna_cae_config(registry[selected_cat])
    except Exception:
        pass


def _render_cae_config_controls() -> tuple[dict[str, Any], bool]:
    """Render input controls for training, preprocessing, and execution.

    Returns:
        Tuple of (configuration dictionary, run_button_clicked).
    """
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

    col1, col2 = st.columns(2)
    data_root = col1.text_input("Dataset Root Directory", key="kcae_root")
    categories = discover_dataset_categories(data_root)
    category = col2.selectbox(
        "Category Name",
        options=categories,
        key="kcae_cat",
        on_change=_load_cae_optuna_defaults,
    )

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

    prep_steps: list[dict[str, Any]] = []
    if use_seg:
        prep_steps.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        prep_steps.append({"name": "clahe", "params": {}})
    if use_gaussian:
        prep_steps.append({"name": "gaussian_blur", "params": {}})

    st.subheader("Execution")
    force_retrain = st.checkbox("Force Retrain Model (bypass cache even if hyperparameters match)", value=False)
    run_clicked = st.button("Run Keras CAE Pipeline")

    cfg = {
        "data_root": data_root,
        "category": category,
        "img_size": int(img_size),
        "latent_channels": int(latent_channels),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "mask_ratio": float(mask_ratio),
        "threshold_method": threshold_method,
        "k_fraction": float(k_fraction),
        "prep_steps": prep_steps,
        "force_retrain": force_retrain,
    }
    return cfg, run_clicked


def _render_cae_loss_history(loss_history: Any) -> None:
    """Render training loss curve chart if available.

    Args:
        loss_history: Dictionary containing loss metric sequences.
    """
    if not isinstance(loss_history, dict):
        return
    clean_history = {k: v for k, v in loss_history.items() if isinstance(v, list) and len(v) > 0}
    if clean_history:
        st.subheader("Training Loss Curve")
        df_loss = pd.DataFrame({k: pd.Series(v) for k, v in clean_history.items()})
        st.line_chart(df_loss)


def _execute_and_display_cae(
    cfg: dict[str, Any],
    selected_hash: str | None,
    selected_meta: dict[str, Any] | None,
    load_selected_clicked: bool,
) -> None:
    """Execute the Keras CAE pipeline and display results and heatmaps.

    Args:
        cfg: Configuration dictionary from input controls.
        selected_hash: Model hash if a specific cached model is loaded.
        selected_meta: Metadata dictionary if loading an existing model.
        load_selected_clicked: Whether execution was triggered via registry load.
    """
    active_hash = selected_hash if load_selected_clicked else None
    active_force_retrain = False if load_selected_clicked else cfg["force_retrain"]
    active_prep = (
        selected_meta.get("_raw_preprocessing_steps", cfg["prep_steps"])
        if load_selected_clicked and selected_meta
        else cfg["prep_steps"]
    )

    with st.spinner("Executing Keras CAE pipeline and evaluating..."):
        try:
            results = run_keras_cae_pipeline(
                data_root=cfg["data_root"],
                category=cfg["category"],
                img_size=cfg["img_size"],
                latent_channels=cfg["latent_channels"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                mask_ratio=cfg["mask_ratio"],
                threshold_method=cfg["threshold_method"],
                k_fraction=cfg["k_fraction"],
                pipeline=active_prep,
                run_heatmap=True,
                force_retrain=active_force_retrain,
                model_hash=active_hash,
            )
            st.success("Pipeline completed successfully!")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    _render_evaluation_summary(results)
    st.caption(
        f"Adaptive Threshold: `{results.get('threshold', 0.0):.6f}` | "
        f"Final Training Loss: `{results.get('final_train_loss', 0.0):.6f}`"
    )
    _render_cae_loss_history(results.get("loss_history"))
    _render_heatmap_explorer(results)


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

    registry_path = Path("data/models/keras_cae")
    selected_hash, selected_meta, load_selected_clicked = _render_cae_registry_section(registry_path)
    _render_cae_trash_section(registry_path)
    st.divider()

    cfg, run_clicked = _render_cae_config_controls()
    if not (load_selected_clicked or run_clicked):
        return

    _execute_and_display_cae(
        cfg=cfg,
        selected_hash=selected_hash,
        selected_meta=selected_meta,
        load_selected_clicked=load_selected_clicked,
    )
