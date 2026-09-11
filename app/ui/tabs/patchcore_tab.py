"""PatchCore evaluation tab for Streamlit UI."""

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from app.domain.categories import discover_dataset_categories
from app.pipelines.evaluation.visualization import render_evaluation_curves
from app.pipelines.modelling.patchcore import (
    delete_cached_patchcore_model,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
    run_patchcore_pipeline,
)
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


def _load_cached_patchcore_models_list(registry_path: Path) -> list[dict[str, Any]]:
    """Scan and load metadata for all active cached PatchCore models.

    Args:
        registry_path: Path to the PatchCore model registry directory.

    Returns:
        List of formatted model metadata dictionaries for UI display.
    """
    cached_models: list[dict[str, Any]] = []
    if not registry_path.exists():
        return cached_models

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
                    "Feature Layers": ", ".join(meta.get("feature_layers", ["layer2", "layer3"])),
                    "Neighbors": meta.get("num_neighbors", 9),
                    "Preprocessing": prep_display,
                    "Created": created_display,
                    "_raw_timestamp": ts_str,
                    "_raw_preprocessing_steps": prep_list,
                    "_raw_feature_layers": meta.get("feature_layers", ["layer2", "layer3"]),
                }
            )
        except Exception:
            pass

    cached_models.sort(key=lambda x: str(x.get("_raw_timestamp", "")), reverse=True)
    return cached_models


def _extract_selected_rows(selection: Any) -> list[int]:
    """Extract list of selected integer row indices from Streamlit dataframe selection state.

    Args:
        selection: Selection object returned by st.dataframe.

    Returns:
        List of selected row indices.
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


def _sync_patchcore_session_state(selected_meta: dict[str, Any]) -> None:
    """Synchronize session state parameters when a new PatchCore model is selected.

    Args:
        selected_meta: Metadata dictionary of the selected model.
    """
    selected_model_hash = str(selected_meta.get("Hash"))
    if st.session_state.get("_last_patchcore_selected_hash") == selected_model_hash:
        return

    st.session_state["_last_patchcore_selected_hash"] = selected_model_hash
    st.session_state["b_cat"] = str(selected_meta.get("Category", "bottle"))
    st.session_state["b_backbone"] = str(selected_meta.get("Backbone", "resnet18"))
    st.session_state["b_coreset_ratio"] = float(selected_meta.get("Coreset Ratio", 0.1))
    raw_layers = tuple(selected_meta.get("_raw_feature_layers", ["layer2", "layer3"]))
    st.session_state["b_feature_layers"] = "l2_l3_l4" if "layer4" in raw_layers else "l2_l3"
    st.session_state["b_num_neighbors"] = int(selected_meta.get("Neighbors", 9))

    raw_prep = selected_meta.get("_raw_preprocessing_steps", [])
    if isinstance(raw_prep, list):
        st.session_state["patchcore_mask"] = any(s.get("name") == "foreground_mask" for s in raw_prep)
        st.session_state["patchcore_clahe"] = any(s.get("name") == "clahe" for s in raw_prep)
        st.session_state["patchcore_gaussian"] = any(s.get("name") == "gaussian_blur" for s in raw_prep)


def _handle_patchcore_single_selection(
    selected_meta: dict[str, Any],
    registry_path: Path,
) -> bool:
    """Handle single model selection, state sync, and action buttons in registry.

    Args:
        selected_meta: Metadata dictionary for the selected model.
        registry_path: Registry root directory path.

    Returns:
        True if the user clicked the Load Saved Results button, False otherwise.
    """
    _sync_patchcore_session_state(selected_meta)
    selected_model_hash = str(selected_meta.get("Hash"))
    st.success(
        f"Selected cached Patchcore model: **`{selected_model_hash}`** ("
        f"Category: `{selected_meta.get('Category')}`, "
        f"Backbone: `{selected_meta.get('Backbone')}`, "
        f"Coreset Ratio: `{selected_meta.get('Coreset Ratio')}`, "
        f"Preprocessing: `{selected_meta.get('Preprocessing')}`, "
        f"Created: `{selected_meta.get('Created')}`)"
    )
    col_load, col_del, _ = st.columns([2, 1, 3])
    load_selected_clicked = col_load.button(
        f"⚡ Load Saved Results `{selected_model_hash}`",
        type="primary",
        key="btn_load_patchcore_selected",
    )
    with col_del.popover("🗑️ Delete Model", help=f"Move Patchcore model {selected_model_hash} to Trash"):
        st.warning(f"Move Patchcore model `{selected_model_hash}` to Trash (can be restored)?")
        if st.button("Move to Trash", type="primary", key="btn_confirm_delete_patchcore_single"):
            if delete_cached_patchcore_model(selected_model_hash, registry_base=registry_path, soft_delete=True):
                st.session_state.pop("_last_patchcore_selected_hash", None)
                st.success(f"Patchcore model `{selected_model_hash}` moved to Trash (reversible).")
                st.rerun()
            else:
                st.error(f"Failed to delete Patchcore model `{selected_model_hash}`.")
    return load_selected_clicked


def _handle_patchcore_multi_deletion(selected_hashes: list[str], registry_path: Path) -> None:
    """Handle batch deletion confirmation for multiple selected PatchCore models.

    Args:
        selected_hashes: List of model hash strings to delete.
        registry_path: Registry root directory path.
    """
    st.warning(f"Selected **{len(selected_hashes)} models**: `{', '.join(selected_hashes)}`")
    col_del_multi, _ = st.columns([2, 4])
    with col_del_multi.popover(
        f"🗑️ Delete {len(selected_hashes)} Models",
        help=f"Move {len(selected_hashes)} selected Patchcore models to Trash",
    ):
        st.warning(f"Move **{len(selected_hashes)}** selected Patchcore models to Trash?")
        st.markdown("\n".join(f"- `{h}`" for h in selected_hashes))
        if st.button(
            f"Move to Trash ({len(selected_hashes)} models)",
            type="primary",
            key="btn_confirm_delete_patchcore_multi",
        ):
            deleted_cnt = 0
            for h in selected_hashes:
                if delete_cached_patchcore_model(h, registry_base=registry_path, soft_delete=True):
                    deleted_cnt += 1
            st.session_state.pop("_last_patchcore_selected_hash", None)
            st.success(f"Moved {deleted_cnt} Patchcore model(s) to Trash (reversible).")
            st.rerun()


def _render_patchcore_registry_section(
    registry_path: Path,
) -> tuple[str | None, dict[str, Any] | None, bool]:
    """Render the cached PatchCore model registry table and handle row selection.

    Args:
        registry_path: Registry root directory path.

    Returns:
        Tuple of (selected_model_hash, selected_model_meta, load_selected_clicked).
    """
    st.subheader("Model Registry (Cached Patchcore Models)")
    cached_models = _load_cached_patchcore_models_list(registry_path)
    if not cached_models:
        st.caption("No cached Patchcore models found in registry.")
        return None, None, False

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

    selected_rows = _extract_selected_rows(selection)
    selected_metas = [cached_models[r] for r in selected_rows if 0 <= r < len(cached_models)]
    selected_hashes = [str(m.get("Hash")) for m in selected_metas]

    if not selected_hashes:
        st.info(
            "💡 **Interactive Patchcore Registry:** Click on any row above to select, load, or delete that "
            "cached model. The pipeline always loads the newest matching cached model automatically when available."
        )
        return None, None, False

    if len(selected_hashes) == 1:
        meta = selected_metas[0]
        load_clicked = _handle_patchcore_single_selection(meta, registry_path)
        return selected_hashes[0], meta, load_clicked

    _handle_patchcore_multi_deletion(selected_hashes, registry_path)
    return None, None, False


def _restore_patchcore_models(hashes: list[str], registry_path: Path) -> int:
    """Restore specified PatchCore models from trash back into the registry.

    Args:
        hashes: List of model hash strings to restore.
        registry_path: Base directory of the model registry.

    Returns:
        Number of models successfully restored.
    """
    restored_cnt = 0
    for th in hashes:
        if th and restore_cached_patchcore_model(th, registry_base=registry_path):
            restored_cnt += 1
    return restored_cnt


def _render_patchcore_trash_section(registry_path: Path) -> None:
    """Render the soft-deleted trash recovery section for PatchCore models.

    Args:
        registry_path: Registry root directory path.
    """
    trashed_models = list_trashed_patchcore_models(registry_base=registry_path)
    if not trashed_models:
        return

    with st.expander(f"🗑️ Trash / Recently Deleted ({len(trashed_models)} models)", expanded=False):
        st.caption("Soft-deleted Patchcore models are safely preserved here and can be restored at any time.")
        trashed_display = [
            {
                "Category": tm.get("category", "unknown"),
                "Hash": tm.get("hash", "unknown"),
                "Backbone": tm.get("backbone", "resnet18"),
                "Coreset Ratio": tm.get("coreset_sampling_ratio", 0.1),
                "Created": str(tm.get("timestamp", ""))[:19].replace("T", " ") if tm.get("timestamp") else "Unknown",
            }
            for tm in trashed_models
        ]
        df_trashed = pd.DataFrame(trashed_display)
        trash_selection = st.dataframe(
            df_trashed,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key="patchcore_trash_selection",
        )

        trashed_sel_rows = _extract_selected_rows(trash_selection)
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
                restored_cnt = _restore_patchcore_models(trashed_selected_hashes, registry_path)
                st.success(f"Restored {restored_cnt} Patchcore model(s) back to registry!")
                st.rerun()
        elif col_rest.button("♻️ Restore All Trashed", key="btn_restore_patchcore_all"):
            all_hashes = [str(tm.get("hash", "")) for tm in trashed_models]
            restored_cnt = _restore_patchcore_models(all_hashes, registry_path)
            st.success(f"Restored all {restored_cnt} Patchcore model(s) back to registry!")
            st.rerun()

        with col_purge.popover(
            "⚠️ Empty Trash (Permanent)",
            help="Permanently delete all Patchcore models in Trash",
        ):
            st.error("Are you sure you want to permanently delete these models? This cannot be undone.")
            if st.button("Yes, Empty Trash", type="primary", key="btn_purge_patchcore_trash"):
                cnt = purge_patchcore_trash(registry_base=registry_path)
                st.success(f"Permanently purged {cnt} Patchcore model(s) from disk.")
                st.rerun()


def _load_patchcore_optuna_defaults() -> None:
    """Load tuned Optuna hyperparameters into Streamlit session state on category change."""
    selected_cat = st.session_state.b_cat
    reg_path = Path("data/hyperparameters/patchcore_best.json")
    if not reg_path.exists():
        return
    try:
        with open(reg_path, encoding="utf-8") as f:
            registry = json.load(f)
        if selected_cat not in registry:
            return
        cfg = registry[selected_cat]
        prep = cfg.get("preprocessing", cfg)
        hp = cfg.get("model_hyperparameters", cfg)

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
    except Exception:
        pass


def _render_patchcore_config_controls() -> tuple[dict[str, Any], bool]:
    """Render interactive inputs for PatchCore hyperparameters and preprocessing.

    Returns:
        Tuple of (config_dictionary, run_clicked_boolean).
    """
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

    categories = discover_dataset_categories(data_root)
    category = st.selectbox("Category Name", options=categories, key="b_cat", on_change=_load_patchcore_optuna_defaults)

    st.subheader("Model Configuration")
    c1, c2, c3, c4 = st.columns(4)
    backbone = c1.selectbox("Backbone", ["resnet18", "wide_resnet50_2"], key="b_backbone")
    feature_layers_str = c2.selectbox("Feature Layers", ["l2_l3", "l2_l3_l4"], key="b_feature_layers")
    feature_layers = ("layer2", "layer3", "layer4") if feature_layers_str == "l2_l3_l4" else ("layer2", "layer3")
    coreset_ratio = c3.slider(
        "Coreset Sampling Ratio", min_value=0.001, max_value=0.2, step=0.005, format="%.3f", key="b_coreset_ratio"
    )
    num_neighbors = c4.slider("Nearest Neighbors", min_value=1, max_value=20, step=1, key="b_num_neighbors")

    st.subheader("Preprocessing Options")
    use_mask = st.checkbox("Apply Otsu+Canny Foreground Masking (zeros out background)", key="patchcore_mask")
    use_clahe = st.checkbox("Apply CLAHE", key="patchcore_clahe")
    use_gaussian = st.checkbox("Apply Gaussian Blur", key="patchcore_gaussian")

    prep_steps: list[dict[str, Any]] = []
    if use_mask:
        prep_steps.append({"name": "foreground_mask", "params": {}})
    if use_clahe:
        prep_steps.append({"name": "clahe", "params": {}})
    if use_gaussian:
        prep_steps.append({"name": "gaussian_blur", "params": {}})

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

    cfg = {
        "data_root": data_root,
        "category": category,
        "backbone": backbone,
        "feature_layers": feature_layers,
        "coreset_ratio": coreset_ratio,
        "num_neighbors": num_neighbors,
        "prep_steps": prep_steps,
        "run_heatmap": run_heatmap,
        "force_retrain": force_retrain,
    }
    return cfg, run_clicked


def _execute_and_display_patchcore(
    cfg: dict[str, Any],
    selected_hash: str | None,
    selected_meta: dict[str, Any] | None,
    load_selected_clicked: bool,
) -> None:
    """Execute PatchCore pipeline and render output metrics and curves.

    Args:
        cfg: Configuration dictionary from input controls.
        selected_hash: Optional cached model hash to load directly.
        selected_meta: Optional cached model metadata dictionary.
        load_selected_clicked: Whether load cached model was clicked.
    """
    active_hash = selected_hash if load_selected_clicked else None
    active_force_retrain = False if load_selected_clicked else cfg["force_retrain"]
    active_prep = (
        selected_meta.get("_raw_preprocessing_steps", cfg["prep_steps"])
        if load_selected_clicked and selected_meta
        else cfg["prep_steps"]
    )

    spinner_msg = (
        f"Loading saved Patchcore results `{active_hash}`..."
        if load_selected_clicked
        else "Fitting Patchcore model and evaluating Image & Pixel level metrics..."
    )

    with st.spinner(spinner_msg):
        try:
            results = run_patchcore_pipeline(
                data_root=Path(cfg["data_root"]),
                category=cfg["category"],
                backbone=cfg["backbone"],
                feature_layers=cfg["feature_layers"],
                coreset_sampling_ratio=cfg["coreset_ratio"],
                num_neighbors=cfg["num_neighbors"],
                pipeline=active_prep,
                run_heatmap=cfg["run_heatmap"],
                force_retrain=active_force_retrain,
                model_hash=active_hash,
            )
            st.success("Patchcore baseline execution finished.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    results_dict = cast("dict[str, Any]", results)
    if isinstance(results_dict, dict):
        st.session_state["_patchcore_displayed_results"] = results_dict
        st.session_state["_patchcore_displayed_signature"] = _patchcore_display_signature(cfg)
        _render_evaluation_summary(results_dict, model_type="patchcore")
        pixel_metrics = results_dict.get("pixel_level", {})
        metrics_path = pixel_metrics.get("metrics_path")
        if metrics_path:
            render_evaluation_curves(metrics_path)
        _render_heatmap_explorer(results_dict)
    else:
        st.text_area("Results Summary", value=str(results_dict), height=180)


def render_baseline_patchcore_tab() -> None:
    """Render the Patchcore anomaly detection evaluation tab with registry and caching support."""
    st.header("Patchcore Anomaly Detection & Evaluation")
    st.markdown(
        "Run Patchcore feature-memory-bank anomaly detection and evaluation on MVTec AD dataset "
        "(Image & Pixel level) with automated caching, model versioning, and soft-delete recovery."
    )

    registry_path = Path("data/models/patchcore")
    selected_hash, selected_meta, load_clicked = _render_patchcore_registry_section(registry_path)
    _render_patchcore_trash_section(registry_path)
    st.divider()

    cfg, run_clicked = _render_patchcore_config_controls()
    if run_clicked or load_clicked:
        _execute_and_display_patchcore(cfg, selected_hash, selected_meta, load_clicked)
    elif isinstance(
        cached_results := st.session_state.get("_patchcore_displayed_results"), dict
    ) and st.session_state.get("_patchcore_displayed_signature") == _patchcore_display_signature(cfg):
        _render_evaluation_summary(cached_results, model_type="patchcore")
        if metrics_path := cached_results.get("pixel_level", {}).get("metrics_path"):
            render_evaluation_curves(metrics_path)
        _render_heatmap_explorer(cached_results)


def _patchcore_display_signature(cfg: dict[str, Any]) -> str:
    """Return a stable identity for the currently visible PatchCore controls."""
    visible_cfg = {key: value for key, value in cfg.items() if key != "force_retrain"}
    return json.dumps(visible_cfg, sort_keys=True, default=list, separators=(",", ":"))
