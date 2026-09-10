"""PatchCore evaluation tab for Streamlit UI."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.pipelines.evaluation.visualization import render_evaluation_curves
from app.pipelines.modelling.patchcore import (
    delete_cached_patchcore_model,
    list_trashed_patchcore_models,
    purge_patchcore_trash,
    restore_cached_patchcore_model,
)
from app.ui.components.api_client import make_api_request
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


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
                        if delete_cached_patchcore_model(
                            selected_model_hash, registry_base=registry_path, soft_delete=True
                        ):
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
                            if delete_cached_patchcore_model(h, registry_base=registry_path, soft_delete=True):
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
    trashed_models = list_trashed_patchcore_models(registry_base=registry_path)
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
                        if restore_cached_patchcore_model(th, registry_base=registry_path):
                            restored_cnt += 1
                    st.success(f"Restored {restored_cnt} Patchcore model(s) back to registry!")
                    st.rerun()
            else:
                if col_rest.button("♻️ Restore All Trashed", key="btn_restore_patchcore_all"):
                    restored_cnt = 0
                    for tm in trashed_models:
                        th_val = str(tm.get("hash", ""))
                        if th_val and restore_cached_patchcore_model(th_val, registry_base=registry_path):
                            restored_cnt += 1
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
        reg_path = Path("data/hyperparameters/patchcore_best.json")
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

            pixel_metrics = results.get("pixel_level", {})
            metrics_path = pixel_metrics.get("metrics_path")
            if metrics_path:
                render_evaluation_curves(metrics_path)

            _render_heatmap_explorer(results)
        else:
            st.text_area("Results Summary", value=str(results), height=180)
