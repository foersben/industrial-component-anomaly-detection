"""DINO vision foundation model evaluation tab for Streamlit UI."""

import json
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import streamlit as st

from app.domain.categories import discover_dataset_categories
from app.pipelines.evaluation.visualization import render_evaluation_curves
from app.pipelines.modelling.dino import (
    delete_cached_dino_model,
    list_trashed_dino_models,
    purge_dino_trash,
    restore_cached_dino_model,
    run_dinov2_baseline,
    run_dinov3_baseline,
)
from app.pipelines.modelling.dino.artifacts import load_completed_category_result
from app.ui.components.heatmaps import _render_heatmap_explorer
from app.ui.components.metrics import _render_evaluation_summary


def _load_cached_dino_runs(registry_path: Path) -> list[dict[str, Any]]:
    """Load display and selection metadata for completed DINO runs.

    Args:
        registry_path: Path to the DINO registry.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing the selected run's metadata.
    """
    runs: list[dict[str, Any]] = []

    if not registry_path.exists():
        return runs

    for meta_file in registry_path.rglob("metadata.json"):
        if ".trash" in meta_file.parts:
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            timestamp = str(meta.get("timestamp", ""))
            runs.append(
                {
                    "Category": meta.get("category", "unknown"),
                    "Hash": meta.get("hash", meta_file.parent.name),
                    "Encoder": meta.get("encoder_name", "unknown"),
                    "k": meta.get("num_neighbors", 1),
                    "Masking": meta.get("masking_mode", "off"),
                    "Variant": meta.get("variant", "baseline"),
                    "Created": timestamp[:19].replace("T", " ") if timestamp else "Unknown",
                    "_raw_timestamp": timestamp,
                    "_artifact_dir": str(meta_file.parent),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(runs, key=lambda x: str(x["_raw_timestamp"]), reverse=True)


def _extract_selected_rows(selection: Any) -> list[int]:
    """Extract selected row indices from Streamlit's dataframe event value.

    Args:
        selection: Streamlit dataframe selection event value.

    Returns:
        list[int]: List of selected row indices.
    """
    if selection is None:
        return []
    if isinstance(selection, dict):
        return list(selection.get("selection", {}).get("rows", []))
    selected = getattr(selection, "selection", None)
    if isinstance(selected, dict):
        return list(selected.get("rows", []))
    if hasattr(selected, "rows"):
        return list(getattr(selected, "rows", []))
    return []


def _sync_dino_session_state(selected_run: dict[str, Any]) -> None:
    """Populate DINO controls from a selected cached run.

    Args:
        selected_run: Dictionary containing the selected run's metadata.
    """
    selected_hash = str(selected_run["Hash"])
    if st.session_state.get("_last_dino_selected_hash") == selected_hash:
        return
    st.session_state["_last_dino_selected_hash"] = selected_hash
    st.session_state["dino_cat"] = str(selected_run["Category"])
    st.session_state["dino_num_neighbors"] = int(selected_run["k"])
    st.session_state["dino_variant"] = str(selected_run["Variant"])
    st.session_state["dino_masking"] = str(selected_run["Masking"])


def _handle_multi_delete(registry_path: Path, architecture: str, selected_hashes: list[str]) -> None:
    """Render popover UI for deleting multiple selected models.

    Args:
        registry_path: Path to the DINO registry.
        architecture: Name of the DINO architecture.
        selected_hashes: List of selected model hashes.
    """
    st.warning(f"Selected **{len(selected_hashes)} models**: `{', '.join(selected_hashes)}`")
    with st.popover(
        f"🗑️ Delete {len(selected_hashes)} Models",
        help=f"Move {len(selected_hashes)} selected {architecture} models to Trash",
    ):
        st.warning(f"Move **{len(selected_hashes)}** selected {architecture} models to Trash?")
        if st.button(
            f"Move to Trash ({len(selected_hashes)} models)",
            type="primary",
            key="btn_confirm_delete_dino_multi",
        ):
            deleted = sum(
                delete_cached_dino_model(model_hash, registry_base=registry_path) for model_hash in selected_hashes
            )
            st.session_state.pop("_last_dino_selected_hash", None)
            st.success(f"Moved {deleted} {architecture} model(s) to Trash (reversible).")
            st.rerun()


def _handle_single_action(
    registry_path: Path, architecture: str, selected_run: dict[str, Any]
) -> dict[str, Any] | None:
    """Render UI actions (Load and Delete) for a single selected model.

    Args:
        registry_path: Path to the DINO registry.
        architecture: Name of the DINO architecture.
        selected_run: Dictionary containing the selected run's metadata.

    Returns:
        Dictionary containing the selected run's metadata and action or None if no action is taken.
    """
    _sync_dino_session_state(selected_run)
    selected_hash = str(selected_run["Hash"])
    st.success(
        f"Selected cached {architecture} run: **`{selected_hash}`** ("
        f"Category: `{selected_run['Category']}`, k: `{selected_run['k']}`, "
        f"Masking: `{selected_run['Masking']}`, Variant: `{selected_run['Variant']}`, "
        f"Created: `{selected_run['Created']}`)"
    )
    col_load, col_delete, _ = st.columns([2, 1, 3])

    if col_load.button(
        f"⚡ Load Saved Results `{selected_hash}`",
        type="primary",
        key="btn_load_dino_selected",
    ):
        selected_run["_action"] = "results"
        return selected_run

    with col_delete.popover("🗑️ Delete Model", help=f"Move {architecture} model {selected_hash} to Trash"):
        st.warning(f"Move {architecture} model `{selected_hash}` to Trash (can be restored)?")
        if st.button("Move to Trash", type="primary", key="btn_confirm_delete_dino_single"):
            if delete_cached_dino_model(selected_hash, registry_base=registry_path):
                st.session_state.pop("_last_dino_selected_hash", None)
                st.success(f"{architecture} model `{selected_hash}` moved to Trash (reversible).")
                st.rerun()
            else:
                st.error(f"Failed to delete {architecture} model `{selected_hash}`.")
    return None


def _render_dino_registry_section(registry_path: Path, architecture: str) -> dict[str, Any] | None:
    """Render the cached-model registry and return a run chosen for loading.

    Args:
        registry_path: Path to the DINO registry.
        architecture: Name of the DINO architecture.

    Returns:
        dict[str, Any]: Dictionary containing the selected run's metadata, or None if no run was selected.
    """
    st.subheader("Model Registry (Cached Models)")
    cached_runs = _load_cached_dino_runs(registry_path)
    if not cached_runs:
        st.caption("No cached models found in registry.")
        return None

    display_runs = pd.DataFrame(
        [{key: value for key, value in run.items() if not key.startswith("_")} for run in cached_runs]
    )
    selection = st.dataframe(
        display_runs,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"dino_registry_selection_{architecture.lower()}",
    )
    selected_rows = _extract_selected_rows(selection)
    if not selected_rows:
        st.info(
            f"💡 **Interactive {architecture} Model Registry:** Click a row above to select, load, or delete that "
            "cached model. Loading validates and displays the exact saved evaluation artifacts."
        )
        return None

    selected_runs = [cached_runs[row] for row in selected_rows if 0 <= row < len(cached_runs)]
    if len(selected_runs) > 1:
        selected_hashes = [str(run["Hash"]) for run in selected_runs]
        _handle_multi_delete(registry_path, architecture, selected_hashes)
        return None

    return _handle_single_action(registry_path, architecture, selected_runs[0])

    # selected_run = selected_runs[0]
    # _sync_dino_session_state(selected_run)
    # selected_hash = str(selected_run["Hash"])
    # st.success(
    #     f"Selected cached {architecture} run: **`{selected_hash}`** ("
    #     f"Category: `{selected_run['Category']}`, k: `{selected_run['k']}`, "
    #     f"Masking: `{selected_run['Masking']}`, Variant: `{selected_run['Variant']}`, "
    #     f"Created: `{selected_run['Created']}`)"
    # )
    # col_load, col_delete, _ = st.columns([2, 1, 3])
    # if col_load.button(
    #     f"⚡ Load Saved Results `{selected_hash}`",
    #     type="primary",
    #     key="btn_load_dino_selected",
    # ):
    #     selected_run["_action"] = "results"
    #     return selected_run
    # with col_delete.popover("🗑️ Delete Model", help=f"Move {architecture} model {selected_hash} to Trash"):
    #     st.warning(f"Move {architecture} model `{selected_hash}` to Trash (can be restored)?")
    #     if st.button("Move to Trash", type="primary", key="btn_confirm_delete_dino_single"):
    #         if delete_cached_dino_model(selected_hash, registry_base=registry_path):
    #             st.session_state.pop("_last_dino_selected_hash", None)
    #             st.success(f"{architecture} model `{selected_hash}` moved to Trash (reversible).")
    #             st.rerun()
    #         else:
    #             st.error(f"Failed to delete {architecture} model `{selected_hash}`.")
    # return None


def _render_dino_trash_section(registry_path: Path, architecture: str) -> None:
    """Render recovery and permanent-purge actions for soft-deleted DINO runs."""
    trashed_models = list_trashed_dino_models(registry_path)
    if not trashed_models:
        return
    with st.expander(f"🗑️ Trash / Recently Deleted ({len(trashed_models)} models)", expanded=False):
        st.caption(f"Soft-deleted {architecture} models are safely preserved here and can be restored at any time.")
        display = [
            {
                "Category": model.get("category", "unknown"),
                "Hash": model.get("hash", "unknown"),
                "Encoder": model.get("encoder_name", "unknown"),
                "k": model.get("num_neighbors", 1),
                "Created": str(model.get("timestamp", ""))[:19].replace("T", " "),
            }
            for model in trashed_models
        ]
        selection = st.dataframe(
            pd.DataFrame(display),
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"dino_trash_selection_{architecture.lower()}",
        )
        selected_rows = _extract_selected_rows(selection)
        selected_hashes = [
            str(trashed_models[row].get("hash", "")) for row in selected_rows if 0 <= row < len(trashed_models)
        ]
        col_restore, col_purge, _ = st.columns([2, 2, 4])
        hashes_to_restore = selected_hashes or [str(model.get("hash", "")) for model in trashed_models]
        restore_label = f"♻️ Restore Selected ({len(selected_hashes)})" if selected_hashes else "♻️ Restore All Trashed"
        if col_restore.button(restore_label, type="primary" if selected_hashes else "secondary"):
            restored = sum(
                restore_cached_dino_model(model_hash, registry_base=registry_path)
                for model_hash in hashes_to_restore
                if model_hash
            )
            st.success(f"Restored {restored} {architecture} model(s) back to registry!")
            st.rerun()
        with col_purge.popover(
            "⚠️ Empty Trash (Permanent)",
            help=f"Permanently delete all {architecture} models in Trash",
        ):
            st.error("Are you sure you want to permanently delete these models? This cannot be undone.")
            if st.button("Yes, Empty Trash", type="primary", key="btn_purge_dino_trash"):
                purged = purge_dino_trash(registry_path)
                st.success(f"Permanently purged {purged} {architecture} model(s) from disk.")
                st.rerun()


def _load_selected_dino_results(selected_run: dict[str, Any], run_heatmap: bool) -> dict[str, Any] | None:
    """Load and validate the complete artifacts belonging to a selected registry row."""
    artifact_dir = Path(str(selected_run["_artifact_dir"]))
    result = load_completed_category_result(
        artifact_dir,
        category=str(selected_run["Category"]),
        model_hash=str(selected_run["Hash"]),
        run_heatmap=run_heatmap,
        load_heatmaps=False,
    )
    if result is None:
        st.error("The selected cached run is incomplete or its metadata does not match its registry entry.")
        return None
    st.success(f"Loaded cached DINO results `{selected_run['Hash']}`.")
    return cast("dict[str, Any]", result)


def _render_dino_config_controls(architecture: str) -> tuple[int, str, Literal["off", "on", "published"]]:
    """Render slider and dropdown controls for DINO model hyperparameters.

    Args:
        architecture: Selected vision transformer model family ('DINOv2' or 'DINOv3').

    Returns:
        Tuple of (num_neighbors, variant, masking_policy).
    """
    col1, col2, col3 = st.columns(3)
    st.session_state.setdefault("dino_num_neighbors", 1)
    num_neighbors = col1.slider(
        "Patch Nearest Neighbors (k)", min_value=1, max_value=10, step=1, key="dino_num_neighbors"
    )

    if architecture == "DINOv2":
        variant = col2.selectbox(
            "Scorer Variant",
            ["baseline", "enhanced"],
            key="dino_variant",
            help="Stock or enhanced position/density-aware scorer",
        )
        masking_options: list[Literal["off", "on", "published"]] = ["published", "off", "on"]
        masking = col3.selectbox("Foreground Masking Policy", options=masking_options, key="dino_masking")
    else:
        variant = "baseline"
        masking = "off"
        col2.info("DINOv3 uses frozen backbone with stock scorer.")

    return num_neighbors, variant, masking


def _execute_dino_run(
    architecture: str,
    data_root: str,
    category: str,
    num_neighbors: int,
    masking: Literal["off", "on", "published"],
    run_heatmap: bool,
    variant: str,
    reuse_complete: bool = False,
) -> dict[str, Any] | None:
    """Execute the selected DINO pipeline under a loading spinner.

    Args:
        architecture: Model architecture ('DINOv2' or 'DINOv3').
        data_root: Root dataset directory path.
        category: Component category name or 'all'.
        num_neighbors: Number of patch nearest neighbors.
        masking: Foreground masking policy.
        run_heatmap: Whether to compute heatmap overlays.
        variant: Scorer variant for DINOv2.
        reuse_complete: Whether to return an exact cached result immediately.

    Returns:
        Evaluation results dictionary if successful, None otherwise.
    """
    with st.spinner(f"Running {architecture} nearest-neighbor evaluation on '{category}'..."):
        try:
            if architecture == "DINOv2":
                results = run_dinov2_baseline(
                    data_root=Path(data_root),
                    category=category,
                    num_neighbors=num_neighbors,
                    masking=masking,
                    run_heatmap=run_heatmap,
                    variant=cast("Literal['baseline', 'enhanced']", variant),
                    reuse_complete=reuse_complete,
                )
            else:
                results = run_dinov3_baseline(
                    data_root=Path(data_root),
                    category=category,
                    num_neighbors=num_neighbors,
                    masking=masking,
                    run_heatmap=run_heatmap,
                    reuse_complete=reuse_complete,
                )
            st.success(f"{architecture} baseline execution finished.")
            return cast("dict[str, Any]", results)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return None


def _render_dino_all_categories_summary(results_dict: dict[str, Any]) -> None:
    """Render macro averages and category breakdown table for multi-category runs.

    Args:
        results_dict: Multi-category evaluation results dictionary.
    """
    st.subheader("Macro Averages Across All Categories")
    macro = results_dict.get("macro_average", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Image AUROC", f"{macro.get('image_auroc', 0.0):.4f}")
    m2.metric("Pixel AUROC", f"{macro.get('pixel_auroc', 0.0):.4f}")
    m3.metric("Pixel AUPIMO", f"{macro.get('pixel_aupimo', 0.0):.4f}")

    st.subheader("Category Breakdown")
    cat_data = []
    for cat_name, cat_res in results_dict.get("categories", {}).items():
        img_m = cat_res.get("image_level", {})
        pix_m = cat_res.get("pixel_level", {})
        cat_data.append(
            {
                "Category": cat_name,
                "Image AUROC": img_m.get("auroc", 0.0),
                "Pixel AUROC": pix_m.get("auroc", 0.0),
                "Pixel AUPIMO": pix_m.get("aupimo", 0.0),
            }
        )
    st.dataframe(cat_data, width="stretch", hide_index=True)


def _render_dino_results(results_dict: dict[str, Any], category: str) -> None:
    """Render metrics cards, curves, and heatmaps for completed DINO evaluation.

    Args:
        results_dict: Evaluation output dictionary.
        category: Selected component category name or 'all'.
    """
    if category == "all" and "macro_average" in results_dict:
        _render_dino_all_categories_summary(results_dict)
    elif isinstance(results_dict, dict):
        _render_evaluation_summary(results_dict, model_type="patchcore")
        pixel_metrics = results_dict.get("pixel_level", {})
        if metrics_path := pixel_metrics.get("metrics_path"):
            if Path(metrics_path).is_file():
                render_evaluation_curves(metrics_path)
        _render_heatmap_explorer(results_dict)
    else:
        st.text_area("Results", value=str(results_dict), height=180)


def _dino_display_signature(
    architecture: str,
    data_root: str,
    category: str,
    num_neighbors: int,
    masking: str,
    variant: str,
    run_heatmap: bool,
) -> str:
    """Build the identity of the controls associated with displayed results.

    Args:
        architecture: Architecture name.
        data_root: Root directory of the dataset.
        category: Selected category name or 'all'.
        num_neighbors: Number of neighbors to use.
        masking: Masking mode.
        variant: Variant of the architecture.
        run_heatmap: Whether to run the heatmap.

    Returns:
        str: Signature of the results.
    """
    return json.dumps(
        [architecture, data_root, category, num_neighbors, masking, variant, run_heatmap], separators=(",", ":")
    )


def _remember_dino_results(results: dict[str, Any], category: str, signature: str) -> None:
    """Persist displayed results across Streamlit widget reruns.

    Args:
        results: Results dictionary.
        category: Selected category name or 'all'.
        signature: Signature of the results to remember.
    """
    st.session_state["_dino_displayed_results"] = results
    st.session_state["_dino_displayed_category"] = category
    st.session_state["_dino_displayed_signature"] = signature


def _render_remembered_dino_results(signature: str) -> bool:
    """Render the most recently loaded result for the active architecture.

    Args:
        signature: Signature of the results to render.

    Returns:
        bool: True if the results were rendered, False otherwise.
    """
    results = st.session_state.get("_dino_displayed_results")
    if not isinstance(results, dict) or st.session_state.get("_dino_displayed_signature") != signature:
        return False
    category = str(st.session_state.get("_dino_displayed_category", results.get("category", "unknown")))
    _render_dino_results(results, category)
    return True


def render_dino_tab() -> None:
    """Render the DINO vision transformer nearest-neighbor baselines tab (DINOv2 & DINOv3)."""
    st.header("DINO Vision Transformer Baselines")
    st.markdown(
        "Evaluate frozen **DINOv2** and **DINOv3** patch nearest-neighbor baselines on MVTec AD. "
        "Supports stock and enhanced multi-layer position/density-aware scorers."
    )

    architecture = st.radio("Architecture", ["DINOv2", "DINOv3"], horizontal=True, key="dino_arch")
    registry_path = Path("data/models/dinov2" if architecture == "DINOv2" else "data/models/dinov3")
    selected_run = _render_dino_registry_section(registry_path, architecture)
    _render_dino_trash_section(registry_path, architecture)
    st.divider()

    col_root, col_cat = st.columns(2)
    st.session_state.setdefault("dino_root", "data/raw/mvtec_ad")
    data_root = col_root.text_input("Dataset Root Directory", key="dino_root")
    categories = [*discover_dataset_categories(data_root), "all"]
    selected_category = st.session_state.get("dino_cat")
    if selected_category not in categories:
        st.session_state["dino_cat"] = categories[0]
    category = col_cat.selectbox("Category", options=categories, key="dino_cat")

    st.subheader("Model Configuration")
    num_neighbors, variant, masking = _render_dino_config_controls(architecture)

    st.subheader("Evaluation Settings")
    run_heatmap = st.checkbox("Generate Anomaly Prediction Heatmaps", value=False, key="dino_heatmap")
    force_retrain = st.checkbox(
        "Force Retrain (Ignore Cache)",
        value=False,
        key="dino_force_retrain",
        help="Rebuild the fitted patch bank even when cached results exist.",
    )
    display_signature = _dino_display_signature(
        architecture, data_root, category, num_neighbors, masking, variant, run_heatmap
    )

    run_clicked = st.button(f"Run {architecture} Baseline Evaluation", type="primary", key="btn_run_dino")
    if selected_run is not None and selected_run.get("_action") == "results":
        cached_results = _load_selected_dino_results(selected_run, run_heatmap)
        if cached_results is not None:
            selected_category = str(selected_run["Category"])
            _remember_dino_results(cached_results, selected_category, display_signature)
            _render_dino_results(cached_results, selected_category)
        return
    if not run_clicked:
        _render_remembered_dino_results(display_signature)
        return

    results = _execute_dino_run(
        architecture=architecture,
        data_root=data_root,
        category=category,
        num_neighbors=num_neighbors,
        masking=masking,
        run_heatmap=run_heatmap,
        variant=variant,
        reuse_complete=not force_retrain,
    )
    if results is not None:
        _remember_dino_results(results, category, display_signature)
        _render_dino_results(results, category)
