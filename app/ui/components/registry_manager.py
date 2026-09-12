"""Reusable Streamlit components for managing cached model registries."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.core.registry import (
    delete_cached_model,
    list_trashed_models,
    purge_trash,
    restore_cached_model,
)


def extract_selected_rows(selection: Any) -> list[int]:
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


def load_cached_models_list(registry_path: Path) -> list[dict[str, Any]]:
    """Scan and load metadata for all active cached models in a registry.

    Args:
        registry_path: Path to the model registry directory.

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
            meta["_raw_timestamp"] = ts_str
            meta["_artifact_dir"] = str(meta_file.parent)

            # Ensure basic fields exist for display
            if "hash" not in meta:
                meta["hash"] = meta_file.parent.name
            if "category" not in meta:
                meta["category"] = "unknown"

            cached_models.append(meta)
        except Exception:
            pass

    cached_models.sort(key=lambda x: str(x.get("_raw_timestamp", "")), reverse=True)
    return cached_models


def handle_multi_deletion(selected_hashes: list[str], registry_path: Path, architecture: str) -> None:
    """Handle batch deletion confirmation for multiple selected models.

    Args:
        selected_hashes: List of model hash strings to delete.
        registry_path: Registry root directory path.
        architecture: Display name of the architecture.
    """
    st.warning(f"Selected **{len(selected_hashes)} models**: `{', '.join(selected_hashes)}`")
    col_del_multi, _ = st.columns([2, 4])
    with col_del_multi.popover(
        f"🗑️ Delete {len(selected_hashes)} Models",
        help=f"Move {len(selected_hashes)} selected {architecture} models to Trash",
    ):
        st.warning(f"Move **{len(selected_hashes)}** selected {architecture} models to Trash?")
        st.markdown("\n".join(f"- `{h}`" for h in selected_hashes))
        if st.button(
            f"Move to Trash ({len(selected_hashes)} models)",
            type="primary",
            key=f"btn_confirm_delete_{architecture.lower()}_multi",
        ):
            deleted_cnt = 0
            for h in selected_hashes:
                if delete_cached_model(h, registry_base=registry_path, soft_delete=True):
                    deleted_cnt += 1
            st.session_state.pop(f"_last_{architecture.lower()}_selected_hash", None)
            st.success(f"Moved {deleted_cnt} {architecture} model(s) to Trash (reversible).")
            st.rerun()


def handle_single_deletion(selected_hash: str, registry_path: Path, architecture: str) -> None:
    """Render a popover to delete a single selected model.

    Args:
        selected_hash: The hash of the model to delete.
        registry_path: Registry root directory path.
        architecture: Display name of the architecture.
    """
    with st.popover("🗑️ Delete Model", help=f"Move {architecture} model {selected_hash} to Trash"):
        st.warning(f"Move {architecture} model `{selected_hash}` to Trash (can be restored)?")
        if st.button("Move to Trash", type="primary", key=f"btn_confirm_delete_{architecture.lower()}_single"):
            if delete_cached_model(selected_hash, registry_base=registry_path, soft_delete=True):
                st.session_state.pop(f"_last_{architecture.lower()}_selected_hash", None)
                st.success(f"{architecture} model `{selected_hash}` moved to Trash (reversible).")
                st.rerun()
            else:
                st.error(f"Failed to delete {architecture} model `{selected_hash}`.")


def render_trash_section(registry_path: Path, architecture: str, display_fields: list[str] | None = None) -> None:
    """Render the soft-deleted trash recovery section for models.

    Args:
        registry_path: Registry root directory path.
        architecture: Display name of the architecture.
        display_fields: List of metadata keys to display in the table. If None, default fields are used.
    """
    trashed_models = list_trashed_models(registry_base=registry_path)
    if not trashed_models:
        return

    with st.expander(f"🗑️ Trash / Recently Deleted ({len(trashed_models)} models)", expanded=False):
        st.caption(f"Soft-deleted {architecture} models are safely preserved here and can be restored at any time.")

        # Build display data
        df_trashed = _build_trash_dataframe(trashed_models, display_fields)

        trash_selection = st.dataframe(
            df_trashed,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"{architecture.lower()}_trash_selection",
        )

        trashed_sel_rows = extract_selected_rows(trash_selection)
        trashed_selected_hashes = [
            str(trashed_models[r].get("hash")) for r in trashed_sel_rows if 0 <= r < len(trashed_models)
        ]

        _render_trash_actions(trashed_selected_hashes, trashed_models, registry_path, architecture)


def _build_trash_dataframe(trashed_models: list[dict[str, Any]], display_fields: list[str] | None) -> pd.DataFrame:
    trashed_display = []
    for tm in trashed_models:
        display_dict = {
            "Hash": tm.get("hash", "unknown"),
            "Category": tm.get("category", "unknown"),
            "Created": str(tm.get("timestamp", ""))[:19].replace("T", " ") if tm.get("timestamp") else "Unknown",
        }
        if display_fields:
            for field in display_fields:
                if field in tm:
                    display_dict[field.title()] = tm[field]
        trashed_display.append(display_dict)
    return pd.DataFrame(trashed_display)


def _render_trash_actions(
    trashed_selected_hashes: list[str],
    trashed_models: list[dict[str, Any]],
    registry_path: Path,
    architecture: str,
) -> None:
    col_rest, col_purge, _ = st.columns([2, 2, 4])
    if trashed_selected_hashes:
        if col_rest.button(
            f"♻️ Restore Selected ({len(trashed_selected_hashes)})",
            type="primary",
            key=f"btn_restore_{architecture.lower()}_selected",
        ):
            restored_cnt = sum(
                1 for th in trashed_selected_hashes if restore_cached_model(th, registry_base=registry_path)
            )
            st.success(f"Restored {restored_cnt} {architecture} model(s) back to registry!")
            st.rerun()
    elif col_rest.button("♻️ Restore All Trashed", key=f"btn_restore_{architecture.lower()}_all"):
        all_hashes = [str(tm.get("hash", "")) for tm in trashed_models]
        restored_cnt = sum(1 for th in all_hashes if restore_cached_model(th, registry_base=registry_path))
        st.success(f"Restored all {restored_cnt} {architecture} model(s) back to registry!")
        st.rerun()

    with col_purge.popover(
        "⚠️ Empty Trash (Permanent)",
        help=f"Permanently delete all {architecture} models in Trash",
    ):
        st.error("Are you sure you want to permanently delete these models? This cannot be undone.")
        if st.button("Yes, Empty Trash", type="primary", key=f"btn_purge_{architecture.lower()}_trash"):
            cnt = purge_trash(registry_base=registry_path)
            st.success(f"Permanently purged {cnt} {architecture} model(s) from disk.")
            st.rerun()
