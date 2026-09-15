"""Standardized Streamlit selectors for dataset directories and categories."""

from collections.abc import Callable
from typing import Any

import streamlit as st

from app.domain.categories import discover_dataset_categories


def render_dataset_and_category_selector(
    key_prefix: str,
    default_root: str = "data/raw/mvtec_ad",
    default_category: str = "bottle",
    on_category_change: Callable[..., Any] | None = None,
) -> tuple[str, str]:
    """Render standardized two-column dataset root directory input and dynamic category selector.

    Automatically scans the dataset root directory on disk to dynamically populate
    available category folders, while falling back cleanly to canonical MVTec AD
    benchmark categories.

    Args:
        key_prefix: Prefix used for Streamlit widget keys and session state (e.g. 'kcae', 'b', 'dino').
        default_root: Default path string for the dataset root input.
        default_category: Default category name if available.
        on_category_change: Optional callback function triggered when the user changes the category.

    Returns:
        A tuple of (dataset_root_directory_string, selected_category_string).
    """
    root_key = f"{key_prefix}_root"
    cat_key = f"{key_prefix}_cat"

    st.session_state.setdefault(root_key, default_root)

    col1, col2 = st.columns(2)
    data_root = str(col1.text_input("Dataset Root Directory", key=root_key))

    categories = discover_dataset_categories(data_root)

    current_cat = st.session_state.get(cat_key, default_category)
    fallback_cat = default_category if default_category in categories else (categories[0] if categories else "bottle")
    if current_cat not in categories:
        st.session_state[cat_key] = fallback_cat
    else:
        st.session_state.setdefault(cat_key, fallback_cat)

    cat_index = categories.index(st.session_state[cat_key]) if st.session_state[cat_key] in categories else 0

    category = str(
        col2.selectbox(
            "Category Name",
            options=categories,
            index=cat_index,
            key=cat_key,
            on_change=on_category_change,
        )
    )

    return data_root, category
