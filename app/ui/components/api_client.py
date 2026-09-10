"""API client utilities for Streamlit UI."""

from typing import Any

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"


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
