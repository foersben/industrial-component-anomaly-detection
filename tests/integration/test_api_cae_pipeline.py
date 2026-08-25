"""Integration tests for the Keras Convolutional Autoencoder (CAE) FastAPI pipeline endpoint.

This test module verifies the HTTP API contract of the `/api/pipelines/keras_cae` endpoint,
validating request parsing, chained preprocessing execution, deterministic cache hit behavior,
and graceful error handling on invalid requests.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.mark.usefixtures("mock_fast_training")
def test_api_successful_execution_contract(mock_mvtec_dataset: str) -> None:
    """Validate the HTTP 200 response schema and payload structure of the CAE pipeline endpoint.

    Args:
        mock_mvtec_dataset: Path to the temporary mock MVTec dataset root.
    """
    payload = {
        "data_root": mock_mvtec_dataset,
        "category": "bottle",
        "img_size": 32,
        "crop_size": 16,
        "crop_stride": 8,
        "latent_channels": 8,
        "epochs": 1,
        "batch_size": 2,
        "run_heatmap": True,
        "force_retrain": True,
    }

    response = client.post("/api/pipelines/keras_cae", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "results" in data

    results = data["results"]
    assert "image_level" in results
    assert "auroc" in results["image_level"]
    assert isinstance(results["image_level"]["auroc"], float)

    assert "pixel_level" in results
    assert "aupimo_score" in results["pixel_level"]
    assert isinstance(results["pixel_level"]["aupimo_score"], float)

    assert "threshold" in results
    assert isinstance(results["threshold"], float)

    assert "scores" in results
    assert isinstance(results["scores"], list)

    assert "heatmap_overlays" in results
    assert "model_hash" in results
    assert isinstance(results["model_hash"], str)
    assert len(results["model_hash"]) == 12


@pytest.mark.usefixtures("mock_fast_training")
def test_api_chained_preprocessing_steps(mock_mvtec_dataset: str) -> None:
    """Validate that the API accepts and applies chained preprocessing transformations.

    Args:
        mock_mvtec_dataset: Path to the temporary mock MVTec dataset root.
    """
    payload = {
        "data_root": mock_mvtec_dataset,
        "category": "bottle",
        "img_size": 32,
        "crop_size": 16,
        "crop_stride": 8,
        "latent_channels": 8,
        "epochs": 1,
        "batch_size": 2,
        "preprocessing_steps": [
            {"name": "foreground_mask", "params": {}},
            {"name": "clahe", "params": {}},
        ],
    }

    response = client.post("/api/pipelines/keras_cae", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "results" in data

    results = data["results"]
    assert "image_level" in results
    assert "auroc" in results["image_level"]
    assert "pixel_level" in results
    assert "aupimo_score" in results["pixel_level"]


@pytest.mark.usefixtures("mock_fast_training")
def test_api_cache_hit_and_evaluation_integrity(mock_mvtec_dataset: str) -> None:
    """Verify that cached models are correctly reloaded via API and produce identical metrics.

    Args:
        mock_mvtec_dataset: Path to the temporary mock MVTec dataset root.
    """
    payload = {
        "data_root": mock_mvtec_dataset,
        "category": "bottle",
        "img_size": 32,
        "crop_size": 16,
        "crop_stride": 8,
        "latent_channels": 8,
        "epochs": 1,
        "batch_size": 2,
        "force_retrain": True,
    }

    # 1. First request: Force training and obtain model hash
    response_1 = client.post("/api/pipelines/keras_cae", json=payload)
    assert response_1.status_code == 200
    data_1 = response_1.json()
    model_hash = data_1["results"]["model_hash"]

    # 2. Second request: Reuse cached model by specifying model hash
    payload["force_retrain"] = False
    payload["model_hash"] = model_hash
    response_2 = client.post("/api/pipelines/keras_cae", json=payload)
    assert response_2.status_code == 200
    data_2 = response_2.json()

    # Verify identical evaluation results
    assert data_1["results"]["threshold"] == data_2["results"]["threshold"]
    assert data_1["results"]["image_level"]["auroc"] == data_2["results"]["image_level"]["auroc"]


def test_api_error_validation_handling(mock_mvtec_dataset: str) -> None:
    """Verify that invalid categories produce an HTTP error response.

    Args:
        mock_mvtec_dataset: Path to the temporary mock MVTec dataset root.
    """
    payload = {
        "data_root": mock_mvtec_dataset,
        "category": "non_existent_category",
        "img_size": 32,
        "crop_size": 16,
        "crop_stride": 8,
        "latent_channels": 8,
        "epochs": 1,
        "batch_size": 2,
    }

    response = client.post("/api/pipelines/keras_cae", json=payload)
    assert response.status_code == 500
