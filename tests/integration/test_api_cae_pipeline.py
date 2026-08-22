from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch

from app.api.main import app

@pytest.fixture(autouse=True)
def mock_train_cae():
    def _mock_train_cae(model, *args, **kwargs):
        return {"train": [0.1], "val_good": [0.1], "val_anomalous": [0.2]}
    with patch("app.pipelines.multi_stage_ae.cae_pipeline.train_cae", side_effect=_mock_train_cae):
        yield

client = TestClient(app, raise_server_exceptions=False)

def test_api_successful_execution_contract(mock_mvtec_dataset: str) -> None:
    """Test successful execution of the keras CAE pipeline via the API."""
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

    # We should have heatmap overlays since we have anomalous samples
    assert "heatmap_overlays" in results

    assert "model_hash" in results
    assert isinstance(results["model_hash"], str)
    assert len(results["model_hash"]) == 12

def test_api_chained_preprocessing_steps(mock_mvtec_dataset: str) -> None:
    """Test API pipeline with chained preprocessing steps."""
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
            {"name": "clahe", "params": {}}
        ]
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

def test_api_cache_hit_and_evaluation_integrity(mock_mvtec_dataset: str) -> None:
    """Test that cache hits correctly bypass retraining and maintain evaluation integrity."""
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

    # 1. First request, forces retrain
    response_1 = client.post("/api/pipelines/keras_cae", json=payload)
    assert response_1.status_code == 200
    data_1 = response_1.json()
    model_hash = data_1["results"]["model_hash"]

    # 2. Second request, use cache
    payload["force_retrain"] = False
    payload["model_hash"] = model_hash
    response_2 = client.post("/api/pipelines/keras_cae", json=payload)
    assert response_2.status_code == 200
    data_2 = response_2.json()

    # Assert model loaded from cache has identical threshold (should evaluate same data identically)
    assert data_1["results"]["threshold"] == data_2["results"]["threshold"]
    assert data_1["results"]["image_level"]["auroc"] == data_2["results"]["image_level"]["auroc"]

def test_api_error_validation_handling(mock_mvtec_dataset: str) -> None:
    """Test API pipeline handles invalid categories gracefully."""
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

    # Expecting the ValueError to result in a 500 from FastAPI because we disabled raise_server_exceptions
    assert response.status_code == 500
