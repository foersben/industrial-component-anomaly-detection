"""Unit tests for application core configuration and settings."""

from app.core.config import settings


def test_default_app_settings() -> None:
    """Verify default application settings and environment variables."""
    assert settings.PROJECT_NAME == "Python Project Template"
    assert settings.ENVIRONMENT == "development"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.DEBUG is False
