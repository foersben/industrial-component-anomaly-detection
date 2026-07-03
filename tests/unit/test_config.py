from app.core.config import settings


def test_default_app_settings() -> None:
    """Validates default app settings."""
    assert settings.PROJECT_NAME == "Python Project Template"
    assert settings.ENVIRONMENT == "development"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.DEBUG is False
