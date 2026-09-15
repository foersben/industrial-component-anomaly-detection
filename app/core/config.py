"""Validated runtime settings for the industrial component anomaly detection system."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Validated runtime settings for the application.

    Attributes:
        model_config: Model configuration for pydantic-settings.
        PROJECT_NAME: Name of the project.
        ENVIRONMENT: Environment in which the application is running.
        API_V1_STR: API version 1 string.
        DEBUG: Whether the application is running in debug mode.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = Field(default="Python Project Template")
    ENVIRONMENT: str = Field(default="development")
    API_V1_STR: str = Field(default="/api/v1")
    DEBUG: bool = Field(default=False)


settings = AppSettings()
