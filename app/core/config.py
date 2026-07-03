from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Validated runtime settings for the application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = Field(default="Python Project Template")
    ENVIRONMENT: str = Field(default="development")
    API_V1_STR: str = Field(default="/api/v1")
    DEBUG: bool = Field(default=False)


settings = AppSettings()
