from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DEEP_COVER_AGENT_",
        extra="ignore",
    )

    internal_agent_secret: str = "dev-agent-secret"
    java_base_url: str = "http://localhost:8080"
    request_timeout_seconds: float = 10.0

    deepseek_api_key: SecretStr | None = Field(default=None)
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_temperature: float = 0.7
    deepseek_max_retries: int = 2
    deepseek_timeout_seconds: float = 30.0

    enable_langchain: bool = True
    idle_check_interval_seconds: float = 5.0
    idle_speech_after_seconds: float = 30.0
    message_history_limit: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
