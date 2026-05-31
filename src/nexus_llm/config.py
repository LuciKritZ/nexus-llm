from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    port: int = 11444

    sqlite_db_path: str = "nexus_llm_state.db"
    models_json_path: str = "models.json"
    default_cooldown_minutes: int = 60
    image_token_reserve: int = 4096
    proxy_password: str | None = None


settings = Settings()
