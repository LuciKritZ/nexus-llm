from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    port: int = 11444
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str | None = None


settings = Settings()
