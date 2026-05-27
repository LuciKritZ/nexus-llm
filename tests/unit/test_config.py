import os
from nexus_llm.config import Settings

def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.port == 11444
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.gemini_api_key is None

def test_settings_override(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    settings = Settings(_env_file=None)
    assert settings.port == 9999
    assert settings.gemini_api_key == "test_key"
