import pytest

from omnimux.config import Settings


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.port == 11444
    assert settings.sqlite_db_path == "omnimux_state.db"


def test_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9999")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.port == 9999
