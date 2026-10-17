import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_llm.__main__ import main


@patch("nexus_llm.__main__.ImageCache.clear")
def test_main_clear_cache(mock_clear: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["nexus", "--clear-cache"]):
        main()
    captured = capsys.readouterr()
    assert "Cache cleared!" in captured.out
    mock_clear.assert_called_once()


@patch("nexus_llm.__main__.ImageCache.clear")
def test_main_cache_subcommand(mock_clear: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["nexus", "cache", "--clear"]):
        main()
    captured = capsys.readouterr()
    assert "Cache cleared!" in captured.out
    mock_clear.assert_called_once()


@patch("nexus_llm.__main__.uvicorn.run")
def test_main_start_subcommand(mock_run: MagicMock) -> None:
    with patch.object(sys, "argv", ["nexus", "start", "--port", "1234", "--host", "127.0.0.1"]):
        main()
    mock_run.assert_called_once_with(
        "nexus_llm.app:create_app", host="127.0.0.1", port=1234, factory=True
    )


@patch("nexus_llm.__main__.uvicorn.run")
def test_main_default_action_starts(mock_run: MagicMock) -> None:
    with patch.object(sys, "argv", ["nexus"]):
        main()
    mock_run.assert_called_once()


@patch("nexus_llm.__main__.create_client_key", new_callable=AsyncMock)
@patch("nexus_llm.__main__._get_db", new_callable=AsyncMock)
def test_main_keys_create(
    mock_get_db: AsyncMock,
    mock_create: AsyncMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_db = AsyncMock()
    mock_get_db.return_value = mock_db
    mock_create.return_value = ("nx_sample_token", {"name": "app1"})

    with patch.object(sys, "argv", ["nexus", "keys", "create", "app1"]):
        main()

    captured = capsys.readouterr()
    assert "nx_sample_token" in captured.out
    assert "Created client API key for 'app1'" in captured.out


@patch("nexus_llm.__main__.list_client_keys", new_callable=AsyncMock)
@patch("nexus_llm.__main__._get_db", new_callable=AsyncMock)
def test_main_keys_list(
    mock_get_db: AsyncMock,
    mock_list: AsyncMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_db = AsyncMock()
    mock_get_db.return_value = mock_db

    # 1. Empty list
    mock_list.return_value = []
    with patch.object(sys, "argv", ["nexus", "keys", "list"]):
        main()
    assert "No client keys found" in capsys.readouterr().out

    # 2. Non-empty list
    mock_list.return_value = [
        {
            "id": 1,
            "name": "app1",
            "prefix": "nx_abc...",
            "status": "active",
            "created_at": "2026-10-17",
        }
    ]
    with patch.object(sys, "argv", ["nexus", "keys", "list"]):
        main()
    out = capsys.readouterr().out
    assert "nx_abc..." in out
    assert "app1" in out


@patch("nexus_llm.__main__.revoke_client_key", new_callable=AsyncMock)
@patch("nexus_llm.__main__._get_db", new_callable=AsyncMock)
def test_main_keys_revoke(
    mock_get_db: AsyncMock,
    mock_revoke: AsyncMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_db = AsyncMock()
    mock_get_db.return_value = mock_db

    mock_revoke.return_value = True
    with patch.object(sys, "argv", ["nexus", "keys", "revoke", "1"]):
        main()
    assert "successfully revoked" in capsys.readouterr().out

    mock_revoke.return_value = False
    with patch.object(sys, "argv", ["nexus", "keys", "revoke", "nonexistent"]):
        main()
    assert "not found or already revoked" in capsys.readouterr().out


@patch("nexus_llm.__main__.run_add_model_wizard", new_callable=AsyncMock)
@patch("nexus_llm.__main__.list_configured_models", new_callable=AsyncMock)
@patch("nexus_llm.__main__._get_db", new_callable=AsyncMock)
def test_main_model_subcommands(
    mock_get_db: AsyncMock,
    mock_list: AsyncMock,
    mock_wizard: AsyncMock,
) -> None:
    mock_db = AsyncMock()
    mock_get_db.return_value = mock_db

    with patch.object(sys, "argv", ["nexus", "model", "add"]):
        main()
    mock_wizard.assert_called_once()

    with patch.object(sys, "argv", ["nexus", "model", "list"]):
        main()
    mock_list.assert_called_once()


@patch("nexus_llm.__main__.export_configuration", new_callable=AsyncMock)
@patch("nexus_llm.__main__._get_db", new_callable=AsyncMock)
def test_main_export_subcommand(
    mock_get_db: AsyncMock,
    mock_export: AsyncMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_db = AsyncMock()
    mock_get_db.return_value = mock_db

    with patch.object(sys, "argv", ["nexus", "export", "my_export.json"]):
        main()
    mock_export.assert_called_once_with(mock_db, "my_export.json")
    assert "Exported configuration to my_export.json" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_main_get_db(tmp_path: Any) -> None:
    from nexus_llm.__main__ import _get_db
    from nexus_llm.config import settings

    test_db_path = str(tmp_path / "main_db.db")
    with patch.object(settings, "sqlite_db_path", test_db_path):
        db = await _get_db()
        assert db is not None
        await db.close()


@patch("nexus_llm.__main__.subprocess.Popen")
def test_main_start_detached(
    mock_popen: MagicMock, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import nexus_llm.__main__ as main_mod

    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_popen.return_value = mock_proc

    test_pid = tmp_path / ".test.pid"
    test_log = tmp_path / "test.log"
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch.object(main_mod, "LOG_FILE", test_log),
        patch.object(sys, "argv", ["nexus", "start", "-d"]),
    ):
        main()
        assert test_pid.exists()
        assert test_pid.read_text().strip() == "99999"
        assert "Nexus proxy started in detached mode (PID: 99999)" in capsys.readouterr().out


def test_main_start_detached_already_running(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import nexus_llm.__main__ as main_mod

    test_pid = tmp_path / ".test.pid"
    test_pid.write_text("12345")
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch("os.kill", return_value=None),
        patch.object(sys, "argv", ["nexus", "start", "-d"]),
    ):
        main()
        assert "Nexus proxy is already running" in capsys.readouterr().out


@patch("nexus_llm.__main__.subprocess.Popen")
def test_main_start_detached_stale_pid(
    mock_popen: MagicMock, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import nexus_llm.__main__ as main_mod

    mock_proc = MagicMock()
    mock_proc.pid = 88888
    mock_popen.return_value = mock_proc

    test_pid = tmp_path / ".test.pid"
    test_pid.write_text("not-a-pid")
    test_log = tmp_path / "test.log"
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch.object(main_mod, "LOG_FILE", test_log),
        patch.object(sys, "argv", ["nexus", "start", "-d"]),
    ):
        main()
        assert test_pid.read_text().strip() == "88888"


def test_main_stop_no_pid(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    import nexus_llm.__main__ as main_mod

    test_pid = tmp_path / ".nonexistent.pid"
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch.object(sys, "argv", ["nexus", "stop"]),
    ):
        main()
        assert "No active Nexus PID file found" in capsys.readouterr().out


def test_main_stop_success(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    import nexus_llm.__main__ as main_mod

    test_pid = tmp_path / ".test.pid"
    test_pid.write_text("54321")
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch("os.kill") as mock_kill,
        patch.object(sys, "argv", ["nexus", "stop"]),
    ):
        main()
        mock_kill.assert_called_once()
        assert not test_pid.exists()
        assert "Stopped Nexus proxy (PID: 54321)" in capsys.readouterr().out


def test_main_stop_process_lookup_error(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    import nexus_llm.__main__ as main_mod

    test_pid = tmp_path / ".test.pid"
    test_pid.write_text("54321")
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch("os.kill", side_effect=ProcessLookupError),
        patch.object(sys, "argv", ["nexus", "stop"]),
    ):
        main()
        assert not test_pid.exists()
        assert "Process not found. Cleaning up stale PID file" in capsys.readouterr().out


def test_main_stop_error(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    import nexus_llm.__main__ as main_mod

    test_pid = tmp_path / ".test.pid"
    test_pid.write_text("54321")
    with (
        patch.object(main_mod, "PID_FILE", test_pid),
        patch("os.kill", side_effect=PermissionError("denied")),
        patch.object(sys, "argv", ["nexus", "stop"]),
    ):
        main()
        assert not test_pid.exists()
        assert "Error stopping proxy: denied" in capsys.readouterr().out
