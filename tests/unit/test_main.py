import sys
import unittest.mock
from unittest.mock import patch

import pytest

from nexus_llm.__main__ import main


@patch("nexus_llm.__main__.ImageCache.clear")
def test_main_clear_cache(
    mock_clear: unittest.mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch.object(sys, "argv", ["nexus-llm", "--clear-cache"]):
        main()
    captured = capsys.readouterr()
    assert "Cache cleared!" in captured.out
    mock_clear.assert_called_once()


@patch("nexus_llm.__main__.uvicorn.run")
@patch("nexus_llm.__main__.settings")
def test_main_run_server(
    mock_settings: unittest.mock.MagicMock, mock_run: unittest.mock.MagicMock
) -> None:
    mock_settings.proxy_password = "secure_password"
    mock_settings.port = 11444
    with patch.object(sys, "argv", ["nexus-llm", "--port", "1234"]):
        main()
    mock_run.assert_called_once_with(
        "nexus_llm.app:create_app", host="0.0.0.0", port=1234, factory=True
    )


@patch("nexus_llm.__main__.uvicorn.run")
@patch("nexus_llm.__main__.settings")
def test_main_password_missing(
    mock_settings: unittest.mock.MagicMock,
    mock_run: unittest.mock.MagicMock,
) -> None:
    mock_settings.proxy_password = None
    with patch.object(sys, "argv", ["nexus-llm", "--port", "1234"]), pytest.raises(SystemExit):
        main()
    mock_run.assert_not_called()
