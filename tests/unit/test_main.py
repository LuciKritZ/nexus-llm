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
def test_main_run_server(mock_run: unittest.mock.MagicMock) -> None:
    with patch.object(sys, "argv", ["nexus-llm", "--port", "1234"]):
        main()
    mock_run.assert_called_once_with(
        "nexus_llm.app:create_app", host="0.0.0.0", port=1234, factory=True
    )
