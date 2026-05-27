import sys
from unittest.mock import patch

from nexus_llm.__main__ import main


def test_main_clear_cache(capsys):
    with patch.object(sys, "argv", ["nexus-llm", "--clear-cache"]):
        main()
    captured = capsys.readouterr()
    assert "Cache cleared!" in captured.out


@patch("nexus_llm.__main__.uvicorn.run")
def test_main_run_server(mock_run):
    with patch.object(sys, "argv", ["nexus-llm", "--port", "1234"]):
        main()
    mock_run.assert_called_once_with(
        "nexus_llm.app:create_app", host="0.0.0.0", port=1234, factory=True
    )
