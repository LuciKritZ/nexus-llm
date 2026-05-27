from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nexus_llm.app import create_app


@pytest.fixture
def mock_httpx_send(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock httpx.AsyncClient.send to return a fake StreamingResponse immediately
    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")

    async def mock_aiter_bytes() -> AsyncGenerator[bytes, None]:
        yield (
            b'{"id":"chatcmpl-123","object":"chat.completion.chunk",'
            b'"choices":[{"delta":{"content":"Hello"}}]}'
        )
        yield (
            b'{"id":"chatcmpl-123","object":"chat.completion.chunk",'
            b'"choices":[{"delta":{"content":" World"}}]}'
        )

    mock_response = httpx.Response(
        status_code=200, request=req, headers={"content-type": "text/event-stream"}
    )
    mock_response.aiter_bytes = mock_aiter_bytes  # type: ignore

    mock_send = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    # Mock unloader
    mock_post_response = httpx.Response(status_code=200, request=req, json={"status": "success"})
    mock_post = AsyncMock(return_value=mock_post_response)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_ttft_benchmark(benchmark: Any, mock_httpx_send: None, test_client: TestClient) -> None:
    """
    Measures the Time-To-First-Token (TTFT) overhead of the proxy route.
    The requirement is < 15ms.
    """
    payload = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": "<html><body>Hello! I have a very long context here.</body></html>",
            }
        ],
    }

    def measure_ttft() -> None:
        with test_client.stream("POST", "/v1/chat/completions", json=payload) as response:
            for _ in response.iter_bytes():
                # Break on the first token yielded to accurately measure TTFT
                break

    # We tell pytest-benchmark to run this function
    benchmark(measure_ttft)

    # We can assert that the average TTFT is under 15ms (0.015 seconds)
    # The benchmark fixture provides stats
    assert benchmark.stats.stats.mean < 0.015, f"TTFT exceeded 15ms: {benchmark.stats.stats.mean}s"
