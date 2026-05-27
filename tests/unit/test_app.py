import typing
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nexus_llm.app import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> typing.Generator[TestClient, None, None]:
    # Mock httpx.AsyncClient.send to return a fake StreamingResponse
    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")

    async def mock_aiter_bytes() -> typing.AsyncGenerator[bytes, None]:
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
    # We must mock aiter_bytes on the response instance
    mock_response.aiter_bytes = mock_aiter_bytes  # type: ignore

    mock_send = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    # We also need to mock post for the unloader
    mock_post_response = httpx.Response(status_code=200, request=req, json={"status": "success"})
    mock_post = AsyncMock(return_value=mock_post_response)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_chat_completions_proxy(client: TestClient) -> None:
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "<html><body>Hi</body></html>"}],
    }
    # This should trigger the unloader, then the compressor, and then stream the response.
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.content.decode("utf-8")
    assert "Hello" in content
    assert " World" in content


def test_chat_completions_proxy_list_content(client: TestClient) -> None:
    payload = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<html><body>Explain this image</body></html>"},
                    {"type": "image_url", "image_url": {"url": "base64..."}},
                ],
            }
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
