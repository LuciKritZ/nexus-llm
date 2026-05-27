import typing
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from nexus_llm.app import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    # Mock httpx.AsyncClient.send to return a fake StreamingResponse
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


def test_chat_completions_proxy_ollama_old_image(client: TestClient) -> None:
    payload = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Old image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            },
            {"role": "user", "content": "New text"},
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


@patch("nexus_llm.services.gemini_client.GeminiClient.stream_generate_content")
def test_chat_completions_proxy_list_content(mock_stream: AsyncMock, client: TestClient) -> None:
    payload = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<html><body>Explain this image</body></html>"},
                    # valid base64
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            }
        ],
    }

    async def fake_stream(*args: typing.Any, **kwargs: typing.Any) -> AsyncGenerator[bytes, None]:
        yield b"Gemini "
        yield b"response"

    mock_stream.return_value = fake_stream()

    with patch("nexus_llm.services.cache.ImageCache.store") as mock_store:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.content == b"Gemini response"
    mock_store.assert_called_once()


@patch("nexus_llm.services.gemini_client.GeminiClient.stream_generate_content")
def test_chat_completions_proxy_list_content_bad_base64(
    mock_stream: AsyncMock, client: TestClient
) -> None:
    payload = {
        "model": "llama3.2-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<html><body>Explain this image</body></html>"},
                    # invalid base64 length
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw"}},
                ],
            }
        ],
    }

    async def fake_stream(*args: typing.Any, **kwargs: typing.Any) -> AsyncGenerator[bytes, None]:
        yield b"Gemini "
        yield b"response"

    mock_stream.return_value = fake_stream()

    with patch("nexus_llm.services.cache.ImageCache.store") as mock_store:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.content == b"Gemini response"
    mock_store.assert_not_called()


def test_exception_handlers(client: TestClient) -> None:
    from fastapi import FastAPI

    from nexus_llm.exceptions import CacheError, GeminiAPIError, NexusLLMError

    app = typing.cast(FastAPI, client.app)

    @app.get("/test-gemini-error")
    async def gemini_error() -> None:
        raise GeminiAPIError("Rate limit hit")

    @app.get("/test-cache-error")
    async def cache_error() -> None:
        raise CacheError("Disk full")

    @app.get("/test-nexus-error")
    async def nexus_error() -> None:
        raise NexusLLMError("Generic error")

    resp1 = client.get("/test-gemini-error")
    assert resp1.status_code == 502
    assert resp1.json() == {"detail": "Rate limit hit"}

    resp2 = client.get("/test-cache-error")
    assert resp2.status_code == 500
    assert resp2.json() == {"detail": "Disk full"}

    resp3 = client.get("/test-nexus-error")
    assert resp3.status_code == 500
    assert resp3.json() == {"detail": "Generic error"}
