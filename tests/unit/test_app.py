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

    async def mock_generate_stream(
        *args: typing.Any, **kwargs: typing.Any
    ) -> AsyncGenerator[str, None]:
        yield "Hello"
        yield " World"

    monkeypatch.setattr(
        "nexus_llm.services.multiplexer.Multiplexer.generate_stream", mock_generate_stream
    )

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


@patch("nexus_llm.routes.proxy.settings")
def test_chat_completions_proxy_ollama_model_override(
    mock_settings: typing.Any, client: TestClient
) -> None:
    mock_settings.ollama_model = "forced-model"
    mock_settings.ollama_base_url = "http://127.0.0.1:11434"
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


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


def test_chat_completions_proxy_list_content(client: TestClient) -> None:
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

    with patch("nexus_llm.services.cache.ImageCache.store") as mock_store:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    content = response.content.decode("utf-8")
    assert "Hello" in content
    assert " World" in content
    mock_store.assert_called_once()


def test_chat_completions_proxy_list_content_bad_base64(client: TestClient) -> None:
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

    with patch("nexus_llm.services.cache.ImageCache.store") as mock_store:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Hello" in content
    assert " World" in content
    mock_store.assert_not_called()


@patch("nexus_llm.routes.proxy.settings")
@patch("nexus_llm.services.gatekeeper.Gatekeeper.classify")
def test_chat_completions_proxy_nexus_auto(
    mock_classify: AsyncMock, mock_settings: typing.Any, client: TestClient
) -> None:
    mock_settings.ollama_model = None
    mock_classify.return_value = "complex"
    payload = {
        "model": "nexus-auto",
        "messages": [{"role": "user", "content": "Complex query"}],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    mock_classify.assert_called_once()

    # test simple routing
    mock_classify.reset_mock()
    mock_classify.return_value = "simple"
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    mock_classify.assert_called_once()


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
