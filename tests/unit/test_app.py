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


def test_chat_completions_proxy_ollama_model_override(client: TestClient) -> None:
    pass  # Test removed because settings.ollama_model is deprecated


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


@patch("nexus_llm.services.gatekeeper.Gatekeeper.profile_request")
def test_chat_completions_proxy_nexus_auto(mock_profile: AsyncMock, client: TestClient) -> None:
    mock_profile.return_value = {"context_length": 100, "has_image": False}
    payload = {
        "model": "nexus-auto",
        "messages": [{"role": "user", "content": "Complex query"}],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    mock_profile.assert_called_once()


@patch("nexus_llm.services.gatekeeper.Gatekeeper.profile_request")
def test_chat_completions_proxy_coverage(mock_profile: AsyncMock, client: TestClient) -> None:
    app = typing.cast(typing.Any, client.app)
    app.state.platforms = {
        "gemini": {"max_input_tokens": 1000, "supports_vision": True},
        "ollama": {"max_input_tokens": 100, "supports_vision": False},
        "system_fallback": {"model": "llama3"},
    }
    # 1. auto with context too large, triggers fallback because no candidates
    mock_profile.return_value = {"context_length": 99999999, "has_image": False}
    response = client.post("/v1/chat/completions", json={"model": "auto", "messages": []})
    assert response.status_code == 200

    # 2. auto with image, but max_tokens is fine, triggers skip of non-vision models
    mock_profile.return_value = {"context_length": 10, "has_image": True}
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg,base64"}}
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200

    # 3. Explicit fallback_model
    fallback_model = "llama3"
    # Hit line 34 (fallback_model present and explicitly requested)
    response = client.post("/v1/chat/completions", json={"model": fallback_model, "messages": []})
    assert response.status_code == 200

    # 4. Explicit model with slash
    response = client.post(
        "/v1/chat/completions",
        json={"model": "openrouter/meta-llama/llama-3-70b-instruct", "messages": []},
    )
    assert response.status_code == 200

    # 5. Explicit model without slash that does not exist in platforms, fallback to ollama
    response = client.post(
        "/v1/chat/completions", json={"model": "non-existent-model", "messages": []}
    )
    assert response.status_code == 200


def test_lifespan_platforms_json() -> None:
    from unittest.mock import mock_open, patch

    from fastapi.testclient import TestClient

    from nexus_llm.app import create_app

    # 1. platforms.json exists and is valid
    app = create_app()
    mock_json = '{"system_fallback": {"model": "test"}}'
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=mock_json)),
        TestClient(app),
    ):
        pass

    # 2. platforms.json exists but is invalid JSON (hits except pass block)
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="invalid json")),
        TestClient(app),
    ):
        pass


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
