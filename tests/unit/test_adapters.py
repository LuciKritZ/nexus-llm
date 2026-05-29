from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

from nexus_llm.exceptions import QuotaExceededError, RateLimitError
from nexus_llm.services.adapters import BaseLLMClient, OpenAICompatibleClient


@pytest.fixture
def openai_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(api_key="test-key", base_url="https://api.test")


@pytest.mark.asyncio
async def test_openai_client_rate_limit(openai_client: OpenAICompatibleClient) -> None:
    # Simulate a 429 response from an OpenAI-compatible API
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, headers={"x-ratelimit-reset": "120"}, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        openai_client.handle_error(response)

    assert exc_info.value.retry_after == 120


@pytest.mark.asyncio
async def test_openai_client_rate_limit_default_reset(
    openai_client: OpenAICompatibleClient,
) -> None:
    # Simulate a 429 response without a reset header (should default to 60)
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        openai_client.handle_error(response)

    assert exc_info.value.retry_after == 60


@pytest.mark.asyncio
async def test_openai_client_quota_exceeded(openai_client: OpenAICompatibleClient) -> None:
    # Simulate a 402 response (insufficient credits)
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(402, request=request)

    with pytest.raises(QuotaExceededError):
        openai_client.handle_error(response)


@pytest.mark.asyncio
async def test_base_llm_client_abstract_methods() -> None:
    # Test that the abstract base class methods don't crash when called generically
    class DummyClient(BaseLLMClient):
        async def generate_stream(
            self, model: str, messages: list[dict[str, Any]], **kwargs: Any
        ) -> AsyncGenerator[str, None]:
            async for chunk in BaseLLMClient.generate_stream(self, model, messages, **kwargs):
                yield chunk

        def handle_error(self, response: httpx.Response) -> None:
            BaseLLMClient.handle_error(self, response)

    dummy = DummyClient("key", "url")
    dummy.handle_error(httpx.Response(200))

    chunks = [c async for c in dummy.generate_stream("m", [])]
    assert chunks == [""]


@pytest.mark.asyncio
async def test_openai_client_rate_limit_invalid_reset(
    openai_client: OpenAICompatibleClient,
) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(429, headers={"x-ratelimit-reset": "invalid-float"}, request=request)

    with pytest.raises(RateLimitError) as exc_info:
        openai_client.handle_error(response)

    assert exc_info.value.retry_after == 60.0


@pytest.mark.asyncio
async def test_openai_client_generic_error(openai_client: OpenAICompatibleClient) -> None:
    request = httpx.Request("POST", "https://api.test")
    response = httpx.Response(500, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        openai_client.handle_error(response)


@pytest.mark.asyncio
async def test_openai_client_generate_stream(
    openai_client: OpenAICompatibleClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # We mock httpx.AsyncClient.stream to simulate the async generator yielding lines
    import contextlib

    class MockStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines
            self.status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            for line in self.lines:
                yield line

    @contextlib.asynccontextmanager
    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MockStreamResponse, None]:
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]",
        ]
        yield MockStreamResponse(lines)

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    chunks = []
    async for chunk in openai_client.generate_stream(
        "model-1", [{"role": "user", "content": "hi"}]
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_openai_client_generate_stream_error(
    openai_client: OpenAICompatibleClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    class MockStreamResponse:
        def __init__(self) -> None:
            self.status_code = 402
            self.headers: dict[str, str] = {}
            self.request = httpx.Request("POST", "https://api.test")

        def raise_for_status(self) -> None:
            pass

    @contextlib.asynccontextmanager
    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MockStreamResponse, None]:
        yield MockStreamResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    with pytest.raises(QuotaExceededError):
        async for _ in openai_client.generate_stream("model-1", []):
            pass


@pytest.mark.asyncio
async def test_openai_client_generate_stream_invalid_json(
    openai_client: OpenAICompatibleClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    class MockStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines
            self.status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            for line in self.lines:
                yield line

    @contextlib.asynccontextmanager
    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[MockStreamResponse, None]:
        lines = ['data: {"invalid_json":', "data: [DONE]"]
        yield MockStreamResponse(lines)

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    chunks = []
    async for chunk in openai_client.generate_stream("model-1", []):
        chunks.append(chunk)

    assert chunks == []
