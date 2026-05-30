from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_llm.exceptions import RateLimitError
from nexus_llm.services.multiplexer import Multiplexer
from nexus_llm.services.router_core import NoKeysAvailableError


class MockStreamGen:
    def __init__(self, items: list[str | Exception]) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self) -> "MockStreamGen":
        return self

    async def __anext__(self) -> str:
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def mock_router() -> MagicMock:
    router = MagicMock()

    class MockContextManager:
        async def __aenter__(self) -> None:
            pass

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    router.use_key.return_value = MockContextManager()
    router.get_best_platform_and_key = AsyncMock()
    router.mark_key_exhausted = AsyncMock()
    return router


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_success(mock_get_client: MagicMock, mock_router: MagicMock) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["chunk1", "chunk2"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["chunk1", "chunk2"]
    mock_router.get_best_platform_and_key.assert_called_once_with(["openrouter"])
    mock_router.mark_key_exhausted.assert_not_called()


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_hot_swap_on_error(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_router.get_best_platform_and_key.side_effect = [
        ("openrouter", {"key_hash": "h1", "key_value": "v1"}),
        ("openrouter", {"key_hash": "h2", "key_value": "v2"}),
    ]

    client1 = MagicMock()
    client1.generate_stream.return_value = MockStreamGen([RateLimitError("429", retry_after=30.0)])

    client2 = MagicMock()
    client2.generate_stream.return_value = MockStreamGen(["success_chunk"])

    mock_get_client.side_effect = [client1, client2]

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["success_chunk"]
    assert mock_router.get_best_platform_and_key.call_count == 2
    mock_router.mark_key_exhausted.assert_called_once_with("h1", 30.0)


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_fallback_to_ollama_on_exhaust(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_router.get_best_platform_and_key.side_effect = NoKeysAvailableError("No keys")

    ollama_client = MagicMock()
    ollama_client.generate_stream.return_value = MockStreamGen(["ollama_chunk"])
    mock_get_client.return_value = ollama_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["ollama_chunk"]
    mock_get_client.assert_called_once_with("ollama", "", None, "http://127.0.0.1:11434")
    mock_router.mark_key_exhausted.assert_not_called()


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_fallback_to_ollama_after_max_retries(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    # 3 attempts fail
    client_fail = MagicMock()
    client_fail.generate_stream.side_effect = lambda *args, **kwargs: MockStreamGen(
        [Exception("Error")]
    )

    ollama_client = MagicMock()
    ollama_client.generate_stream.side_effect = lambda *args, **kwargs: MockStreamGen(
        ["ollama_fallback"]
    )

    mock_get_client.side_effect = [client_fail, client_fail, client_fail, ollama_client]

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == ["ollama_fallback"]


def test_get_client_for_platform() -> None:
    from nexus_llm.services.adapters import OpenAICompatibleClient
    from nexus_llm.services.gemini_client import GeminiClient
    from nexus_llm.services.multiplexer import get_client_for_platform

    client1 = get_client_for_platform("gemini", "key1")
    assert isinstance(client1, GeminiClient)

    client2 = get_client_for_platform("openrouter", "key2")
    assert isinstance(client2, OpenAICompatibleClient)

    client3 = get_client_for_platform("ollama", "")
    assert isinstance(client3, OpenAICompatibleClient)

    with pytest.raises(ValueError):
        get_client_for_platform("unknown", "key")


def test_get_client_for_groq() -> None:
    from nexus_llm.services.adapters import OpenAICompatibleClient
    from nexus_llm.services.multiplexer import get_client_for_platform

    client = get_client_for_platform("groq", "test_key")
    assert isinstance(client, OpenAICompatibleClient)


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_empty_stream(mock_get_client: MagicMock, mock_router: MagicMock) -> None:
    mock_router.get_best_platform_and_key.return_value = (
        "openrouter",
        {"key_hash": "h1", "key_value": "v1"},
    )

    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen([])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["openrouter/model"], [])]

    assert chunks == []
    assert mock_router.get_best_platform_and_key.call_count == 1
    assert mock_router.mark_key_exhausted.call_count == 0


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_system_fallback(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["Fallback", " works"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(
        mock_router,
        platforms_data={
            "system_fallback": {"model": "fallback-model", "apiBase": "http://fallback:11434"}
        },
    )
    chunks = [c async for c in multiplexer.generate_stream(["system_fallback"], [])]

    assert chunks == ["Fallback", " works"]
    mock_get_client.assert_called_once_with("ollama", "", None, "http://fallback:11434")
    assert mock_router.get_best_platform_and_key.call_count == 0


@pytest.mark.asyncio
@patch("nexus_llm.services.multiplexer.get_client_for_platform")
async def test_multiplexer_ollama_fast_path(
    mock_get_client: MagicMock, mock_router: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_client.generate_stream.return_value = MockStreamGen(["Ollama", " is", " fast"])
    mock_get_client.return_value = mock_client

    multiplexer = Multiplexer(mock_router)
    chunks = [c async for c in multiplexer.generate_stream(["ollama/qwen"], [])]

    assert chunks == ["Ollama", " is", " fast"]
    assert mock_router.get_best_platform_and_key.call_count == 0
