from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nexus_llm.exceptions import GeminiAPIError
from nexus_llm.services.gemini_client import GeminiClient


def test_convert_openai_to_gemini() -> None:
    client = GeminiClient(httpx.AsyncClient())
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is in this image?"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this one?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE"},
                    },
                ],
            },
            {"role": "assistant", "content": "It is a pixel."},
        ]
    }

    gemini_format = client._convert_openai_to_gemini(payload)

    expected = {
        "contents": [
            {"role": "user", "parts": [{"text": "You are a helpful assistant."}]},
            {"role": "user", "parts": [{"text": "What is in this image?"}]},
            {
                "role": "user",
                "parts": [
                    {"text": "And this one?"},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAE",
                        }
                    },
                ],
            },
            {"role": "model", "parts": [{"text": "It is a pixel."}]},
        ]
    }

    assert gemini_format == expected


@pytest.mark.asyncio
async def test_stream_generate_content_success() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_aiter_bytes() -> AsyncGenerator[bytes, None]:
        yield b"chunk1"
        yield b"chunk2"

    mock_response.aiter_bytes = mock_aiter_bytes

    # stream() returns an async context manager
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(mock_client)

    chunks = []
    async for chunk in client.stream_generate_content({"messages": []}):
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2"]
    mock_client.stream.assert_called_once()


@pytest.mark.asyncio
@patch("nexus_llm.services.gemini_client.asyncio.sleep", new_callable=AsyncMock)
async def test_stream_generate_content_retry_on_429(mock_sleep: AsyncMock) -> None:
    # First response: 429
    mock_response_429 = MagicMock()
    mock_response_429.status_code = 429

    request = httpx.Request("POST", "url")
    error = httpx.HTTPStatusError(
        "429 Too Many Requests", request=request, response=mock_response_429
    )
    mock_response_429.raise_for_status.side_effect = error

    # Second response: 200 OK
    mock_response_200 = MagicMock()
    mock_response_200.status_code = 200

    async def mock_aiter_bytes() -> AsyncGenerator[bytes, None]:
        yield b"success_chunk"

    mock_response_200.aiter_bytes = mock_aiter_bytes
    mock_response_200.raise_for_status.return_value = None

    # We need to simulate __aenter__ returning different responses sequentially
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=[mock_response_429, mock_response_200])
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(mock_client)

    chunks = []
    async for chunk in client.stream_generate_content({"messages": []}):
        chunks.append(chunk)

    assert chunks == [b"success_chunk"]
    assert mock_client.stream.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@pytest.mark.asyncio
@patch("nexus_llm.services.gemini_client.asyncio.sleep", new_callable=AsyncMock)
async def test_stream_generate_content_max_retries(mock_sleep: AsyncMock) -> None:
    mock_response_500 = MagicMock()
    mock_response_500.status_code = 500

    request = httpx.Request("POST", "url")
    error = httpx.HTTPStatusError("500 Server Error", request=request, response=mock_response_500)
    mock_response_500.raise_for_status.side_effect = error

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response_500)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(mock_client)

    with pytest.raises(GeminiAPIError, match="Max retries reached"):
        async for _ in client.stream_generate_content({"messages": []}):
            pass

    assert mock_client.stream.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_stream_generate_content_non_retryable_error() -> None:
    mock_response_400 = MagicMock()
    mock_response_400.aread = AsyncMock()
    mock_response_400.status_code = 400

    request = httpx.Request("POST", "url")
    _ = httpx.HTTPStatusError("400 Bad Request", request=request, response=mock_response_400)

    # raise_for_status() is not called for 400, but we raise GeminiAPIError manually
    mock_response_400.raise_for_status.return_value = None
    mock_response_400.aread.return_value = b"Bad payload"

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response_400)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(mock_client)

    with pytest.raises(GeminiAPIError, match="Gemini API returned 400: Bad payload"):
        async for _ in client.stream_generate_content({"messages": []}):
            pass

    assert mock_client.stream.call_count == 1


@pytest.mark.asyncio
@patch("nexus_llm.services.gemini_client.asyncio.sleep", new_callable=AsyncMock)
async def test_stream_generate_content_request_error_retries(mock_sleep: AsyncMock) -> None:
    request = httpx.Request("POST", "url")
    error = httpx.RequestError("Network unreachable", request=request)

    mock_context = MagicMock()
    # Mocking __aenter__ to raise the RequestError, simulating stream() failing
    mock_context.__aenter__ = AsyncMock(side_effect=error)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(mock_client)

    with pytest.raises(GeminiAPIError, match="Network error after 3 attempts"):
        async for _ in client.stream_generate_content({"messages": []}):
            pass

    assert mock_client.stream.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_stream_generate_content_unexpected_http_error() -> None:
    request = httpx.Request("POST", "url")
    mock_response = MagicMock()
    mock_response.status_code = 403
    error = httpx.HTTPStatusError("403 Forbidden", request=request, response=mock_response)

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=error)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context

    client = GeminiClient(mock_client)

    with pytest.raises(GeminiAPIError, match="Unexpected HTTPStatusError: 403 Forbidden"):
        async for _ in client.stream_generate_content({"messages": []}):
            pass
