import json
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from omnimux.app import create_app
from omnimux.models.schemas import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    convert_anthropic_to_chat_completion,
)
from omnimux.services.adapters import AnthropicClient


@pytest.fixture
def test_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    mock_post_response = httpx.Response(status_code=200, request=req, json={"status": "success"})
    monkeypatch.setattr(httpx.AsyncClient, "post", AsyncMock(return_value=mock_post_response))

    app = create_app()
    with TestClient(app) as client:
        yield client


def test_reproduce_stream_false_chat_completions(test_client: TestClient) -> None:
    # Task 1: /v1/chat/completions with stream=False must return application/json chat.completion,
    # not text/event-stream StreamingResponse
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "Hello"
            yield " world"

        mock_stream.return_value = mock_gen()

        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("application/json")
        data = res.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["finish_reason"] == "stop"


def test_reproduce_convert_anthropic_preserves_tools_and_top_p() -> None:
    # Task 2: convert_anthropic_to_chat_completion must preserve tools and top_p
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }
    ]
    req = AnthropicMessagesRequest(
        model="claude-3-5-sonnet",
        messages=[AnthropicMessage(role="user", content="What's the weather?")],
        tools=tools,
        top_p=0.9,
    )
    converted = convert_anthropic_to_chat_completion(req)
    assert getattr(converted, "tools", None) == tools
    assert getattr(converted, "top_p", None) == 0.9


def test_reproduce_openai_sse_chunk_has_index(test_client: TestClient) -> None:
    # Task 3: OpenAI SSE chunks must include "index": 0 in choices
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:

        async def mock_gen(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
            yield "chunk1"

        mock_stream.return_value = mock_gen()

        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert res.status_code == 200
        lines = [
            line
            for line in res.text.split("\n")
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        assert len(lines) > 0
        chunk_data = json.loads(lines[0][6:])
        assert "index" in chunk_data["choices"][0]
        assert chunk_data["choices"][0]["index"] == 0


@pytest.mark.asyncio
async def test_reproduce_anthropic_client_forwards_anthropic_beta() -> None:
    # Task 4: AnthropicClient forwards anthropic-beta header
    client = AnthropicClient("https://api.anthropic.com", "test-key")
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        async def aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "text_delta", "text": "ok"}}'
            )

        mock_resp.aiter_lines = aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_resp

        chunks = []
        async for c in client.generate_stream(
            "claude-3-5-sonnet",
            [{"role": "user", "content": "hi"}],
            anthropic_beta="prompt-caching-2024-07-31",
        ):
            chunks.append(c)

        _, kwargs = mock_stream.call_args
        assert "anthropic-beta" in kwargs["headers"]
        assert kwargs["headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"


@pytest.mark.asyncio
async def test_reproduce_anthropic_client_thinking_and_partial_json() -> None:
    # Task 5: AnthropicClient yields thinking_delta and partial_json
    client = AnthropicClient("https://api.anthropic.com", "test-key")
    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        async def aiter_lines() -> AsyncGenerator[str, None]:
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "thinking_delta", "thinking": "Let me think."}}'
            )
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "input_json_delta", "partial_json": "{\\"arg\\": 1}"}}'
            )
            yield (
                'data: {"type": "content_block_delta", "delta": '
                '{"type": "text_delta", "text": "Done"}}'
            )

        mock_resp.aiter_lines = aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_resp

        chunks = []
        async for c in client.generate_stream(
            "claude-3-5-sonnet",
            [{"role": "user", "content": "hi"}],
        ):
            chunks.append(c)

        assert "Let me think." in chunks
        assert '{"arg": 1}' in chunks
        assert "Done" in chunks


def test_reproduce_non_data_image_url_preserved(test_client: TestClient) -> None:
    # Task 6: Non-data image URLs (http://, https://) should not become [Image: unknown_image]
    with patch("omnimux.services.multiplexer.Multiplexer.generate_stream") as mock_stream:
        captured_messages = []

        async def mock_gen(
            candidate_models: list[str],
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> AsyncGenerator[str, None]:
            captured_messages.extend(messages)
            yield "Analysis"

        mock_stream.side_effect = mock_gen

        # Historical message with web image URL
        res = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/photo.png"},
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    },
                    {"role": "user", "content": "Next question"},
                ],
                "stream": True,
            },
        )
        assert res.status_code == 200
        # Historical message image should not be "unknown_image"
        first_content = str(captured_messages[0]["content"])
        assert "unknown_image" not in first_content


def test_reproduce_count_tokens_includes_system_and_tools(test_client: TestClient) -> None:
    # Task 7: /v1/messages/count_tokens must factor in system prompt and tools
    res_base = test_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    tokens_base = res_base.json()["input_tokens"]

    res_with_tools = test_client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup database records",
                    "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
                }
            ],
        },
    )
    tokens_with_tools = res_with_tools.json()["input_tokens"]
    assert tokens_with_tools > tokens_base
