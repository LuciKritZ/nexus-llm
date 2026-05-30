import typing
from collections.abc import AsyncGenerator

import aiosqlite
import httpx
import pytest
import pytest_asyncio
from httpx import Response

from nexus_llm.app import create_app, lifespan

pytestmark = pytest.mark.asyncio(loop_scope="function")


class MockStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        yield b'data: {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}\n\n'
        yield b'data: {"candidates": [{"content": {"parts": [{"text": "World"}]}}]}\n\n'


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> Response:
        self.call_count += 1

        auth = request.headers.get("x-goog-api-key")

        # Simulate a 429 for the first key on the second turn
        if auth == "val1" and self.call_count > 1:
            return Response(
                429,
                content=b'{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}',
                request=request,
            )

        return Response(200, stream=MockStream(), request=request)


@pytest_asyncio.fixture
async def e2e_app(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[httpx.AsyncClient, None]:
    # Setup mock transport for all external gemini calls via httpx.AsyncClient
    original_send = httpx.AsyncClient.send

    transport = MockTransport()

    async def mock_send(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs: typing.Any
    ) -> Response:
        print(f"\\n[mock_send] URL: {request.url}")
        if "gemini" in str(request.url):
            print(f"[mock_send] Gemini call with auth: {request.headers.get('x-goog-api-key')}")
            res = await transport.handle_async_request(request)
            print(f"[mock_send] Returning status {res.status_code}")
            return res
        return await original_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", mock_send)

    app = create_app()

    # Trigger lifespan explicitly
    async with lifespan(app):
        # Inject our mock model after lifespan loads platforms.json
        app.state.platforms["gemini/gemini-1.5-pro"] = {
            "max_input_tokens": 100000,
            "supports_vision": True,
        }

        from nexus_llm.config import settings

        # Connect to the DB to insert 2 gemini keys
        async with aiosqlite.connect(settings.sqlite_db_path) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO api_keys
                (platform, key_hash, key_value, priority, last_used_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("gemini", "hash1", "val1", 1, "2023-01-01T00:00:00+00:00"),
                    ("gemini", "hash2", "val2", 1, "2023-01-01T00:00:01+00:00"),
                ],
            )
            await db.commit()

        # Create a client pointing to the ASGI app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


async def test_e2e_multi_turn_with_key_rotation(e2e_app: httpx.AsyncClient) -> None:
    client = e2e_app

    # Turn 1
    payload: dict[str, typing.Any] = {
        "model": "gemini-1.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg,dummy"}},
                ],
            }
        ],
        "stream": True,
    }

    content = b""
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        async for chunk in response.aiter_bytes():
            content += chunk

    assert b"Hello " in content
    assert b"World" in content

    # Turn 2 - This will trigger 429 on key1, multiplexer should rotate to key2 automatically and
    # it will retry seamlessly
    payload["messages"].append({"role": "assistant", "content": "Hello World"})
    payload["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "How are you?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg,dummy"}},
            ],
        }
    )

    content2 = b""
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response2:
        assert response2.status_code == 200
        async for chunk in response2.aiter_bytes():
            content2 += chunk

    assert b"Hello " in content2
    assert b"World" in content2
