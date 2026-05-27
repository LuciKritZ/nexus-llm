import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from nexus_llm.config import settings
from nexus_llm.exceptions import GeminiAPIError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with the Google Gemini API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.api_key = settings.gemini_api_key

    def _convert_openai_to_gemini(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Converts an OpenAI ChatCompletion payload to Gemini format."""
        contents = []

        for message in payload.get("messages", []):
            role = message.get("role", "user")
            # Gemini maps 'assistant' to 'model'
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System instructions in Gemini are typically handled via a top-level field,
                # but for simplicity in chat history we can map them to 'user' or omit.
                # According to latest docs, we map it to user if we just inline it.
                role = "user"

            parts: list[dict[str, Any]] = []
            content = message.get("content", "")

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text" and "text" in block:
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image_url" and "image_url" in block:
                        url = block["image_url"].get("url", "")
                        if url.startswith("data:"):
                            # Parse data:image/jpeg;base64,xxxxx
                            # Format: data:[<mediatype>][;base64],<data>
                            header, b64_data = url.split(",", 1)
                            mime_type = header.split(";")[0].removeprefix("data:")
                            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})

            if parts:
                contents.append({"role": role, "parts": parts})

        return {"contents": contents}

    async def stream_generate_content(self, payload: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        """Sends the payload to Gemini and yields raw SSE bytes, with retry logic."""
        gemini_payload = self._convert_openai_to_gemini(payload)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:streamGenerateContent"

        headers = {"x-goog-api-key": self.api_key or ""}
        params = {"alt": "sse"}

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # We use stream() instead of send() because it's a context manager for streaming
                async with self.client.stream(
                    "POST",
                    url,
                    json=gemini_payload,
                    headers=headers,
                    params=params,
                    timeout=30.0,
                ) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        response.raise_for_status()

                    if response.status_code != 200:
                        body = await response.aread()
                        raise GeminiAPIError(
                            f"Gemini API returned {response.status_code}: {body.decode()}"
                        )

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if not data_str:
                                continue
                            if data_str == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                continue

                            try:
                                data_json = json.loads(data_str)
                                text = ""
                                if data_json.get("candidates"):
                                    candidate = data_json["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                text += part["text"]
                                if text:
                                    openai_chunk = {
                                        "id": "chatcmpl-gemini",
                                        "object": "chat.completion.chunk",
                                        "choices": [{"delta": {"content": text}}],
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                            except json.JSONDecodeError:
                                pass

                # If we finish streaming successfully, return to exit the retry loop
                return

            except httpx.HTTPStatusError as e:
                # Catch specific status codes for retries
                if e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise GeminiAPIError(f"Max retries reached. Last error: {e}") from e
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Gemini API returned {e.response.status_code}, retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    raise GeminiAPIError(f"Unexpected HTTPStatusError: {e}") from e
            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    raise GeminiAPIError(f"Network error after {max_retries} attempts: {e}") from e
                delay = base_delay * (2**attempt)
                logger.warning(f"Network error {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)
