import abc
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from nexus_llm.exceptions import QuotaExceededError, RateLimitError


class BaseLLMClient(abc.ABC):
    """Abstract base class for all LLM provider adapters."""

    def __init__(self, api_key: str, base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abc.abstractmethod
    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generates a streaming response from the LLM provider."""
        yield ""

    @abc.abstractmethod
    def handle_error(self, response: httpx.Response) -> None:
        """Parses an HTTP error response and raises the appropriate domain exception."""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """Client adapter for OpenAI-compatible APIs (OpenRouter, Groq, Ollama, vLLM)."""

    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", headers=headers, json=payload
            ) as response,
        ):
            if response.status_code != 200:
                self.handle_error(response)

            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

    def handle_error(self, response: httpx.Response) -> None:
        if response.status_code == 429:
            # Try to parse standard reset headers
            reset_time = response.headers.get("x-ratelimit-reset") or response.headers.get(
                "retry-after"
            )
            try:
                retry_after = float(reset_time) if reset_time else 60.0
            except ValueError:
                retry_after = 60.0
            raise RateLimitError(f"Rate limited by {self.base_url}", retry_after=retry_after)

        if response.status_code in (402, 403):
            # 402 is standard for Insufficient Quota (e.g. OpenRouter). Some use 403.
            raise QuotaExceededError(f"Quota exceeded or unauthorized for {self.base_url}")

        # Fallback for generic errors
        response.raise_for_status()
