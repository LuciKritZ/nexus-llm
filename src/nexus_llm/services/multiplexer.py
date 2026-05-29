import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from nexus_llm.config import settings
from nexus_llm.exceptions import RateLimitError
from nexus_llm.services.adapters import OpenAICompatibleClient
from nexus_llm.services.gemini_client import GeminiClient
from nexus_llm.services.router_core import NoKeysAvailableError, RouterCore

logger = logging.getLogger(__name__)


def get_client_for_platform(
    platform: str, api_key: str, http_client: httpx.AsyncClient | None = None
) -> Any:
    """Factory to get the right client for a platform."""
    if platform == "gemini":
        return GeminiClient(api_key=api_key, client=http_client)
    elif platform == "openrouter":
        return OpenAICompatibleClient(
            api_key=api_key, base_url="https://openrouter.ai/api", client=http_client
        )
    elif platform == "ollama":
        return OpenAICompatibleClient(
            api_key="ollama", base_url=settings.ollama_base_url, client=http_client
        )
    else:
        raise ValueError(f"Unknown platform: {platform}")


class Multiplexer:
    def __init__(
        self, router_core: RouterCore, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self.router_core = router_core
        self.http_client = http_client

    async def generate_stream(
        self, platform: str, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Attempts to generate a stream. If the stream fails before yielding the first chunk,
        it hot-swaps to the next key. If all keys for the platform fail, falls back to Ollama.
        """
        if platform == "ollama":
            ollama_client = get_client_for_platform("ollama", "", self.http_client)
            async for chunk in ollama_client.generate_stream(model, messages, **kwargs):
                yield chunk
            return

        max_retries = 3
        attempts = 0

        while attempts < max_retries:
            try:
                key_info = await self.router_core.get_next_key(platform)
            except NoKeysAvailableError:
                logger.warning(f"No keys available for {platform}. Falling back to Ollama.")
                break

            key_hash = key_info["key_hash"]
            key_value = key_info["key_value"]

            client = get_client_for_platform(platform, key_value, self.http_client)

            try:
                async with self.router_core.use_key(key_hash):
                    stream_gen = client.generate_stream(model, messages, **kwargs)

                    try:
                        first_chunk = await anext(stream_gen)
                    except StopAsyncIteration:
                        return

                    yield first_chunk

                    async for chunk in stream_gen:
                        yield chunk

                    return

            except RateLimitError as e:
                logger.warning(f"Rate limit on {platform} key {key_hash}: {e}. Swapping...")
                await self.router_core.mark_key_exhausted(key_hash, e.retry_after or 60.0)
                attempts += 1
            except Exception as e:
                logger.warning(f"Error on {platform} key {key_hash}: {e}. Swapping...")
                await self.router_core.mark_key_exhausted(key_hash, 60.0)
                attempts += 1

        # Fallback to Ollama if all attempts exhausted or no keys
        logger.info("Falling back to local Ollama model")
        ollama_client = get_client_for_platform("ollama", "", self.http_client)
        async for chunk in ollama_client.generate_stream(settings.ollama_model, messages, **kwargs):
            yield chunk
