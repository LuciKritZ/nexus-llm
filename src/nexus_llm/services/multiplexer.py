import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from nexus_llm.exceptions import RateLimitError
from nexus_llm.services.adapters import OpenAICompatibleClient
from nexus_llm.services.gemini_client import GeminiClient
from nexus_llm.services.router_core import NoKeysAvailableError, RouterCore

logger = logging.getLogger(__name__)


def get_client_for_platform(
    platform: str,
    api_key: str,
    http_client: httpx.AsyncClient | None = None,
    api_base: str | None = None,
) -> Any:
    """Factory to get the right client for a platform."""
    if platform == "gemini":
        return GeminiClient(api_key=api_key, client=http_client)
    elif platform == "openrouter":
        return OpenAICompatibleClient(
            api_key=api_key, base_url=api_base or "https://openrouter.ai/api", client=http_client
        )
    elif platform == "ollama":
        return OpenAICompatibleClient(
            api_key="ollama", base_url=api_base or "http://127.0.0.1:11434", client=http_client
        )
    elif platform == "groq":
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=api_base or "https://api.groq.com/openai",
            client=http_client,
        )
    else:
        raise ValueError(f"Unknown platform: {platform}")


class Multiplexer:
    def __init__(
        self,
        router_core: RouterCore,
        http_client: httpx.AsyncClient | None = None,
        platforms_data: dict[str, Any] | None = None,
    ) -> None:
        self.router_core = router_core
        self.http_client = http_client
        self.platforms_data = platforms_data or {}

    async def generate_stream(
        self,
        candidate_models: list[str],
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Attempts to generate a stream across multiple candidate models dynamically.
        """
        if "system_fallback" in candidate_models:
            logger.info("Using system fallback model")
            fallback_info = self.platforms_data.get("system_fallback", {})
            fallback_model = fallback_info.get("model", "llama3")
            fallback_url = fallback_info.get("apiBase", "http://127.0.0.1:11434")

            ollama_client = get_client_for_platform("ollama", "", self.http_client, fallback_url)
            async for chunk in ollama_client.generate_stream(fallback_model, messages, **kwargs):
                yield chunk
            return

        # Map platforms to their corresponding target models from the candidates list
        platform_to_models: dict[str, list[tuple[str, str]]] = {}
        for c in candidate_models:
            parts = c.split("/", 1)
            p = parts[0]
            m = parts[1] if len(parts) > 1 else parts[0]

            # Special case for explicit ollama
            if p == "ollama":
                logger.info(f"Using explicit ollama model {m}")
                ollama_client = get_client_for_platform("ollama", "", self.http_client, None)
                async for chunk in ollama_client.generate_stream(m, messages, **kwargs):
                    yield chunk
                return

            if p not in platform_to_models:
                platform_to_models[p] = []
            platform_to_models[p].append((c, m))

        candidate_platforms = list(platform_to_models.keys())
        max_retries = max(3, len(candidate_platforms) * 2) if candidate_platforms else 0
        attempts = 0

        while attempts < max_retries and candidate_platforms:
            try:
                platform, key_info = await self.router_core.get_best_platform_and_key(
                    candidate_platforms
                )
            except NoKeysAvailableError:
                logger.warning(f"No keys available for any of {candidate_platforms}. Falling back.")
                break

            key_hash = key_info["key_hash"]
            key_value = key_info["key_value"]

            # Pick the first model matched for this platform
            full_model_key, target_model = platform_to_models[platform][0]
            api_base = self.platforms_data.get(full_model_key, {}).get("apiBase")

            client = get_client_for_platform(platform, key_value, self.http_client, api_base)

            try:
                async with self.router_core.use_key(key_hash):
                    stream_gen = client.generate_stream(target_model, messages, **kwargs)

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
        logger.info("Falling back to system fallback model")
        fallback_info = self.platforms_data.get("system_fallback", {})
        fallback_model = fallback_info.get("model", "llama3")
        fallback_url = fallback_info.get("apiBase", "http://127.0.0.1:11434")

        ollama_client = get_client_for_platform("ollama", "", self.http_client, fallback_url)
        async for chunk in ollama_client.generate_stream(fallback_model, messages, **kwargs):
            yield chunk
