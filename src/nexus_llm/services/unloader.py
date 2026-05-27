import asyncio
import sys

import httpx


class ModelUnloader:
    """Tracks the active model in Ollama VRAM and handles unloading on model switch."""

    def __init__(self, ollama_url: str, http_client: httpx.AsyncClient | None = None) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self._http_client = http_client
        self._active_model: str | None = None
        self._lock = asyncio.Lock()

    async def unload_if_needed(self, target_model: str) -> bool:
        """Unloads the active model if it differs from the target model.

        Args:
            target_model: The model name requested for the upcoming generation.

        Returns:
            True if unloading was triggered, False otherwise.
        """
        # Quick check outside the lock
        if self._active_model == target_model:
            return False

        async with self._lock:
            # Recheck after acquiring lock (double-checked locking)
            if self._active_model == target_model:
                return False

            if self._active_model is None:
                # Cold start: first model load has no previous active model to unload
                self._active_model = target_model
                return False

            # Perform unloading of self._active_model
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self._active_model,
                "keep_alive": 0,
            }

            print(
                f"Unloading model '{self._active_model}' to load '{target_model}'",
                file=sys.stderr,
            )

            if self._http_client is not None:
                # Use provided client (e.g. mock or shared client)
                response = await self._http_client.post(url, json=payload)
                response.raise_for_status()
            else:
                # Fallback to creating a client
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()

            self._active_model = target_model
            return True
