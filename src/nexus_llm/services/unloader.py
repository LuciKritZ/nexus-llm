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

        Execution Flow:
        1. Performs a quick check outside the lock to return early if no change is needed.
        2. Rechecks inside a lock (double-checked locking).
        3. If no active model exists (cold start), sets it and returns.
        4. Triggers the unloading POST request to the Ollama API with keep_alive: 0.
        5. Updates the active model state.

        Args:
            target_model: The model name requested for the upcoming generation.

        Returns:
            True if unloading was triggered, False otherwise.
        """
        if self._active_model == target_model:
            return False

        async with self._lock:
            if self._active_model == target_model:
                return False

            if self._active_model is None:
                self._active_model = target_model
                return False

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
                response = await self._http_client.post(url, json=payload)
                response.raise_for_status()
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()

            self._active_model = target_model
            return True
