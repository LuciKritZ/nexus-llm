import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Gatekeeper:
    """Evaluates payload complexity and routes to simple or complex model tiers."""

    def __init__(
        self, http_client: httpx.AsyncClient, platforms_data: dict[str, Any] | None = None
    ) -> None:
        self.http_client = http_client
        self.platforms_data = platforms_data or {}

    async def profile_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Profiles the request for context length and vision requirements.
        """
        messages = payload.get("messages", [])
        if not messages:
            return {"context_length": 0, "has_image": False}

        # Context length approximation
        total_length = sum(len(str(m.get("content", ""))) for m in messages)

        # Vision requirement (has_image is handled primarily by proxy)
        # To strictly answer the user requirement, gatekeeper can just calculate context_length.

        return {"context_length": total_length}
