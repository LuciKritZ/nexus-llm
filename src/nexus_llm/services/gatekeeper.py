import logging
from typing import Any

import httpx

from nexus_llm.config import settings

logger = logging.getLogger(__name__)


class Gatekeeper:
    """Evaluates payload complexity and routes to simple or complex model tiers."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self.http_client = http_client

    async def classify(self, payload: dict[str, Any]) -> str:
        """
        Classifies the request as 'simple' or 'complex'.
        Uses fast heuristics first, falling back to LLM-as-a-judge.
        """
        messages = payload.get("messages", [])
        if not messages:
            return "simple"

        total_length = sum(len(str(m.get("content", ""))) for m in messages)
        system_prompts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]

        latest_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_msg = str(m.get("content", ""))
                break

        # Fast Heuristic 1: Title generation, short commits, or simple summaries
        lower_msg = latest_user_msg.lower()
        if total_length < 2000 and any(
            keyword in lower_msg for keyword in ["commit message", "summarize", "generate a title"]
        ):
            return "simple"

        # Fast Heuristic 2: IDE Fingerprinting (Cursor, Copilot, Cline)
        # If the system prompt is massive, it's an IDE agent. Agents usually need complex models.
        if any(len(sp) > 3000 for sp in system_prompts):
            return "complex"

        # Fast Heuristic 3: Huge context usually requires a complex model
        if total_length > 8000:
            return "complex"

        # Fast Heuristic 4: Very short queries are usually simple
        if len(messages) <= 2 and total_length < 200:
            return "simple"

        # Fallback: LLM-as-a-judge using local Ollama model
        judge_prompt = (
            "Analyze the following user request and classify its complexity.\n"
            "If it is a simple greeting, basic question, formatting fix, or straightforward "
            "task, reply with 'simple'.\n"
            "If it requires complex reasoning, architecture design, deep debugging, or "
            "multi-file refactoring, reply with 'complex'.\n"
            "Only reply with the exact word 'simple' or 'complex'.\n\n"
            f"User Request: {latest_user_msg[:1000]}"
        )

        try:
            response = await self.http_client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": judge_prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
                timeout=1.5,
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip().lower()
            if "complex" in result:
                return "complex"
            return "simple"
        except httpx.TimeoutException:
            logger.warning("Gatekeeper judge timed out, defaulting to complex")
            return "complex"
        except Exception as e:
            logger.warning(f"Gatekeeper judge failed ({e}), defaulting to complex")
            return "complex"
