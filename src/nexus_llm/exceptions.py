class NexusLLMError(Exception):
    """Base exception for all domain-specific errors in Nexus LLM."""


class GeminiAPIError(NexusLLMError):
    """Raised when the Gemini API responds with an error (e.g., 429, 500, 503)."""


class CacheError(NexusLLMError):
    """Raised when the image caching layer fails to read or write to disk."""


class RateLimitError(NexusLLMError):
    """Raised when an API responds with 429 Too Many Requests."""

    def __init__(self, message: str, retry_after: int | float = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(NexusLLMError):
    """Raised when an API responds with 402 or 403 indicating out of credits/quota."""
