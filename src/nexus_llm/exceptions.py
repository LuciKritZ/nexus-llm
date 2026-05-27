class NexusLLMError(Exception):
    """Base exception for all domain-specific errors in Nexus LLM."""


class GeminiAPIError(NexusLLMError):
    """Raised when the Gemini API responds with an error (e.g., 429, 500, 503)."""


class CacheError(NexusLLMError):
    """Raised when the image caching layer fails to read or write to disk."""
