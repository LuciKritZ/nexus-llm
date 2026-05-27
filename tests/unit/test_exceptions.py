from nexus_llm.exceptions import CacheError, GeminiAPIError, NexusLLMError


def test_nexus_llm_error() -> None:
    error = NexusLLMError("base error")
    assert str(error) == "base error"
    assert isinstance(error, Exception)


def test_gemini_api_error() -> None:
    error = GeminiAPIError("rate limit")
    assert str(error) == "rate limit"
    assert isinstance(error, NexusLLMError)


def test_cache_error() -> None:
    error = CacheError("disk full")
    assert str(error) == "disk full"
    assert isinstance(error, NexusLLMError)
