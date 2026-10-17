from omnimux.exceptions import CacheError, GeminiAPIError, NexusLLMError, OmniMuxError


def test_omnimux_error() -> None:
    error = OmniMuxError("base error")
    assert str(error) == "base error"
    assert isinstance(error, Exception)
    assert NexusLLMError is OmniMuxError


def test_gemini_api_error() -> None:
    error = GeminiAPIError("rate limit")
    assert str(error) == "rate limit"
    assert isinstance(error, NexusLLMError)


def test_cache_error() -> None:
    error = CacheError("disk full")
    assert str(error) == "disk full"
    assert isinstance(error, NexusLLMError)
