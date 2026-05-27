from nexus_llm.services.compressor import ContextCompressor


def test_compressor_skips_short_text() -> None:
    """Should return exact original text since it's under 16,384 chars."""
    compressor = ContextCompressor()
    text = "<html><body>" + "a" * 100 + "</body></html>"
    result = compressor.compress_if_needed(text)
    assert result == text


def test_compressor_compresses_large_html() -> None:
    """
    Strips nav and footer from a large HTML string and returns the main content in markdown.
    Must reduce the character count by at least 65%.
    """
    compressor = ContextCompressor()
    html = (
        "<html><body><nav>"
        + "nav_content " * 1000
        + "</nav><main>Main content</main><footer>"
        + "footer_content " * 1000
        + "</footer></body></html>"
    )

    result = compressor.compress_if_needed(html)

    assert "Main content" in result
    assert "nav_content" not in result
    assert "footer_content" not in result

    reduction = 1.0 - (len(result) / len(html))
    assert reduction >= 0.65


def test_compressor_returns_original_if_not_compressible() -> None:
    """
    If the reduction isn't at least 65% (e.g. string has no HTML boilerplate),
    it should return the original text to prevent data loss.
    """
    compressor = ContextCompressor()
    # Include basic HTML to bypass the early exit check (line 28)
    text = "<html><body>" + "Just some very long text. " * 1000 + "</body></html>"
    assert len(text) > 16384

    result = compressor.compress_if_needed(text)

    assert result == text


def test_compressor_skips_non_html_text() -> None:
    """
    If the string is over 16,384 characters but has no HTML indicators,
    it should return the original text immediately.
    """
    compressor = ContextCompressor()
    text = "Just some very long plain text. " * 1000
    assert len(text) > 16384

    result = compressor.compress_if_needed(text)

    assert result == text


def test_compressor_handles_fully_stripped_payload() -> None:
    """
    If the HTML is completely stripped (e.g. only script tags),
    it should fallback to the original to prevent complete data loss.
    """
    compressor = ContextCompressor()
    # Over 16384 chars of pure script tags
    text = "<html><body><script>" + "var x = 1;\n" * 2000 + "</script></body></html>"
    assert len(text) > 16384

    result = compressor.compress_if_needed(text)

    assert result == text
