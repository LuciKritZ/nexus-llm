from nexus_llm.utils.html import compress_html_to_markdown


class ContextCompressor:
    """Intercepts and compresses raw HTML payloads to protect the local LLM context window."""

    def __init__(self, char_threshold: int = 16384, min_reduction_ratio: float = 0.65) -> None:
        self.char_threshold = char_threshold
        self.min_reduction_ratio = min_reduction_ratio

    def compress_if_needed(self, text: str) -> str:
        """
        Evaluates the text and compresses it if it exceeds the character threshold
        and successfully reduces the size by the minimum reduction ratio.

        Execution Flow:
        1. Skips if text is under the character threshold.
        2. Checks for basic HTML indicators to avoid parsing massive plain text files.
        3. Attempts compression via HTML to Markdown conversion.
        4. Measures reduction. Falls back to original text if the payload was completely stripped
           or if the compression didn't meet the minimum reduction ratio.
        """
        original_length = len(text)

        if original_length <= self.char_threshold:
            return text

        if "<html" not in text.lower() and "<body" not in text.lower() and "</" not in text:
            return text

        compressed_text = compress_html_to_markdown(text)
        compressed_length = len(compressed_text)

        if compressed_length == 0:
            return text

        reduction = 1.0 - (compressed_length / original_length)

        if reduction >= self.min_reduction_ratio:
            return compressed_text

        return text
