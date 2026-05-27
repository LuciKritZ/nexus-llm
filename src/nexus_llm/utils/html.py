from bs4 import BeautifulSoup
from markdownify import markdownify


def compress_html_to_markdown(html_string: str) -> str:
    """
    Strips layout boilerplate and converts HTML to Markdown to reduce context bloat.

    Execution Flow:
    1. Returns empty string if input is empty.
    2. Uses BeautifulSoup to remove script, style, and structural tags (nav, header, footer, aside)
       that don't add semantic text value for LLMs.
    3. Converts the cleaned HTML to Markdown using markdownify.
    4. Strips multiple consecutive blank lines to save tokens.
    """
    if not html_string or not html_string.strip():
        return ""

    soup = BeautifulSoup(html_string, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    cleaned_html = str(soup)

    md_text: str = markdownify(cleaned_html, heading_style="ATX", strip=["img"])

    lines = [line.strip() for line in md_text.splitlines() if line.strip()]
    return "\n".join(lines)
