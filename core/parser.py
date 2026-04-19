from dataclasses import dataclass, field
from html.parser import HTMLParser as _HTMLParser
from urllib.parse import urljoin, urlparse


@dataclass
class ParseResult:
    title: str
    text: str
    links: list[str] = field(default_factory=list)


class _ContentExtractor(_HTMLParser):
    """Extracts visible text, title, and links from HTML using stdlib only."""

    SKIP_TAGS = frozenset({"script", "style", "noscript"})

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True

        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self._links.append(value)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if not stripped:
            return
        self._text_parts.append(stripped)
        if self._in_title:
            self._title_parts.append(stripped)

    def get_result(self) -> tuple[str, str, list[str]]:
        title = " ".join(self._title_parts)
        text = " ".join(self._text_parts)
        return title, text, self._links


def parse_html(html_content: str, base_url: str) -> ParseResult:
    """Parse HTML content and extract title, visible text, and absolute HTTP(S) links."""
    extractor = _ContentExtractor()
    try:
        extractor.feed(html_content)
    except Exception:
        pass

    title, text, raw_links = extractor.get_result()

    links: list[str] = []
    for href in raw_links:
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            clean = parsed._replace(fragment="").geturl()
            if clean not in links:
                links.append(clean)

    return ParseResult(title=title, text=text, links=links)
