"""Text cleaning and deduplication utilities for document ingestion."""

import hashlib
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


class TextCleaner:
    """Normalises, sanitises, and filters raw document text."""

    def clean(self, text: str) -> str:
        """Normalise unicode, strip control chars, collapse whitespace."""
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace("\x00", "")
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {3,}", " ", text)
        return text.strip()

    def _parse_markup(self, html: str) -> BeautifulSoup:
        """Parse HTML or SEC XHTML/XML without spurious parser warnings."""
        head = html.lstrip()[:500].lower()
        # SEC EDGAR filings are XHTML/XML; RSS/news summaries are plain HTML.
        if head.startswith("<?xml") or "xmlns=" in head or "<xbrl" in head:
            return BeautifulSoup(html, features="xml")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            return BeautifulSoup(html, "lxml")

    def clean_html(self, html: str) -> str:
        """Parse HTML, remove boilerplate tags, extract and clean text."""
        soup = self._parse_markup(html)
        for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        body = soup.body or soup
        return self.clean(body.get_text(separator="\n", strip=True))

    def is_meaningful(self, text: str, min_words: int = 50) -> bool:
        """Return True if cleaned text has at least min_words words."""
        cleaned = self.clean(text)
        return len(cleaned.split()) >= min_words

    def deduplicate(self, texts: list[str]) -> list[str]:
        """Remove exact duplicates while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for text in texts:
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            if digest not in seen:
                seen.add(digest)
                result.append(text)
        return result
