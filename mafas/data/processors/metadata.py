"""Document metadata models and extraction."""

import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from pydantic import BaseModel, Field

# Accepted input date formats, tried in order. Day-less formats default to the 1st.
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %Y",
    "%b %Y",
    "%Y%m%d",
    "%m/%d/%Y",
]


def normalize_date(raw: str | None) -> str:
    """Normalise a raw date string to ISO 'YYYY-MM-DD', or 'unknown'.

    Handles ISO dates, 'April 28, 2026', 'January 2024' (-> 2024-01-01),
    compact '20260429', and RFC822 strings from RSS feeds.
    """
    if not raw:
        return "unknown"
    value = raw.strip()
    if not value or value.lower() == "unknown":
        return "unknown"

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    try:
        parsed = parsedate_to_datetime(value)
        if parsed is not None:
            return parsed.strftime("%Y-%m-%d")
    except (TypeError, ValueError, IndexError):
        pass

    return "unknown"


def date_to_int(iso_date: str | None) -> int | None:
    """Convert 'YYYY-MM-DD' to a sortable integer YYYYMMDD, or None."""
    if not iso_date or iso_date == "unknown":
        return None
    try:
        return int(datetime.strptime(iso_date, "%Y-%m-%d").strftime("%Y%m%d"))
    except ValueError:
        return None


class DocumentMetadata(BaseModel):
    """Structured metadata for an ingested financial document."""

    source: str
    doc_type: str
    title: str
    date: str
    tickers: list[str] = Field(default_factory=list)
    word_count: int


class MetadataExtractor:
    """Extracts title, date, and word count from document text."""

    _DATE_PATTERNS = [
        re.compile(
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        re.compile(
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ]

    def extract(
        self,
        text: str,
        source: str,
        doc_type: str,
        **kwargs: Any,
    ) -> DocumentMetadata:
        """Build DocumentMetadata from text and optional kwargs.

        Pass date=<str> to supply an authoritative date (e.g. a filing date or
        RSS published date); it is normalised to ISO. Otherwise the date is
        extracted from the text via regex and normalised.
        """
        title = self._infer_title(text)
        explicit_date = kwargs.get("date")
        if explicit_date:
            date = normalize_date(explicit_date)
            if date == "unknown":
                date = normalize_date(self._extract_date(text))
        else:
            date = normalize_date(self._extract_date(text))
        tickers: list[str] = kwargs.get("tickers", [])
        word_count = len(text.split())
        return DocumentMetadata(
            source=source,
            doc_type=doc_type,
            title=title,
            date=date,
            tickers=tickers,
            word_count=word_count,
        )

    def _infer_title(self, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
        return "Untitled"

    def _extract_date(self, text: str) -> str:
        for pattern in self._DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return "unknown"
