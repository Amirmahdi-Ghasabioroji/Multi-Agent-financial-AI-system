"""Document metadata models and extraction."""

import re
from typing import Any

from pydantic import BaseModel, Field


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
        """Build DocumentMetadata from text and optional kwargs."""
        title = self._infer_title(text)
        date = self._extract_date(text)
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
