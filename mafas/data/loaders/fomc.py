"""Federal Reserve FOMC meeting minutes loader."""

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import fitz
import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from data.processors.cleaner import TextCleaner
from data.processors.metadata import MetadataExtractor

USER_AGENT = (
    "Mozilla/5.0 (compatible; MAFAS/1.0; +https://github.com/mafas-project)"
)
FOMC_CALENDAR_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
# Only fetch minutes from the official Federal Reserve domain.
_FOMC_ALLOWED_HOST = "www.federalreserve.gov"


class FOMCLoader:
    """Downloads and parses FOMC meeting minutes PDFs from federalreserve.gov."""

    def __init__(self, cache_dir: str = "./data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    def _fetch_html(self, url: str, timeout: float) -> str:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    def _fetch_bytes(self, url: str, timeout: float) -> bytes:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    @staticmethod
    def _pdf_url_for_date(date_str: str) -> str:
        """Build the canonical minutes PDF URL for a YYYYMMDD meeting date.

        The Fed calendar links minutes as an HTML page
        (``/monetarypolicy/fomcminutesYYYYMMDD.htm``), but the actual PDF lives
        under the ``/files/`` sub-directory. Naively swapping ``.htm`` for
        ``.pdf`` yields a 404, so we always reconstruct the ``/files/`` path.
        """
        return (
            f"https://{_FOMC_ALLOWED_HOST}/monetarypolicy/files/"
            f"fomcminutes{date_str}.pdf"
        )

    def fetch_minutes_page(self) -> list[dict]:
        """Return available FOMC minutes as canonical PDF URLs, one per meeting."""
        try:
            html = self._fetch_html(FOMC_CALENDAR_URL, timeout=15.0)
        except Exception as exc:
            logger.error("Failed to fetch FOMC calendar: {}", exc)
            return []

        soup = BeautifulSoup(html, "lxml")
        results: list[dict] = []
        seen_dates: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "fomcminutes" not in href:
                continue
            if href.startswith("http"):
                url = href
            else:
                url = f"https://{_FOMC_ALLOWED_HOST}{href}"
            # Allowlist: only accept URLs on the official Fed domain.
            if urlparse(url).hostname != _FOMC_ALLOWED_HOST:
                logger.warning("Skipping off-domain FOMC URL: {}", url)
                continue
            # The calendar lists both the .htm minutes page and the /files/*.pdf
            # for the same meeting; dedupe by meeting date so each meeting is
            # fetched exactly once from its canonical PDF URL.
            match = re.search(r"fomcminutes(\d{8})", url)
            if not match:
                continue
            date_str = match.group(1)
            if date_str in seen_dates:
                continue
            seen_dates.add(date_str)
            results.append(
                {"url": self._pdf_url_for_date(date_str), "date_str": date_str}
            )

        return results

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"fomc_{digest}.pdf"

    def load_minutes_pdf(self, url: str) -> dict | None:
        """Download (or load cached) PDF and return cleaned text with metadata."""
        # Accept either a canonical /files/*.pdf URL or a legacy .htm page and
        # normalise to the canonical PDF location.
        date_match = re.search(r"fomcminutes(\d{8})", url)
        if url.lower().endswith(".pdf"):
            pdf_url = url
        elif date_match:
            pdf_url = self._pdf_url_for_date(date_match.group(1))
        else:
            pdf_url = url.replace(".htm", ".pdf")

        try:
            cache_path = self._cache_path(pdf_url)

            if cache_path.exists():
                pdf_bytes = cache_path.read_bytes()
            else:
                pdf_bytes = self._fetch_bytes(pdf_url, timeout=30.0)
                cache_path.write_bytes(pdf_bytes)

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = [page.get_text() for page in doc]
            doc.close()
            raw_text = "\n".join(pages_text)
            cleaned = self.cleaner.clean(raw_text)

            if not self.cleaner.is_meaningful(cleaned, min_words=200):
                logger.warning("FOMC document too short, skipping: {}", url)
                return None

            # The meeting date is encoded in the URL (fomcminutesYYYYMMDD).
            authoritative_date = date_match.group(1) if date_match else None

            metadata = self.metadata_extractor.extract(
                cleaned,
                source=pdf_url,
                doc_type="fomc_minutes",
                date=authoritative_date,
            )
            return {"text": cleaned, "metadata": metadata.model_dump()}
        except httpx.HTTPStatusError as exc:
            # 404 simply means the minutes for a recent meeting have not been
            # published yet (they trail the meeting by ~3 weeks) — not an error.
            if exc.response.status_code == 404:
                logger.info("FOMC minutes not yet published: {}", pdf_url)
            else:
                logger.error("HTTP error loading FOMC minutes {}: {}", pdf_url, exc)
            return None
        except Exception as exc:
            logger.error("Failed to load FOMC minutes from {}: {}", pdf_url, exc)
            return None

    def load_recent(self, n: int = 5) -> list[dict]:
        """Load the n most recent FOMC minutes documents.

        Iterates newest-first and keeps going past meetings whose minutes are
        not yet published, so we always return up to ``n`` real documents.
        """
        pages = self.fetch_minutes_page()
        pages.sort(key=lambda x: x["date_str"], reverse=True)
        documents: list[dict] = []
        for entry in pages:
            if len(documents) >= n:
                break
            loaded = self.load_minutes_pdf(entry["url"])
            if loaded is not None:
                documents.append(loaded)
        return documents
