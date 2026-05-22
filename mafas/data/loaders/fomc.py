"""Federal Reserve FOMC meeting minutes loader."""

import hashlib
import re
from pathlib import Path

import fitz
import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from data.processors.cleaner import TextCleaner
from data.processors.metadata import MetadataExtractor

USER_AGENT = (
    "Mozilla/5.0 (compatible; MAFAS/1.0; +https://github.com/mafas-project)"
)
FOMC_CALENDAR_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)


class FOMCLoader:
    """Downloads and parses FOMC meeting minutes PDFs from federalreserve.gov."""

    def __init__(self, cache_dir: str = "./data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _fetch_html(self, url: str, timeout: float) -> str:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _fetch_bytes(self, url: str, timeout: float) -> bytes:
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    def fetch_minutes_page(self) -> list[dict]:
        """Return available FOMC minutes page URLs and date strings."""
        try:
            html = self._fetch_html(FOMC_CALENDAR_URL, timeout=15.0)
        except Exception as exc:
            logger.error("Failed to fetch FOMC calendar: {}", exc)
            return []

        soup = BeautifulSoup(html, "lxml")
        results: list[dict] = []
        seen_urls: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "fomcminutes" not in href:
                continue
            if href.startswith("http"):
                url = href
            else:
                url = f"https://www.federalreserve.gov{href}"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            match = re.search(r"fomcminutes(\d{8})", url)
            date_str = match.group(1) if match else "unknown"
            results.append({"url": url, "date_str": date_str})

        return results

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"fomc_{digest}.pdf"

    def load_minutes_pdf(self, url: str) -> dict | None:
        """Download (or load cached) PDF and return cleaned text with metadata."""
        try:
            pdf_url = url.replace(".htm", ".pdf")
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

            metadata = self.metadata_extractor.extract(
                cleaned,
                source=url,
                doc_type="fomc_minutes",
            )
            return {"text": cleaned, "metadata": metadata.model_dump()}
        except Exception as exc:
            logger.error("Failed to load FOMC minutes from {}: {}", url, exc)
            return None

    def load_recent(self, n: int = 5) -> list[dict]:
        """Load the n most recent FOMC minutes documents."""
        pages = self.fetch_minutes_page()
        pages.sort(key=lambda x: x["date_str"], reverse=True)
        documents: list[dict] = []
        for entry in pages[:n]:
            loaded = self.load_minutes_pdf(entry["url"])
            if loaded is not None:
                documents.append(loaded)
        return documents
