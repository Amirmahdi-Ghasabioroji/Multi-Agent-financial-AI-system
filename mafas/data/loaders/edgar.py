"""SEC EDGAR filings loader."""

from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from data.processors.cleaner import TextCleaner
from data.processors.metadata import MetadataExtractor

SEC_HEADERS = {"User-Agent": "MAFAS project student@example.com"}
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class EDGARLoader:
    """Fetches and parses SEC EDGAR filings for a given ticker."""

    def __init__(self, cache_dir: str = "./data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()
        self._cik_cache: dict[str, str] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _get(self, url: str, timeout: float = 30.0) -> httpx.Response:
        with httpx.Client(timeout=timeout, headers=SEC_HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            return response

    def _resolve_cik(self, ticker: str) -> str | None:
        """Resolve ticker symbol to zero-padded 10-digit CIK."""
        upper = ticker.upper()
        if upper in self._cik_cache:
            return self._cik_cache[upper]

        try:
            response = self._get(COMPANY_TICKERS_URL)
            data = response.json()
            for entry in data.values():
                if str(entry.get("ticker", "")).upper() == upper:
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[upper] = cik
                    return cik
        except Exception as exc:
            logger.error("CIK lookup failed for {}: {}", ticker, exc)

        try:
            atom_url = (
                "https://www.sec.gov/cgi-bin/browse-edgar"
                f"?company=&CIK={upper}&type=10-K&dateb=&owner=include"
                f"&count=5&search_text=&action=getcompany&output=atom"
            )
            response = self._get(atom_url)
            soup = BeautifulSoup(response.text, "xml")
            cik_el = soup.find("cik") or soup.find("name", attrs={"cik": True})
            if cik_el:
                cik_text = cik_el.get_text(strip=True) or cik_el.get("cik", "")
                if cik_text:
                    cik = str(int(cik_text)).zfill(10)
                    self._cik_cache[upper] = cik
                    return cik
        except Exception as exc:
            logger.error("Atom CIK lookup failed for {}: {}", ticker, exc)

        return None

    def get_recent_filings(
        self,
        ticker: str,
        form_type: str = "10-Q",
        limit: int = 3,
    ) -> list[dict]:
        """Search EDGAR for recent filings of the given form type."""
        try:
            end = datetime.utcnow().date()
            start = end - timedelta(days=365 * 3)
            params = {
                "q": f'"{ticker.upper()}"',
                "dateRange": "custom",
                "startdt": start.isoformat(),
                "enddt": end.isoformat(),
                "forms": form_type,
            }
            with httpx.Client(timeout=30.0, headers=SEC_HEADERS) as client:
                response = client.get(EFTS_SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()

            filings: list[dict] = []
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[: limit * 3]:
                source = hit.get("_source", {})
                accession = source.get("adsh") or source.get("accession_no", "")
                if not accession:
                    continue
                if "-" not in accession and len(accession) == 18:
                    accession = (
                        f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
                    )
                filings.append(
                    {
                        "accession_number": accession,
                        "date": source.get("file_date", source.get("period_ending", "")),
                        "form_type": source.get("form", form_type),
                        "ticker": ticker.upper(),
                    }
                )
                if len(filings) >= limit:
                    break

            if filings:
                return filings

            return self._filings_from_submissions(ticker, form_type, limit)
        except Exception as exc:
            logger.error("get_recent_filings search failed for {}: {}", ticker, exc)
            return self._filings_from_submissions(ticker, form_type, limit)

    def _filings_from_submissions(
        self,
        ticker: str,
        form_type: str,
        limit: int,
    ) -> list[dict]:
        """Fallback: read recent filings from SEC submissions JSON."""
        cik = self._resolve_cik(ticker)
        if not cik:
            return []
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self._get(url)
            data = response.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            results: list[dict] = []
            for form, accession, date in zip(forms, accessions, dates):
                if form != form_type:
                    continue
                results.append(
                    {
                        "accession_number": accession,
                        "date": date,
                        "form_type": form,
                        "ticker": ticker.upper(),
                    }
                )
                if len(results) >= limit:
                    break
            return results
        except Exception as exc:
            logger.error("Submissions fallback failed for {}: {}", ticker, exc)
            return []

    def _filing_index_url(self, cik: str, accession_number: str) -> str:
        cik_num = str(int(cik))
        accession_nodash = accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik_num}/"
            f"{accession_nodash}/index.json"
        )

    def _select_primary_document(
        self,
        items: list[dict],
        ticker: str,
        base_url: str,
    ) -> str | None:
        """Pick the main 10-Q/10-K HTML (largest .htm), not exhibits or index pages."""
        ticker_lower = ticker.lower()
        candidates: list[tuple[int, str]] = []

        for item in items:
            name = item.get("name", "")
            lower = name.lower()
            if not lower.endswith((".htm", ".html")):
                continue
            if any(
                skip in lower
                for skip in ("index", "exhibit", "ex-", "xsl", "xml", "xsd", "cal", "def", "lab", "pre")
            ):
                continue
            if lower.startswith("r") and lower.endswith(".htm"):
                continue

            size = 0
            try:
                size = int(str(item.get("size", "0")).replace(",", ""))
            except ValueError:
                size = 0

            # Prefer ticker-prefixed main filing (e.g. aapl-20251227.htm)
            if ticker_lower in lower and "exhibit" not in lower:
                size += 1_000_000

            doc_type = str(item.get("type", "")).upper()
            if doc_type in ("10-Q", "10-K", "10-K/A", "10-Q/A"):
                size += 500_000

            candidates.append((size, urljoin(base_url, name)))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Fallback: full submission text bundle
        for item in items:
            name = item.get("name", "")
            if name.endswith(".txt") and "index" not in name.lower():
                return urljoin(base_url, name)

        return None

    def load_filing_text(self, ticker: str, accession_number: str) -> dict | None:
        """Download primary filing document and return cleaned text."""
        try:
            cik = self._resolve_cik(ticker)
            if not cik:
                logger.error("Could not resolve CIK for {}", ticker)
                return None

            index_url = self._filing_index_url(cik, accession_number)
            response = self._get(index_url)
            index_data = response.json()

            base_url = index_url.replace("index.json", "")
            items = index_data.get("directory", {}).get("item", [])
            doc_url = self._select_primary_document(items, ticker, base_url)

            if not doc_url:
                logger.error("No primary document found for {}", accession_number)
                return None

            doc_response = self._get(doc_url)
            if doc_url.lower().endswith(".txt"):
                cleaned = self.cleaner.clean(doc_response.text)
            else:
                cleaned = self.cleaner.clean_html(doc_response.text)

            if not self.cleaner.is_meaningful(cleaned, min_words=500):
                logger.warning("Filing too short, skipping: {}", accession_number)
                return None

            metadata = self.metadata_extractor.extract(
                cleaned,
                source=doc_url,
                doc_type="sec_filing",
                tickers=[ticker.upper()],
            )
            meta = metadata.model_dump()
            meta["accession_number"] = accession_number
            return {"text": cleaned, "metadata": meta}
        except Exception as exc:
            logger.error(
                "load_filing_text failed for {} {}: {}",
                ticker,
                accession_number,
                exc,
            )
            return None

    def load_recent_for_ticker(self, ticker: str, limit: int = 2) -> list[dict]:
        """Load recent 10-Q filings for a ticker."""
        filings = self.get_recent_filings(ticker, form_type="10-Q", limit=limit)
        documents: list[dict] = []
        for filing in filings:
            loaded = self.load_filing_text(
                ticker,
                filing["accession_number"],
            )
            if loaded is not None:
                documents.append(loaded)
        return documents
