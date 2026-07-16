"""Financial news loader from public RSS feeds."""

import httpx
import feedparser
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from data.processors.cleaner import TextCleaner
from data.processors.metadata import MetadataExtractor

# Multiple independent finance feeds provide redundancy: a single feed timing
# out or going offline (Reuters discontinued feeds.reuters.com) no longer
# starves the corpus of news. Each feed is fetched in isolation.
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://www.marketwatch.com/rss/topstories",
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",
]

USER_AGENT = "Mozilla/5.0 (compatible; MAFAS/1.0; +https://github.com/mafas-project)"


class NewsLoader:
    """Fetches and normalises articles from configured RSS feeds."""

    def __init__(self) -> None:
        self.cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    def _fetch_rss_xml(self, url: str) -> str:
        """Download raw RSS XML via httpx (feedparser's urllib fails on some networks)."""
        with httpx.Client(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    def fetch_feed(self, url: str, max_items: int = 10) -> list[dict]:
        """Parse an RSS feed and return meaningful articles."""
        try:
            xml = self._fetch_rss_xml(url)
            parsed = feedparser.parse(xml)
            if getattr(parsed, "bozo", False) and not parsed.entries:
                logger.error("RSS parse error for {}: {}", url, parsed.bozo_exception)
                return []

            articles: list[dict] = []
            for entry in parsed.entries[:max_items]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                link = getattr(entry, "link", url)
                published = getattr(entry, "published", "unknown")

                body = self.cleaner.clean_html(summary)
                full_text = f"{title}\n\n{body}".strip()

                if not self.cleaner.is_meaningful(full_text, min_words=30):
                    continue

                metadata = self.metadata_extractor.extract(
                    full_text,
                    source=link,
                    doc_type="news_article",
                    date=published,
                )
                meta = metadata.model_dump()
                meta["title"] = title
                articles.append({"text": full_text, "metadata": meta})

            return articles
        except Exception as exc:
            logger.error("fetch_feed failed for {}: {}", url, exc)
            return []

    def load_all(self, max_per_feed: int = 10) -> list[dict]:
        """Fetch all configured feeds and deduplicate by article URL."""
        seen_links: set[str] = set()
        combined: list[dict] = []
        for feed_url in RSS_FEEDS:
            for article in self.fetch_feed(feed_url, max_items=max_per_feed):
                link = article["metadata"].get("source", "")
                if link in seen_links:
                    continue
                seen_links.add(link)
                combined.append(article)
        return combined
