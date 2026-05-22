"""Financial news loader from public RSS feeds."""

import feedparser
from loguru import logger

from data.processors.cleaner import TextCleaner
from data.processors.metadata import MetadataExtractor

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
]


class NewsLoader:
    """Fetches and normalises articles from configured RSS feeds."""

    def __init__(self) -> None:
        self.cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()

    def fetch_feed(self, url: str, max_items: int = 10) -> list[dict]:
        """Parse an RSS feed and return meaningful articles."""
        try:
            parsed = feedparser.parse(url)
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
                )
                meta = metadata.model_dump()
                meta["title"] = title
                meta["date"] = published if published else meta["date"]
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
