"""Historical daily OHLCV loader backed by the Twelve Data REST API.

Used by the Execution Agent to fetch price history for trade simulation. The
free tier is rate-limited (8 requests/min, 800/day), so responses are cached on
disk and reused within the same day. Callers should fall back to another source
(e.g. MarketDataLoader / yfinance) if no API key is configured.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"


class TwelveDataError(RuntimeError):
    """Raised when Twelve Data is misconfigured or returns an error payload."""


class TwelveDataLoader:
    """Fetches cached daily OHLCV bars from Twelve Data."""

    def __init__(
        self,
        api_key: str = "",
        cache_dir: str = "./data/cache",
        cache_ttl_hours: float = 24.0,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours

    def is_configured(self) -> bool:
        """Return True if a non-placeholder API key is set."""
        return bool(self.api_key) and "your_" not in self.api_key.lower()

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("/", "_").upper()
        return self.cache_dir / f"td_{safe}_{interval}.json"

    def _cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        return age_hours < self.cache_ttl_hours

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _fetch(self, symbol: str, interval: str, outputsize: int) -> dict:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": str(outputsize),
            "apikey": self.api_key,
            "format": "JSON",
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.get(TWELVE_DATA_BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

    def get_daily(
        self,
        symbol: str,
        outputsize: int = 2000,
        interval: str = "1day",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return a date-ascending OHLCV DataFrame with lowercase columns.

        Raises TwelveDataError if not configured or the API returns an error.
        """
        if not self.is_configured():
            raise TwelveDataError("TWELVE_DATA_API_KEY is not configured")

        cache_path = self._cache_path(symbol, interval)
        payload: dict | None = None

        if not force_refresh and self._cache_fresh(cache_path):
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                logger.debug("Twelve Data cache hit for {}", symbol)
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload is None:
            logger.info("Fetching {} daily bars from Twelve Data for {}", outputsize, symbol)
            payload = self._fetch(symbol, interval, outputsize)
            if payload.get("status") == "error":
                raise TwelveDataError(
                    f"Twelve Data error for {symbol}: {payload.get('message', 'unknown')}"
                )
            try:
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not cache Twelve Data response ({})", type(exc).__name__)

        values = payload.get("values") or []
        if not values:
            raise TwelveDataError(f"Twelve Data returned no values for {symbol}")

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]
        df = df.dropna(subset=["close"])
        if df.empty:
            raise TwelveDataError(f"Twelve Data OHLCV for {symbol} empty after parsing")
        return df

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
