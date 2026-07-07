"""Market and macroeconomic data loader (non-RAG)."""

from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred
from loguru import logger


class MarketDataLoader:
    """Provides OHLCV, FRED series, and VIX data for trading agents."""

    def __init__(self, fred_api_key: str, cache_dir: str = "./data/cache") -> None:
        self.fred_api_key = fred_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fred = Fred(api_key=fred_api_key)

    def get_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Download OHLCV history for a ticker."""
        df = yf.download(ticker, period=f"{days}d", progress=False)
        if df.empty:
            raise ValueError(f"No OHLCV data returned for ticker '{ticker}'")
        # yfinance returns MultiIndex columns (field, ticker); flatten to the
        # field level BEFORE lowercasing so 'Close' -> 'close' resolves cleanly.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.dropna()
        if df.empty:
            raise ValueError(f"OHLCV data for '{ticker}' is empty after dropping NaNs")
        return df

    def get_fred_series(self, series_id: str, days: int = 365) -> pd.Series:
        """Fetch a FRED series for the last N days."""
        try:
            series = self.fred.get_series(series_id)
            if series is None or series.empty:
                logger.warning("Empty FRED series: {}", series_id)
                return pd.Series(dtype=float)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            filtered = series[series.index >= cutoff]
            return filtered
        except Exception as exc:
            # Log only the exception type — FRED errors can include the API key
            # in the request URL embedded in the exception message.
            logger.warning(
                "FRED fetch failed for {} ({})", series_id, type(exc).__name__
            )
            return pd.Series(dtype=float)

    def get_vix(self) -> float:
        """Return current VIX level with a safe fallback."""
        try:
            price = yf.Ticker("^VIX").fast_info["last_price"]
            return float(price)
        except Exception as exc:
            logger.warning("VIX fetch failed, using fallback 20.0 ({})", type(exc).__name__)
            return 20.0
