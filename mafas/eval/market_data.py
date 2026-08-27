"""OHLCV loader for evaluation suites (yfinance, no FRED required)."""

from __future__ import annotations

import os

import pandas as pd
from loguru import logger


def live_market_eval_enabled() -> bool:
    """Unit tests set MAFAS_EVAL_OFFLINE=1 so pytest never hits the network."""
    return os.getenv("MAFAS_EVAL_OFFLINE", "").lower() not in {"1", "true", "yes"}


def load_ohlcv_frame(ticker: str, days: int = 504) -> pd.DataFrame | None:
    """Download daily OHLCV. Returns None on failure (eval must not crash)."""
    if not live_market_eval_enabled():
        return None
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance unavailable ({})", type(exc).__name__)
        return None
    try:
        df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OHLCV download failed for {} ({})", ticker, type(exc).__name__)
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        return None
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 80:
        return None
    return df
