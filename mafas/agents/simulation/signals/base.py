"""Shared technical helpers and signal types for playbook entry rules."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from agents.risk_metrics import average_true_range

SignalDirection = Literal["long", "short", "flat"]

WARMUP_BARS = 60


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rolling_high(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(period, min_periods=period).max()


def rolling_low(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(period, min_periods=period).min()


def realised_vol_series(close: pd.Series, window: int) -> pd.Series:
    rets = close.astype(float).pct_change()
    return rets.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Bar-by-bar ATR approximation using expanding window mean TR."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def atr_at_bar(df: pd.DataFrame, bar_index: int, period: int = 14) -> float:
    return average_true_range(df.iloc[: bar_index + 1], period=period)


def spread_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
  mean = spread.rolling(window, min_periods=window).mean()
  std = spread.rolling(window, min_periods=window).std(ddof=1)
  return (spread - mean) / std.replace(0, np.nan)


def log_spread(close_a: pd.Series, close_b: pd.Series) -> pd.Series:
    return np.log(close_a.astype(float)) - np.log(close_b.astype(float))


def align_pair_data(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join two OHLCV frames on index."""
    combined = df_a[["open", "high", "low", "close"]].join(
        df_b[["open", "high", "low", "close"]],
        how="inner",
        lsuffix="_a",
        rsuffix="_b",
    ).dropna()
    if combined.empty:
        return df_a.iloc[:0], df_b.iloc[:0]
    idx = combined.index
    return df_a.loc[idx].copy(), df_b.loc[idx].copy()
