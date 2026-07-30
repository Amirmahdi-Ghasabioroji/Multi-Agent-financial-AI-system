"""Playbook-specific entry signal generators."""

from __future__ import annotations

import pandas as pd

from agents.simulation.signals.base import (
    WARMUP_BARS,
    SignalDirection,
    atr_series,
    log_spread,
    realised_vol_series,
    rolling_high,
    rolling_low,
    rsi,
    sma,
    spread_zscore,
)


def signal_trend_following(df: pd.DataFrame, bar: int) -> SignalDirection:
    if bar < WARMUP_BARS:
        return "flat"
    close = df["close"].astype(float)
    s20 = sma(close, 20)
    s50 = sma(close, 50)
    c, f, s = close.iloc[bar], s20.iloc[bar], s50.iloc[bar]
    if pd.isna(f) or pd.isna(s):
        return "flat"
    if c > s and f > s:
        return "long"
    if c < s and f < s:
        return "short"
    return "flat"


def signal_ma_crossover(df: pd.DataFrame, bar: int) -> SignalDirection:
    if bar < WARMUP_BARS:
        return "flat"
    close = df["close"].astype(float)
    s20 = sma(close, 20)
    s50 = sma(close, 50)
    prev_f, prev_s = s20.iloc[bar - 1], s50.iloc[bar - 1]
    cur_f, cur_s = s20.iloc[bar], s50.iloc[bar]
    if any(pd.isna(x) for x in (prev_f, prev_s, cur_f, cur_s)):
        return "flat"
    if prev_f <= prev_s and cur_f > cur_s:
        return "long"
    if prev_f >= prev_s and cur_f < cur_s:
        return "short"
    return "flat"


def signal_momentum_breakout(df: pd.DataFrame, bar: int) -> SignalDirection:
    if bar < 21:
        return "flat"
    close = df["close"].astype(float)
    prior_high = rolling_high(close.iloc[:bar], 20).iloc[-1]
    prior_low = rolling_low(close.iloc[:bar], 20).iloc[-1]
    c = close.iloc[bar]
    if pd.isna(prior_high) or pd.isna(prior_low):
        return "flat"
    if c > prior_high:
        return "long"
    if c < prior_low:
        return "short"
    return "flat"


def signal_mean_reversion(df: pd.DataFrame, bar: int) -> SignalDirection:
    if bar < 15:
        return "flat"
    r = rsi(df["close"], 14).iloc[bar]
    if pd.isna(r):
        return "flat"
    if r < 30:
        return "long"
    if r > 70:
        return "short"
    return "flat"


def signal_range_support_resistance(df: pd.DataFrame, bar: int) -> SignalDirection:
    if bar < 20:
        return "flat"
    row = df.iloc[bar]
    close = float(row["close"])
    open_ = float(row["open"])
    low20 = float(rolling_low(df["close"], 20).iloc[bar])
    high20 = float(rolling_high(df["close"], 20).iloc[bar])
    if low20 <= 0 or high20 <= 0:
        return "flat"
    near_low = abs(close - low20) / low20 <= 0.01
    near_high = abs(close - high20) / high20 <= 0.01
    if near_low and close > open_:
        return "long"
    if near_high and close < open_:
        return "short"
    return "flat"


def signal_volatility_based(df: pd.DataFrame, bar: int) -> SignalDirection:
    """Proxy: elevated ATR percentile + directional SMA filter."""
    if bar < WARMUP_BARS:
        return "flat"
    atr = atr_series(df, 14)
    window = atr.iloc[max(0, bar - 59) : bar + 1].dropna()
    if len(window) < 20:
        return "flat"
    threshold = window.quantile(0.75)
    cur_atr = atr.iloc[bar]
    s20 = sma(df["close"], 20).iloc[bar]
    c = float(df["close"].iloc[bar])
    if pd.isna(cur_atr) or pd.isna(s20) or pd.isna(threshold):
        return "flat"
    if cur_atr <= threshold:
        return "flat"
    if c > s20:
        return "long"
    if c < s20:
        return "short"
    return "flat"


def signal_carry(df: pd.DataFrame, bar: int) -> SignalDirection:
    """Proxy on equities: low realised vol + long-term trend."""
    if bar < 252:
        return "flat"
    close = df["close"].astype(float)
    vol20 = realised_vol_series(close, 20).iloc[bar]
    vol_hist = realised_vol_series(close, 20).iloc[max(0, bar - 251) : bar + 1].dropna()
    s200 = sma(close, 200).iloc[bar]
    c = close.iloc[bar]
    if pd.isna(vol20) or len(vol_hist) < 60 or pd.isna(s200):
        return "flat"
    threshold = vol_hist.quantile(0.30)
    if vol20 >= threshold:
        return "flat"
    if c > s200:
        return "long"
    if c < s200:
        return "short"
    return "flat"


def signal_pairs_relative_value(
    df: pd.DataFrame, bar: int, df_b: pd.DataFrame | None = None
) -> SignalDirection:
    """Spread z-score mean-reversion; requires aligned second leg in df_b."""
    if df_b is None or bar < WARMUP_BARS:
        return "flat"
    spread = log_spread(df["close"], df_b["close"])
    z = spread_zscore(spread, 60).iloc[bar]
    if pd.isna(z):
        return "flat"
    if z < -2:
        return "long"
    if z > 2:
        return "short"
    return "flat"
