"""Deterministic quantitative risk metrics for the Risk Agent.

Every function here is pure and side-effect free: given the same price data it
returns the same numbers. The Risk Agent treats these outputs as the source of
truth; the LLM only narrates them. Keeping the maths isolated makes the agent
fully testable and reproducible offline.

Conventions:
    * Returns are simple daily percentage changes unless noted.
    * Volatility is annualised with 252 trading days.
    * All ratios/percentages are expressed as fractions (0.20 == 20%).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Annualised realised-volatility thresholds used to bucket a single asset.
_ASSET_VOL_LOW = 0.20
_ASSET_VOL_HIGH = 0.35

# VIX thresholds used to bucket the overall market regime.
_VIX_LOW = 15.0
_VIX_HIGH = 25.0

# Absolute daily-return correlation above which a pair is flagged.
_CORR_WARN = 0.80

_REGIME_SIZE_SCALAR = {"low": 1.0, "medium": 0.7, "high": 0.4}


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns from a close-price series, NaNs dropped."""
    return close.astype(float).pct_change().dropna()


def realised_vol(close: pd.Series, annualise: bool = True) -> float:
    """Annualised realised volatility (std of daily returns).

    Returns 0.0 when there are too few observations to be meaningful.
    """
    rets = daily_returns(close)
    if len(rets) < 2:
        return 0.0
    vol = float(rets.std(ddof=1))
    if annualise:
        vol *= math.sqrt(TRADING_DAYS)
    return vol


def average_true_range(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's Average True Range over the last `period` bars.

    Expects lowercase 'high', 'low', 'close' columns (as produced by
    MarketDataLoader.get_ohlcv). Returns 0.0 if data is insufficient.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns) or len(df) < 2:
        return 0.0

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.dropna()
    if true_range.empty:
        return 0.0

    window = min(period, len(true_range))
    return float(true_range.tail(window).mean())


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """ATR expressed as a fraction of the latest close price."""
    if "close" not in df.columns or df.empty:
        return 0.0
    last_close = float(df["close"].astype(float).iloc[-1])
    if last_close <= 0:
        return 0.0
    return average_true_range(df, period) / last_close


def classify_asset_regime(annualised_vol: float) -> str:
    """Bucket a single asset's realised vol into low/medium/high."""
    if annualised_vol < _ASSET_VOL_LOW:
        return "low"
    if annualised_vol >= _ASSET_VOL_HIGH:
        return "high"
    return "medium"


def classify_market_regime(vix_level: float, mean_asset_vol: float) -> str:
    """Blend the VIX and the basket's mean realised vol into one regime.

    VIX is the primary signal (forward-looking, market-wide); the basket's own
    realised vol acts as a confirming/escalating input.
    """
    if vix_level < _VIX_LOW:
        vix_regime = "low"
    elif vix_level >= _VIX_HIGH:
        vix_regime = "high"
    else:
        vix_regime = "medium"

    asset_regime = classify_asset_regime(mean_asset_vol)

    order = {"low": 0, "medium": 1, "high": 2}
    inverse = {0: "low", 1: "medium", 2: "high"}
    # Take the more cautious (higher) of the two signals.
    return inverse[max(order[vix_regime], order[asset_regime])]


def correlation_matrix(returns_by_ticker: dict[str, pd.Series]) -> pd.DataFrame:
    """Pairwise correlation of daily returns across aligned dates.

    Tickers with no data are dropped. Returns an empty frame if fewer than two
    usable series remain.
    """
    usable = {t: r for t, r in returns_by_ticker.items() if r is not None and len(r) > 2}
    if len(usable) < 2:
        return pd.DataFrame()
    frame = pd.DataFrame(usable).dropna(how="any")
    if len(frame) < 3:
        return pd.DataFrame()
    return frame.corr()


def high_correlation_pairs(
    corr: pd.DataFrame, threshold: float = _CORR_WARN
) -> list[tuple[str, str, float]]:
    """Return (a, b, corr) for each upper-triangle pair above `threshold`."""
    pairs: list[tuple[str, str, float]] = []
    if corr.empty:
        return pairs
    tickers = list(corr.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            value = float(corr.iloc[i, j])
            if abs(value) >= threshold:
                pairs.append((tickers[i], tickers[j], value))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs


def mean_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Average of the off-diagonal correlation entries."""
    if corr.empty or len(corr) < 2:
        return 0.0
    n = len(corr)
    mask = ~np.eye(n, dtype=bool)
    return float(corr.values[mask].mean())


def effective_number_of_bets(corr: pd.DataFrame) -> float:
    """Diversification proxy: N / (1 + (N-1) * mean_correlation).

    Equals N when assets are uncorrelated and collapses toward 1 as the basket
    becomes perfectly correlated (i.e. one concentrated bet).
    """
    n = len(corr)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    mean_corr = max(0.0, mean_pairwise_correlation(corr))
    denom = 1.0 + (n - 1) * mean_corr
    if denom <= 0:
        return float(n)
    return float(n / denom)


def suggested_position_sizing(
    asset_vols: dict[str, float],
    regime: str,
    mean_corr: float,
    target_portfolio_vol: float = 0.10,
    max_position: float = 0.25,
    base_risk_per_trade: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Volatility-targeted position-size caps per asset.

    Inverse-vol weighting (size ~ target_vol / asset_vol) is scaled down in
    higher vol regimes and when the basket is highly correlated (a correlated
    basket behaves like a single larger position). Values are fractions of the
    notional portfolio and are advisory constraints, not signals.
    """
    regime_scalar = _REGIME_SIZE_SCALAR.get(regime, 0.7)
    # High average correlation shrinks independent capacity; floor at 0.3.
    corr_scalar = max(0.3, 1.0 - max(0.0, mean_corr))

    result: dict[str, dict[str, float]] = {}
    for ticker, vol in asset_vols.items():
        safe_vol = max(vol, 1e-6)
        inverse_vol_weight = min(max_position, target_portfolio_vol / safe_vol)
        max_pos = inverse_vol_weight * regime_scalar * corr_scalar
        result[ticker] = {
            "max_position_pct": round(max(0.0, max_pos), 4),
            "risk_per_trade_pct": round(base_risk_per_trade * regime_scalar, 4),
        }
    return result
