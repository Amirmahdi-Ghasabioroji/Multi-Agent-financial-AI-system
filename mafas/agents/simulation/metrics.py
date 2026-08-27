"""Performance metrics from trade lists and equity curves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agents.risk_metrics import TRADING_DAYS


@dataclass
class RawTrade:
    pnl_r: float
    pnl_amount: float
    bars_held: int
    entry_date: str | None = None
    exit_date: str | None = None


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """Return (max_drawdown_amount, max_drawdown_pct) as positive numbers."""
    if equity.size == 0:
        return 0.0, 0.0
    peak = equity[0]
    max_dd_amt = 0.0
    max_dd_pct = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd > max_dd_amt:
            max_dd_amt = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd_amt, max_dd_pct


def drawdown_curve(equity: np.ndarray) -> np.ndarray:
    if equity.size == 0:
        return np.array([], dtype=float)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return dd


def downsample_series(values: list[float], max_points: int = 120) -> list[float]:
    if len(values) <= max_points:
        return [round(v, 2) for v in values]
    indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return [round(float(values[i]), 2) for i in indices]


def sharpe_from_equity(equity: np.ndarray, periods_per_year: float = TRADING_DAYS) -> float | None:
    if equity.size < 3:
        return None
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 2 or float(rets.std(ddof=1)) == 0.0:
        return None
    return float(rets.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year))


def sortino_from_equity(equity: np.ndarray, periods_per_year: float = TRADING_DAYS) -> float | None:
    if equity.size < 3:
        return None
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    downside = rets[rets < 0]
    if downside.size < 1:
        return None
    down_std = float(downside.std(ddof=1))
    if down_std == 0.0:
        return None
    return float(rets.mean() / down_std * math.sqrt(periods_per_year))


def profit_factor(trades: list[RawTrade]) -> float:
    gross_win = sum(t.pnl_amount for t in trades if t.pnl_amount > 0)
    gross_loss = abs(sum(t.pnl_amount for t in trades if t.pnl_amount < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def mc_robustness_trade_r(
    pnl_r_values: list[float],
    n_draws: int = 500,
    seed: int = 42,
) -> dict[str, float] | None:
    if len(pnl_r_values) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(pnl_r_values, dtype=float)
    n = len(arr)
    totals = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        sample = rng.choice(arr, size=n, replace=True)
        totals[i] = sample.sum()
    return {
        "expected_r_mean": round(float(totals.mean()), 4),
        "expected_r_p5": round(float(np.percentile(totals, 5)), 4),
        "expected_r_p95": round(float(np.percentile(totals, 95)), 4),
        "total_r_mean": round(float(totals.mean()), 4),
        "total_r_p5": round(float(np.percentile(totals, 5)), 4),
        "total_r_p95": round(float(np.percentile(totals, 95)), 4),
    }


def build_equity_curve(initial_equity: float, trade_pnls: list[float]) -> np.ndarray:
    """Per-trade equity (not calendar). Prefer ``build_calendar_equity`` for ratios."""
    equity = [initial_equity]
    for pnl in trade_pnls:
        equity.append(equity[-1] + pnl)
    return np.array(equity, dtype=float)


def build_calendar_equity(
    initial_equity: float,
    calendar_dates: list[str],
    trades: list[RawTrade],
) -> np.ndarray:
    """Mark cash equity once per calendar session; PnL hits on the exit date.

    The returned series has length ``len(calendar_dates) + 1`` (initial plus
    end-of-day marks). Sharpe/Sortino/Calmar must be computed on this series
    with ``periods_per_year=TRADING_DAYS``.
    """
    if not calendar_dates:
        return np.array([initial_equity], dtype=float)
    pnl_on: dict[str, float] = {}
    for trade in trades:
        key = trade.exit_date or ""
        if not key:
            continue
        pnl_on[key] = pnl_on.get(key, 0.0) + trade.pnl_amount
    date_set = set(calendar_dates)
    unmatched = [d for d in pnl_on if d not in date_set]
    if unmatched:
        last = calendar_dates[-1]
        for day in unmatched:
            pnl_on[last] = pnl_on.get(last, 0.0) + pnl_on.pop(day)
    equity = [initial_equity]
    running = initial_equity
    for day in calendar_dates:
        running += pnl_on.get(day, 0.0)
        equity.append(running)
    return np.array(equity, dtype=float)


def compute_metrics(
    trades: list[RawTrade],
    equity: np.ndarray,
    initial_equity: float,
    min_trades: int = 5,
    *,
    calendar_days: int | None = None,
) -> dict:
    n = len(trades)
    low_sample = n < min_trades
    empty = {
        "n_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "expectancy_r": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_amount": 0.0,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "calmar_ratio": None,
        "avg_win_r": 0.0,
        "avg_loss_r": 0.0,
        "avg_bars_held": 0.0,
        "low_sample": True,
    }
    if n == 0:
        return empty

    wins = [t for t in trades if t.pnl_r > 0]
    losses = [t for t in trades if t.pnl_r <= 0]
    total_pnl = sum(t.pnl_amount for t in trades)
    total_return_pct = total_pnl / initial_equity if initial_equity > 0 else 0.0
    max_dd_amt, max_dd_pct = max_drawdown(equity)
    pf = profit_factor(trades)

    # Annualised ratios are only defined on a calendar (session) equity curve.
    # Passing a per-trade curve without calendar_days used to treat each trade
    # as one trading day and inflate Sharpe/Calmar by ~sqrt(252 / n_trades).
    sharpe = None
    sortino = None
    calmar = None
    if calendar_days is not None and calendar_days > 0:
        sharpe = sharpe_from_equity(equity, TRADING_DAYS)
        sortino = sortino_from_equity(equity, TRADING_DAYS)
        years = calendar_days / TRADING_DAYS
        ann_return = total_return_pct / years if years > 0 else 0.0
        calmar = (ann_return / max_dd_pct) if max_dd_pct > 0 else None

    return {
        "n_trades": n,
        "win_rate": round(len(wins) / n, 4),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else 999.0,
        "expectancy_r": round(float(np.mean([t.pnl_r for t in trades])), 4),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "max_drawdown_amount": round(max_dd_amt, 2),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "calmar_ratio": round(calmar, 4) if calmar is not None else None,
        "avg_win_r": round(float(np.mean([t.pnl_r for t in wins])), 4) if wins else 0.0,
        "avg_loss_r": round(float(np.mean([t.pnl_r for t in losses])), 4) if losses else 0.0,
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 2),
        "low_sample": low_sample,
    }
