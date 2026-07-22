"""Monte Carlo barrier simulation and historical bar-by-bar trade walks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Max holding period (in daily bars) implied by a setup's horizon.
HORIZON_BARS: dict[str, int] = {"intraday": 5, "swing": 20, "position": 60}


@dataclass
class SimulationResult:
    """Aggregated statistics from a barrier simulation."""

    n_sims: int
    horizon_bars: int
    prob_tp_before_sl: float
    prob_sl_before_tp: float
    prob_timeout: float
    expected_r: float
    win_rate: float
    planned_rr: float
    avg_bars_to_exit: float
    mae_mean_r: float
    mae_p95_r: float
    seed: int


@dataclass
class BarrierWalkResult:
    """Outcome of walking forward on real bars after entry."""

    outcome: str  # tp | sl | timeout
    exit_price: float
    exit_bar_index: int
    bars_held: int
    pnl_r: float
    pnl_per_unit: float


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _barrier_hit_long(
    low: float, high: float, stop: float, target: float
) -> str | None:
    """Return which barrier is hit first on a long; stop wins if both touched."""
    hit_sl = low <= stop
    hit_tp = high >= target
    if hit_sl and hit_tp:
        return "sl"
    if hit_sl:
        return "sl"
    if hit_tp:
        return "tp"
    return None


def _barrier_hit_short(
    low: float, high: float, stop: float, target: float
) -> str | None:
    """Return which barrier is hit first on a short; stop wins if both touched."""
    hit_sl = high >= stop
    hit_tp = low <= target
    if hit_sl and hit_tp:
        return "sl"
    if hit_sl:
        return "sl"
    if hit_tp:
        return "tp"
    return None


def walk_forward_barrier(
    df: pd.DataFrame,
    entry_bar: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str,
    max_bars: int,
    slippage_bps: float = 5.0,
) -> BarrierWalkResult | None:
    """Walk real OHLCV bars forward from entry until TP, SL, or timeout."""
    is_long = direction.lower() != "short"
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0 or entry_bar < 0 or entry_bar >= len(df):
        return None

    slip = slippage_bps / 1e4
    entry_fill = entry_price * (1 + slip) if is_long else entry_price * (1 - slip)

    end_bar = min(len(df) - 1, entry_bar + max_bars)
    outcome = "timeout"
    exit_price = float(df["close"].astype(float).iloc[end_bar])
    exit_bar_index = end_bar

    for bar in range(entry_bar + 1, end_bar + 1):
        row = df.iloc[bar]
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])

        if is_long:
            hit = _barrier_hit_long(low, high, stop_price, target_price)
        else:
            hit = _barrier_hit_short(low, high, stop_price, target_price)

        if hit == "sl":
            outcome, exit_price, exit_bar_index = "sl", stop_price, bar
            break
        if hit == "tp":
            outcome, exit_price, exit_bar_index = "tp", target_price, bar
            break
        exit_price = close
        exit_bar_index = bar

    exit_fill = exit_price * (1 - slip) if is_long else exit_price * (1 + slip)
    pnl_per_unit = (exit_fill - entry_fill) if is_long else (entry_fill - exit_fill)
    pnl_r = pnl_per_unit / risk_per_unit

    return BarrierWalkResult(
        outcome=outcome,
        exit_price=round(exit_price, 4),
        exit_bar_index=exit_bar_index,
        bars_held=exit_bar_index - entry_bar,
        pnl_r=round(pnl_r, 4),
        pnl_per_unit=round(pnl_per_unit, 4),
    )


def simulate_barrier_bootstrap(
    returns: np.ndarray,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: str = "long",
    max_bars: int = 20,
    n_sims: int = 5000,
    slippage_bps: float = 5.0,
    seed: int = 42,
) -> SimulationResult:
    """Bootstrap forward paths and measure TP-before-SL barrier outcomes."""
    is_long = direction.lower() != "short"
    pool = _clean_returns(returns)
    risk_per_unit = abs(entry_price - stop_price)
    reward_per_unit = abs(target_price - entry_price)
    planned_rr = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

    if pool.size == 0 or risk_per_unit <= 0 or n_sims <= 0 or max_bars <= 0:
        return SimulationResult(
            n_sims=0, horizon_bars=max_bars, prob_tp_before_sl=0.0,
            prob_sl_before_tp=0.0, prob_timeout=1.0, expected_r=0.0, win_rate=0.0,
            planned_rr=round(planned_rr, 3), avg_bars_to_exit=0.0,
            mae_mean_r=0.0, mae_p95_r=0.0, seed=seed,
        )

    rng = np.random.default_rng(seed)
    slip = slippage_bps / 1e4
    entry_fill = entry_price * (1 + slip) if is_long else entry_price * (1 - slip)

    tp_count = sl_count = timeout_count = 0
    r_values = np.empty(n_sims, dtype=float)
    bars_to_exit = np.empty(n_sims, dtype=float)
    mae_values = np.empty(n_sims, dtype=float)

    samples = rng.choice(pool, size=(n_sims, max_bars), replace=True)

    for i in range(n_sims):
        price = entry_price
        worst_adverse = 0.0
        outcome = "timeout"
        exit_price = price
        exit_bar = max_bars

        for bar in range(max_bars):
            price *= 1.0 + samples[i, bar]
            adverse = (entry_fill - price) if is_long else (price - entry_fill)
            if adverse > worst_adverse:
                worst_adverse = adverse

            if is_long:
                if price <= stop_price:
                    outcome, exit_price, exit_bar = "sl", stop_price, bar + 1
                    break
                if price >= target_price:
                    outcome, exit_price, exit_bar = "tp", target_price, bar + 1
                    break
            else:
                if price >= stop_price:
                    outcome, exit_price, exit_bar = "sl", stop_price, bar + 1
                    break
                if price <= target_price:
                    outcome, exit_price, exit_bar = "tp", target_price, bar + 1
                    break
        else:
            exit_price = price

        exit_fill = exit_price * (1 - slip) if is_long else exit_price * (1 + slip)
        pnl_per_unit = (exit_fill - entry_fill) if is_long else (entry_fill - exit_fill)

        r_values[i] = pnl_per_unit / risk_per_unit
        bars_to_exit[i] = exit_bar
        mae_values[i] = worst_adverse / risk_per_unit
        if outcome == "tp":
            tp_count += 1
        elif outcome == "sl":
            sl_count += 1
        else:
            timeout_count += 1

    return SimulationResult(
        n_sims=n_sims,
        horizon_bars=max_bars,
        prob_tp_before_sl=round(tp_count / n_sims, 4),
        prob_sl_before_tp=round(sl_count / n_sims, 4),
        prob_timeout=round(timeout_count / n_sims, 4),
        expected_r=round(float(r_values.mean()), 4),
        win_rate=round(float((r_values > 0).mean()), 4),
        planned_rr=round(planned_rr, 3),
        avg_bars_to_exit=round(float(bars_to_exit.mean()), 2),
        mae_mean_r=round(float(mae_values.mean()), 4),
        mae_p95_r=round(float(np.percentile(mae_values, 95)), 4),
        seed=seed,
    )
