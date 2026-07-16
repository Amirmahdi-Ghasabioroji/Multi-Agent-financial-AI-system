"""Pure Monte Carlo barrier-simulation engine for the Execution Agent.

Given an instrument's historical daily returns and a trade's entry / stop /
target levels, this bootstraps thousands of forward price paths and measures how
often the take-profit is hit before the stop-loss, the realised reward in R
multiples, and the max adverse excursion. It is deterministic for a fixed seed
and has no I/O, so it is fully unit-testable.

Modelling notes / simplifying assumptions:
    * Paths are close-to-close: returns are resampled (bootstrap) from history,
      so intrabar high/low are not modelled. When a single simulated bar would
      breach both barriers, the STOP is assumed to trigger first (conservative).
    * Slippage is applied to both the entry and exit fills, in basis points.
    * "R" is one unit of risk = |entry - stop| per unit of the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


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
    """Bootstrap forward paths and measure TP-before-SL barrier outcomes.

    Args:
        returns: 1-D array of historical simple daily returns.
        entry_price: reference entry price (barriers are defined around it).
        stop_price: stop-loss price level.
        target_price: take-profit price level.
        direction: "long" or "short".
        max_bars: max holding period in bars before a timeout exit at market.
        n_sims: number of bootstrap paths.
        slippage_bps: per-fill slippage in basis points (applied entry + exit).
        seed: RNG seed for reproducibility.
    """
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
    # Adverse slippage on the entry fill.
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


@dataclass
class PositionSize:
    """Risk-based position size for a single trade."""

    units: float
    notional: float
    notional_pct: float
    risk_amount: float
    risk_pct: float
    capped: bool


def compute_position_size(
    account_equity: float,
    entry_price: float,
    risk_per_unit: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
) -> PositionSize:
    """Size a trade from risk-per-trade, capped by a max notional weight.

    Units are chosen so that a stop-out loses `risk_per_trade_pct` of equity;
    if that notional exceeds `max_position_pct` of equity, the position is
    scaled down (and the realised risk falls below target).
    """
    if entry_price <= 0 or risk_per_unit <= 0 or account_equity <= 0:
        return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, False)

    target_risk = account_equity * max(0.0, risk_per_trade_pct)
    units = target_risk / risk_per_unit
    notional = units * entry_price
    cap = account_equity * max(0.0, max_position_pct)

    capped = False
    if cap > 0 and notional > cap:
        units = cap / entry_price
        notional = cap
        capped = True

    risk_amount = units * risk_per_unit
    return PositionSize(
        units=round(units, 4),
        notional=round(notional, 2),
        notional_pct=round(notional / account_equity, 4),
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_amount / account_equity, 4),
        capped=capped,
    )
