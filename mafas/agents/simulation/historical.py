"""Historical playbook-driven backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.execution_schemas import BacktestMetrics, BacktestResult, TradeRecord
from agents.simulation.barrier import HORIZON_BARS, walk_forward_barrier
from agents.simulation.levels import compute_levels_at_bar
from agents.simulation.metrics import (
    RawTrade,
    build_calendar_equity,
    compute_metrics,
    downsample_series,
    drawdown_curve,
    mc_robustness_trade_r,
)
from agents.simulation.signals.base import align_pair_data, log_spread, spread_zscore
from agents.simulation.signals.registry import (
    matches_setup_direction,
    signal_at_bar,
    warmup_bars,
)
from agents.simulation.sizing import compute_position_size
from agents.strategy_schemas import StrategySetup


@dataclass
class BacktestConfig:
    account_equity: float = 100_000.0
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    slippage_bps: float = 5.0
    min_trades_for_metrics: int = 5
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.10
    mc_robustness_draws: int = 500
    seed: int = 42
    max_trades_in_json: int = 50


def _date_str(df: pd.DataFrame, bar: int) -> str:
    idx = df.index[bar]
    if hasattr(idx, "strftime"):
        return idx.strftime("%Y-%m-%d")
    return str(idx)


def _walk_pairs_barrier(
    spread_z_series: pd.Series,
    entry_bar: int,
    entry_z: float,
    direction: str,
    max_bars: int,
    sl_z: float = 1.0,
) -> tuple[str, float, int, float]:
    """Walk spread z-score toward mean reversion at z=0."""
    is_long = direction != "short"
    if is_long:
        stop_z = entry_z - sl_z
        target_z = 0.0
    else:
        stop_z = entry_z + sl_z
        target_z = 0.0
    risk = abs(entry_z - stop_z)
    if risk <= 0:
        return "timeout", entry_z, entry_bar, 0.0

    end = min(len(spread_z_series) - 1, entry_bar + max_bars)
    outcome = "timeout"
    exit_z = float(spread_z_series.iloc[end])
    exit_bar = end

    for bar in range(entry_bar + 1, end + 1):
        z = float(spread_z_series.iloc[bar])
        if is_long:
            if z <= stop_z:
                outcome, exit_z, exit_bar = "sl", stop_z, bar
                break
            if z >= target_z:
                outcome, exit_z, exit_bar = "tp", target_z, bar
                break
        else:
            if z >= stop_z:
                outcome, exit_z, exit_bar = "sl", stop_z, bar
                break
            if z <= target_z:
                outcome, exit_z, exit_bar = "tp", target_z, bar
                break
        exit_z = z
        exit_bar = bar

    pnl_r = (exit_z - entry_z) / risk if is_long else (entry_z - exit_z) / risk
    return outcome, exit_z, exit_bar, round(pnl_r, 4)


def run_historical_backtest(
    df: pd.DataFrame,
    setup: StrategySetup,
    config: BacktestConfig,
    df_b: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run non-overlapping historical backtest for one strategy setup."""
    max_bars = HORIZON_BARS.get(setup.horizon, 20)
    is_pairs = setup.strategy == "pairs_relative_value" and df_b is not None

    if is_pairs:
        df, df_b = align_pair_data(df, df_b)
        if df.empty:
            return _empty_result()

    start_warmup = warmup_bars(setup.strategy)
    raw_trades: list[RawTrade] = []
    records: list[TradeRecord] = []
    bar = start_warmup
    last_exit = -1

    spread_z_series: pd.Series | None = None
    if is_pairs and df_b is not None:
        spread = log_spread(df["close"], df_b["close"])
        spread_z_series = spread_zscore(spread, 60)

    while bar < len(df) - 1:
        if bar <= last_exit:
            bar += 1
            continue

        sig = signal_at_bar(setup.strategy, df, bar, df_b)
        if not matches_setup_direction(sig, setup.direction):
            bar += 1
            continue

        direction = setup.direction

        if is_pairs and spread_z_series is not None:
            entry_z = float(spread_z_series.iloc[bar])
            if pd.isna(entry_z):
                bar += 1
                continue
            outcome, exit_z, exit_bar, pnl_r = _walk_pairs_barrier(
                spread_z_series, bar, entry_z, direction, max_bars
            )
            entry_price = entry_z
            exit_price = exit_z
            risk_per_unit = 1.0
            pos = compute_position_size(
                config.account_equity,
                max(abs(entry_price), 0.01),
                risk_per_unit,
                config.risk_per_trade_pct,
                config.max_position_pct,
            )
            pnl_amount = round(pnl_r * pos.risk_amount, 2)
        else:
            levels = compute_levels_at_bar(
                df, bar, direction, config.sl_atr_mult, config.tp_atr_mult
            )
            if levels is None:
                bar += 1
                continue
            walk = walk_forward_barrier(
                df,
                bar,
                levels.entry,
                levels.stop_loss,
                levels.take_profit,
                direction,
                max_bars,
                config.slippage_bps,
            )
            if walk is None:
                bar += 1
                continue
            outcome = walk.outcome
            exit_bar = walk.exit_bar_index
            entry_price = levels.entry
            exit_price = walk.exit_price
            pnl_r = walk.pnl_r
            pos = compute_position_size(
                config.account_equity,
                levels.entry,
                levels.risk_per_unit,
                config.risk_per_trade_pct,
                config.max_position_pct,
            )
            pnl_amount = round(pnl_r * pos.risk_amount, 2)

        raw_trades.append(
            RawTrade(
                pnl_r=pnl_r,
                pnl_amount=pnl_amount,
                bars_held=exit_bar - bar,
                entry_date=_date_str(df, bar),
                exit_date=_date_str(df, exit_bar),
            )
        )
        records.append(
            TradeRecord(
                entry_date=_date_str(df, bar),
                exit_date=_date_str(df, exit_bar),
                direction=direction,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                outcome=outcome,
                pnl_r=pnl_r,
                pnl_amount=pnl_amount,
                bars_held=exit_bar - bar,
            )
        )
        last_exit = exit_bar
        bar = exit_bar + 1

    calendar_dates = (
        [_date_str(df, i) for i in range(start_warmup, len(df))]
        if len(df) > start_warmup
        else []
    )
    equity = build_calendar_equity(config.account_equity, calendar_dates, raw_trades)
    metrics_dict = compute_metrics(
        raw_trades,
        equity,
        config.account_equity,
        config.min_trades_for_metrics,
        calendar_days=len(calendar_dates) or None,
    )
    dd = drawdown_curve(equity)
    pnl_r_vals = [t.pnl_r for t in raw_trades]

    period_start = _date_str(df, start_warmup) if len(df) > start_warmup else ""
    period_end = _date_str(df, len(df) - 1)

    return BacktestResult(
        period_start=period_start,
        period_end=period_end,
        metrics=BacktestMetrics(**metrics_dict),
        trades=records[: config.max_trades_in_json],
        equity_curve=downsample_series(equity.tolist()),
        drawdown_curve=downsample_series(dd.tolist()),
        mc_robustness=mc_robustness_trade_r(
            pnl_r_vals, config.mc_robustness_draws, config.seed
        ),
    )


def _empty_result() -> BacktestResult:
    empty_metrics = BacktestMetrics(low_sample=True)
    return BacktestResult(
        period_start="",
        period_end="",
        metrics=empty_metrics,
        trades=[],
        equity_curve=[],
        drawdown_curve=[],
        mc_robustness=None,
    )
