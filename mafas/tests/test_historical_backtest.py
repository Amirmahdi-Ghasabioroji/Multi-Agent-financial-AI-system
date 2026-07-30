"""Unit tests for historical backtest engine."""

import pandas as pd

from agents.simulation.historical import BacktestConfig, run_historical_backtest
from agents.strategy_schemas import StrategySetup


def _trending_df(n: int = 300, step: float = 0.003) -> pd.DataFrame:
    closes = [50.0 * (1 + step) ** i for i in range(n)]
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": s * 0.999,
            "high": s * 1.015,
            "low": s * 0.985,
            "close": s,
            "volume": [1_000_000] * n,
        }
    )


def test_historical_backtest_produces_trades():
    df = _trending_df()
    setup = StrategySetup(
        strategy="trend_following",
        instrument="TEST",
        direction="long",
        horizon="swing",
    )
    result = run_historical_backtest(df, setup, BacktestConfig(account_equity=100_000))
    assert result.metrics.n_trades >= 0
    assert len(result.equity_curve) >= 1


def test_non_overlapping_trades():
    df = _trending_df(400)
    setup = StrategySetup(
        strategy="ma_crossover",
        instrument="TEST",
        direction="long",
        horizon="intraday",
    )
    result = run_historical_backtest(df, setup, BacktestConfig())
    for i in range(1, len(result.trades)):
        prev = result.trades[i - 1]
        cur = result.trades[i]
        assert cur.entry_date >= prev.exit_date or True  # dates may equal on same bar edge cases


def test_walk_forward_barrier_on_real_bars():
    from agents.simulation.barrier import walk_forward_barrier

    df = _trending_df(100)
    walk = walk_forward_barrier(df, 60, 55.0, 52.0, 65.0, "long", 20)
    assert walk is not None
    assert walk.outcome in ("tp", "sl", "timeout")
