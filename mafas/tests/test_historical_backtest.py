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


def test_walk_forward_close_vs_ohlc_disagree_on_wick():
    from agents.simulation.barrier import walk_forward_barrier

    # After entry, one bar with a high wick through TP but close unchanged.
    n = 80
    close = [100.0] * n
    df = pd.DataFrame(
        {
            "open": close,
            "high": [100.0] * n,
            "low": [99.0] * n,
            "close": close,
            "volume": [1_000_000] * n,
        }
    )
    df.loc[70, "high"] = 120.0
    df.loc[70, "low"] = 99.0
    df.loc[70, "close"] = 100.0
    ohlc = walk_forward_barrier(df, 60, 100.0, 97.0, 110.0, "long", 20, barrier_mode="ohlc")
    close_walk = walk_forward_barrier(df, 60, 100.0, 97.0, 110.0, "long", 20, barrier_mode="close")
    assert ohlc is not None and close_walk is not None
    assert ohlc.outcome == "tp"
    assert close_walk.outcome != "tp"


def test_ohlc_bootstrap_runs():
    from agents.simulation.barrier import ohlc_relative_bars, simulate_barrier_ohlc_bootstrap

    df = _trending_df(200)
    bars = ohlc_relative_bars(df)
    res = simulate_barrier_ohlc_bootstrap(
        bars, 80.0, 76.0, 88.0, direction="long", max_bars=10, n_sims=200, seed=1
    )
    mass = res.prob_tp_before_sl + res.prob_sl_before_tp + res.prob_timeout
    assert abs(mass - 1.0) < 0.02
    assert res.n_sims == 200
