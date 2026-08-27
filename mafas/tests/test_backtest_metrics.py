"""Unit tests for backtest performance metrics."""

import numpy as np

from agents.simulation.metrics import (
    RawTrade,
    build_calendar_equity,
    build_equity_curve,
    compute_metrics,
    max_drawdown,
    mc_robustness_trade_r,
    profit_factor,
    sharpe_from_equity,
)


def test_max_drawdown():
    equity = np.array([100_000, 105_000, 102_000, 98_000, 103_000], dtype=float)
    amt, pct = max_drawdown(equity)
    assert amt == 7000.0
    assert pct > 0


def test_profit_factor():
    trades = [
        RawTrade(1.0, 1000, 5),
        RawTrade(-0.5, -500, 3),
        RawTrade(0.5, 500, 4),
    ]
    assert profit_factor(trades) == 3.0


def test_compute_metrics_basic():
    trades = [
        RawTrade(1.0, 1000, 5),
        RawTrade(-1.0, -1000, 4),
        RawTrade(2.0, 2000, 6),
        RawTrade(0.5, 500, 3),
        RawTrade(-0.5, -500, 2),
    ]
    equity = build_equity_curve(100_000, [t.pnl_amount for t in trades])
    m = compute_metrics(trades, equity, 100_000, min_trades=5)
    assert m["n_trades"] == 5
    assert m["total_pnl"] == 2000.0
    assert m["win_rate"] == 0.6
    assert m["low_sample"] is False


def test_sharpe_from_equity_positive():
    equity = np.linspace(100_000, 110_000, 100)
    sharpe = sharpe_from_equity(equity)
    assert sharpe is not None
    assert sharpe > 0


def test_compute_metrics_without_calendar_does_not_annualise():
    trades = [
        RawTrade(1.0, 1000, 5),
        RawTrade(-1.0, -1000, 4),
        RawTrade(2.0, 2000, 6),
        RawTrade(0.5, 500, 3),
        RawTrade(-0.5, -500, 2),
    ]
    equity = build_equity_curve(100_000, [t.pnl_amount for t in trades])
    m = compute_metrics(trades, equity, 100_000, min_trades=5)
    assert m["sharpe_ratio"] is None
    assert m["calmar_ratio"] is None


def test_calendar_sharpe_not_inflated_like_per_trade_annualisation():
    trades = [
        RawTrade(1.0, 1000, 5, entry_date="2024-01-10", exit_date="2024-01-15"),
        RawTrade(1.0, 1000, 5, entry_date="2024-06-10", exit_date="2024-06-14"),
    ]
    dates = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 23)][:252]
    calendar_equity = build_calendar_equity(100_000, dates, trades)
    calendar = compute_metrics(
        trades, calendar_equity, 100_000, min_trades=2, calendar_days=len(dates)
    )
    trade_equity = build_equity_curve(100_000, [1000.0, 1000.0])
    naive = sharpe_from_equity(trade_equity)
    assert calendar["sharpe_ratio"] is not None
    assert naive is not None
    assert calendar["sharpe_ratio"] < naive
    assert calendar["total_pnl"] == 2000.0


def test_mc_robustness_returns_bands():
    result = mc_robustness_trade_r([0.5, -0.3, 1.0, -0.2, 0.4], n_draws=200, seed=1)
    assert result is not None
    assert result["expected_r_p5"] <= result["expected_r_mean"] <= result["expected_r_p95"]
