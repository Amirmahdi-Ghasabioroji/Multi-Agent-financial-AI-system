"""Unit tests for the Execution Agent (synthetic data, mocked LLM, no network)."""

import numpy as np
import pandas as pd
import pytest

from agents.backtest import (
    HORIZON_BARS,
    compute_position_size,
    simulate_barrier_bootstrap,
)
from agents.execution import ExecutionAgent
from agents.execution_schemas import TradeCard
from agents.risk_schemas import PositionSizingConstraint, RiskSummary
from agents.strategy_schemas import StrategySetup


def _ohlcv(prices: list[float]) -> pd.DataFrame:
    close = pd.Series(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [1_000_000] * len(close),
        }
    )


class FakeMarket:
    def __init__(self, frames):
        self._frames = frames

    def get_ohlcv(self, ticker, days=252):
        if ticker not in self._frames:
            raise ValueError("no data")
        return self._frames[ticker]


class FakeLLM:
    model = "mistral"

    def __init__(self, response=None, available=True, raise_error=False):
        self._response = response or {}
        self._available = available
        self._raise = raise_error

    def is_available(self):
        return self._available

    def chat_json(self, messages, options=None):
        if self._raise:
            from agents.llm import OllamaError

            raise OllamaError("outage")
        return self._response


# --------------------------------------------------------------------------- #
# Barrier simulation
# --------------------------------------------------------------------------- #
class TestSimulation:
    def test_strong_uptrend_favours_tp_for_long(self):
        rets = np.full(300, 0.02)  # relentless up
        res = simulate_barrier_bootstrap(
            rets, entry_price=100, stop_price=97, target_price=106,
            direction="long", max_bars=20, n_sims=500, seed=1,
        )
        assert res.prob_tp_before_sl > 0.9
        assert res.expected_r > 0

    def test_downtrend_hits_sl_for_long(self):
        rets = np.full(300, -0.02)
        res = simulate_barrier_bootstrap(
            rets, entry_price=100, stop_price=97, target_price=106,
            direction="long", max_bars=20, n_sims=500, seed=1,
        )
        assert res.prob_sl_before_tp > 0.9
        assert res.expected_r < 0

    def test_short_direction_inverts(self):
        rets = np.full(300, -0.02)  # down helps a short
        res = simulate_barrier_bootstrap(
            rets, entry_price=100, stop_price=103, target_price=94,
            direction="short", max_bars=20, n_sims=500, seed=1,
        )
        assert res.prob_tp_before_sl > 0.9

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0, 0.01, 300)
        a = simulate_barrier_bootstrap(rets, 100, 97, 106, "long", 20, 400, seed=7)
        b = simulate_barrier_bootstrap(rets, 100, 97, 106, "long", 20, 400, seed=7)
        assert a.prob_tp_before_sl == b.prob_tp_before_sl
        assert a.expected_r == b.expected_r

    def test_probabilities_sum_to_one(self):
        rng = np.random.default_rng(3)
        rets = rng.normal(0.0, 0.015, 300)
        res = simulate_barrier_bootstrap(rets, 100, 96, 108, "long", 15, 1000, seed=5)
        total = res.prob_tp_before_sl + res.prob_sl_before_tp + res.prob_timeout
        assert abs(total - 1.0) < 1e-9

    def test_zero_risk_returns_empty(self):
        res = simulate_barrier_bootstrap(np.array([0.01, -0.01]), 100, 100, 106, "long")
        assert res.n_sims == 0


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #
class TestSizing:
    def test_risk_based_units(self):
        pos = compute_position_size(
            account_equity=100_000, entry_price=100, risk_per_unit=5,
            risk_per_trade_pct=0.01, max_position_pct=0.5,
        )
        # risk budget 1000 / 5 per unit = 200 units, notional 20k (< 50k cap)
        assert pos.units == 200
        assert pos.notional == 20_000
        assert pos.capped is False
        assert abs(pos.risk_pct - 0.01) < 1e-9

    def test_cap_applies(self):
        pos = compute_position_size(
            account_equity=100_000, entry_price=100, risk_per_unit=1,
            risk_per_trade_pct=0.02, max_position_pct=0.10,
        )
        # unconstrained notional would be 2000 units * 100 = 200k >> 10k cap
        assert pos.capped is True
        assert pos.notional == 10_000
        assert pos.risk_pct < 0.02  # realised risk below target after capping

    def test_degenerate_inputs(self):
        pos = compute_position_size(100_000, 0, 5, 0.01, 0.1)
        assert pos.units == 0


# --------------------------------------------------------------------------- #
# ExecutionAgent orchestration
# --------------------------------------------------------------------------- #
def _uptrend(n=300, start=50.0, step=0.004):
    return [start * (1 + step) ** i for i in range(n)]


@pytest.fixture
def agent_market():
    frames = {"NVDA": _ohlcv(_uptrend())}
    return FakeMarket(frames)


def _risk_with(ticker="NVDA"):
    return RiskSummary(
        universe=[ticker],
        position_sizing=[
            PositionSizingConstraint(ticker=ticker, max_position_pct=0.10, risk_per_trade_pct=0.01)
        ],
    )


def test_simulate_produces_card(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=FakeLLM(available=False))
    setup = StrategySetup(strategy="trend_following", strategy_name="Trend Following",
                          instrument="NVDA", direction="long", horizon="swing",
                          confidence=0.7, playbook_fit=0.8)
    card = agent.simulate(setup, risk=_risk_with())
    assert isinstance(card, TradeCard)
    assert card.simulated is True
    assert card.backtest is not None
    assert card.data_source == "yfinance"
    assert card.levels.entry > 0
    assert card.levels.take_profit > card.levels.entry  # long
    assert card.stats.n_sims > 0
    assert card.sizing.units > 0
    assert card.verdict  # fallback verdict present


def test_horizon_sets_sim_bars(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=None, n_sims=300)
    setup = StrategySetup(strategy="momentum_breakout", instrument="NVDA",
                          direction="long", horizon="position")
    card = agent.simulate(setup, risk=_risk_with())
    assert card.stats.horizon_bars == HORIZON_BARS["position"]


def test_neutral_setup_not_simulated(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=None)
    setup = StrategySetup(strategy="pairs_relative_value", instrument="NVDA/AAPL",
                          direction="neutral", horizon="swing")
    card = agent.simulate(setup)
    assert card.simulated is False
    assert "neutral" in card.skip_reason.lower()


def test_simulate_report_returns_comparison(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=None)
    setups = [
        StrategySetup(strategy="trend_following", instrument="NVDA", direction="long"),
        StrategySetup(strategy="ma_crossover", instrument="NVDA", direction="long"),
    ]
    cards, comparison = agent.simulate_report(setups, risk=_risk_with())
    assert len(cards) == 2
    assert all(c.backtest is not None for c in cards if c.simulated)


def test_missing_data_skips_gracefully(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=None)
    setup = StrategySetup(strategy="trend_following", instrument="ZZZZ", direction="long")
    card = agent.simulate(setup, risk=_risk_with())
    assert card.simulated is False
    assert "history" in card.skip_reason.lower()


def test_llm_verdict_used_when_available(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market,
                           llm=FakeLLM(response={"verdict": "Take it."}, available=True))
    setup = StrategySetup(strategy="trend_following", instrument="NVDA", direction="long")
    card = agent.simulate(setup, risk=_risk_with())
    assert card.llm_used is True
    assert card.verdict == "Take it."


def test_llm_outage_falls_back(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market,
                           llm=FakeLLM(available=True, raise_error=True))
    setup = StrategySetup(strategy="trend_following", instrument="NVDA", direction="long")
    card = agent.simulate(setup, risk=_risk_with())
    assert card.llm_used is False
    assert card.verdict


def test_default_constraints_when_no_risk(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=None)
    setup = StrategySetup(strategy="trend_following", instrument="NVDA", direction="long")
    card = agent.simulate(setup, risk=None)
    # Falls back to 1% risk / 10% max position defaults.
    assert card.sizing.max_position_pct == 0.10
    assert card.sizing.units > 0


def test_render_smoke(agent_market):
    agent = ExecutionAgent(twelvedata=None, market=agent_market, llm=None)
    setup = StrategySetup(strategy="trend_following", strategy_name="Trend Following",
                          instrument="NVDA", direction="long")
    card = agent.simulate(setup, risk=_risk_with())
    text = card.render()
    assert "TRADE CARD" in text
    assert "SIMULATION" in text
    assert "BACKTEST" in text
    assert "SIZING" in text
