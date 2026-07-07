"""Unit tests for the Risk Agent (synthetic market data, mocked LLM, no network)."""

import numpy as np
import pandas as pd
import pytest

from agents.risk import DEFAULT_WATCHLIST, RiskAgent
from agents.risk_metrics import (
    atr_percent,
    average_true_range,
    classify_asset_regime,
    classify_market_regime,
    correlation_matrix,
    effective_number_of_bets,
    high_correlation_pairs,
    mean_pairwise_correlation,
    realised_vol,
    suggested_position_sizing,
)
from agents.risk_schemas import RiskSummary
from agents.schemas import MacroBriefing


def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a close-price path."""
    close = pd.Series(prices, dtype=float)
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": high,
            "low": low,
            "close": close,
            "volume": [1_000_000] * len(close),
        }
    )


def _random_walk(seed: int, n: int = 260, vol: float = 0.01, start: float = 100.0) -> list[float]:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, vol, size=n)
    return list(start * np.exp(np.cumsum(steps)))


class FakeMarket:
    """Stand-in for MarketDataLoader returning canned OHLCV / VIX."""

    def __init__(self, frames: dict[str, pd.DataFrame], vix: float = 18.0) -> None:
        self._frames = frames
        self._vix = vix

    def get_ohlcv(self, ticker: str, days: int = 252) -> pd.DataFrame:
        if ticker not in self._frames:
            raise ValueError(f"no data for {ticker}")
        return self._frames[ticker]

    def get_vix(self) -> float:
        return self._vix


class FakeLLM:
    """Stand-in for OllamaClient."""

    model = "mistral"

    def __init__(self, response=None, available=True, raise_error=False):
        self._response = response or {}
        self._available = available
        self._raise = raise_error

    def is_available(self) -> bool:
        return self._available

    def chat_json(self, messages, options=None):
        if self._raise:
            from agents.llm import OllamaError

            raise OllamaError("simulated outage")
        return self._response


# --------------------------------------------------------------------------- #
# Pure metric functions
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_realised_vol_zero_for_flat_series(self):
        flat = pd.Series([100.0] * 50)
        assert realised_vol(flat) == 0.0

    def test_realised_vol_positive_and_annualised(self):
        vol = realised_vol(pd.Series(_random_walk(1, vol=0.02)))
        assert vol > 0.0
        # ~0.02 daily * sqrt(252) ≈ 0.32 annualised, allow a wide band.
        assert 0.1 < vol < 0.6

    def test_atr_nonnegative_and_pct(self):
        df = _make_ohlcv(_random_walk(2))
        assert average_true_range(df) >= 0.0
        pct = atr_percent(df)
        assert 0.0 <= pct < 0.2

    def test_atr_handles_missing_columns(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        assert average_true_range(df) == 0.0

    def test_classify_asset_regime_thresholds(self):
        assert classify_asset_regime(0.10) == "low"
        assert classify_asset_regime(0.25) == "medium"
        assert classify_asset_regime(0.50) == "high"

    def test_classify_market_regime_takes_cautious_signal(self):
        # Low VIX but high realised vol -> escalates to high.
        assert classify_market_regime(12.0, 0.50) == "high"
        # High VIX dominates a calm basket.
        assert classify_market_regime(30.0, 0.05) == "high"
        assert classify_market_regime(12.0, 0.05) == "low"

    def test_correlation_and_high_pairs(self):
        base = _random_walk(3)
        returns = {
            "A": pd.Series(base).pct_change().dropna().reset_index(drop=True),
            "B": pd.Series(base).pct_change().dropna().reset_index(drop=True),
            "C": pd.Series(_random_walk(99)).pct_change().dropna().reset_index(drop=True),
        }
        corr = correlation_matrix(returns)
        assert not corr.empty
        pairs = high_correlation_pairs(corr, threshold=0.8)
        flagged = {frozenset((a, b)) for a, b, _ in pairs}
        # A and B are identical paths -> must be flagged.
        assert frozenset(("A", "B")) in flagged

    def test_effective_bets_bounds(self):
        # Two uncorrelated-ish assets -> close to 2 effective bets.
        returns = {
            "A": pd.Series(_random_walk(4)).pct_change().dropna().reset_index(drop=True),
            "B": pd.Series(_random_walk(5)).pct_change().dropna().reset_index(drop=True),
        }
        corr = correlation_matrix(returns)
        eff = effective_number_of_bets(corr)
        assert 1.0 <= eff <= 2.0

    def test_position_sizing_scales_down_in_high_regime(self):
        vols = {"AAPL": 0.30, "MSFT": 0.30}
        low = suggested_position_sizing(vols, "low", mean_corr=0.2)
        high = suggested_position_sizing(vols, "high", mean_corr=0.2)
        assert high["AAPL"]["max_position_pct"] < low["AAPL"]["max_position_pct"]
        assert high["AAPL"]["risk_per_trade_pct"] < low["AAPL"]["risk_per_trade_pct"]

    def test_position_sizing_never_exceeds_cap(self):
        vols = {"AAPL": 0.05}  # very low vol would push inverse-vol weight high
        sizing = suggested_position_sizing(vols, "low", mean_corr=0.0, max_position=0.25)
        assert sizing["AAPL"]["max_position_pct"] <= 0.25


# --------------------------------------------------------------------------- #
# RiskAgent orchestration
# --------------------------------------------------------------------------- #
@pytest.fixture
def three_asset_market() -> FakeMarket:
    frames = {
        "AAPL": _make_ohlcv(_random_walk(10, vol=0.012)),
        "MSFT": _make_ohlcv(_random_walk(11, vol=0.012)),
        "NVDA": _make_ohlcv(_random_walk(12, vol=0.03)),
    }
    return FakeMarket(frames, vix=18.0)


def test_assess_produces_full_summary(three_asset_market):
    agent = RiskAgent(market=three_asset_market, llm=FakeLLM(available=False))
    summary = agent.assess(universe=["AAPL", "MSFT", "NVDA"])
    assert isinstance(summary, RiskSummary)
    assert summary.vol_regime in {"low", "medium", "high"}
    assert len(summary.per_asset) == 3
    assert len(summary.position_sizing) == 3
    assert summary.concentration.n_assets == 3
    assert summary.narrative  # fallback narrative present


def test_default_watchlist_is_included(three_asset_market):
    # Only AAPL/MSFT/NVDA have data; the rest of the watchlist is skipped safely.
    agent = RiskAgent(market=three_asset_market, llm=None)
    summary = agent.assess()
    assert set(summary.universe).issubset(set(DEFAULT_WATCHLIST))
    assert len(summary.per_asset) == 3


def test_llm_narrative_used_when_available(three_asset_market):
    llm = FakeLLM(
        response={"narrative": "Regime is benign.", "watch_items": ["VIX spike"]},
        available=True,
    )
    agent = RiskAgent(market=three_asset_market, llm=llm)
    summary = agent.assess(universe=["AAPL"])
    assert summary.llm_used is True
    assert summary.narrative == "Regime is benign."
    assert summary.watch_items == ["VIX spike"]


def test_llm_outage_falls_back_gracefully(three_asset_market):
    llm = FakeLLM(available=True, raise_error=True)
    agent = RiskAgent(market=three_asset_market, llm=llm)
    summary = agent.assess(universe=["AAPL"])
    assert summary.llm_used is False
    assert summary.narrative  # deterministic fallback

def test_briefing_context_threaded_in(three_asset_market):
    briefing = MacroBriefing(
        query="rates outlook",
        summary="The Fed is on hold.",
        confidence=0.62,
    )
    agent = RiskAgent(market=three_asset_market, llm=None)
    summary = agent.assess(universe=["AAPL"], briefing=briefing)
    assert summary.analyst_query == "rates outlook"
    assert summary.analyst_confidence == 0.62
    assert summary.macro_context == "The Fed is on hold."


def test_no_price_data_returns_safe_summary():
    empty_market = FakeMarket({}, vix=28.0)
    agent = RiskAgent(market=empty_market, llm=None)
    summary = agent.assess(universe=["ZZZZ"])
    assert summary.per_asset == []
    assert summary.vol_regime == "high"  # driven by VIX fallback
    assert summary.narrative


def test_render_smoke(three_asset_market):
    agent = RiskAgent(market=three_asset_market, llm=None)
    summary = agent.assess(universe=["AAPL", "MSFT"])
    text = summary.render()
    assert "RISK ENVIRONMENT" in text
    assert "PER-ASSET VOLATILITY" in text
    assert "SUGGESTED POSITION-SIZING CONSTRAINTS" in text
