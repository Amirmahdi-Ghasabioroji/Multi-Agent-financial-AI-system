"""Unit tests for the Strategy Agent (deterministic scoring + mocked LLM)."""

import pytest

from agents.risk_schemas import (
    AssetVolMetrics,
    ConcentrationRisk,
    CorrelationWarning,
    PositionSizingConstraint,
    RiskSummary,
)
from agents.schemas import MacroBriefing
from agents.strategy import StrategyAgent
from agents.strategy_playbooks import (
    PLAYBOOKS,
    Playbook,
    rank_playbooks,
    score_playbook,
)
from agents.strategy_schemas import StrategyReport


class FakeLLM:
    """Stand-in for OllamaClient supporting bias + strategy JSON responses."""

    model = "mistral"

    def __init__(self, responses=None, available=True, raise_error=False):
        # responses: list consumed in order across chat_json calls.
        self._responses = list(responses or [])
        self._available = available
        self._raise = raise_error

    def is_available(self) -> bool:
        return self._available

    def chat_json(self, messages, options=None):
        if self._raise:
            from agents.llm import OllamaError

            raise OllamaError("simulated outage")
        return self._responses.pop(0) if self._responses else {}


def _risk(regime="medium", mean_corr=0.25, warnings=False) -> RiskSummary:
    per_asset = [
        AssetVolMetrics(ticker="NVDA", last_price=195.0, realised_vol=0.42, regime="high"),
        AssetVolMetrics(ticker="AAPL", last_price=312.0, realised_vol=0.24, regime="medium"),
        AssetVolMetrics(ticker="JPM", last_price=337.0, realised_vol=0.18, regime="low"),
    ]
    corr_warnings = (
        [CorrelationWarning(pair=["NVDA", "AAPL"], correlation=0.86, note="coupled")]
        if warnings
        else []
    )
    return RiskSummary(
        universe=["NVDA", "AAPL", "JPM"],
        vol_regime=regime,
        vix_level=18.0,
        mean_realised_vol=0.28,
        per_asset=per_asset,
        correlation_warnings=corr_warnings,
        concentration=ConcentrationRisk(
            mean_pairwise_correlation=mean_corr,
            effective_number_of_bets=2.6,
            n_assets=3,
        ),
        position_sizing=[
            PositionSizingConstraint(ticker="NVDA", max_position_pct=0.10, risk_per_trade_pct=0.007),
            PositionSizingConstraint(ticker="AAPL", max_position_pct=0.15, risk_per_trade_pct=0.007),
            PositionSizingConstraint(ticker="JPM", max_position_pct=0.18, risk_per_trade_pct=0.007),
        ],
    )


# --------------------------------------------------------------------------- #
# Deterministic playbook scoring
# --------------------------------------------------------------------------- #
class TestScoring:
    def test_all_playbooks_score_in_range(self):
        for pb in PLAYBOOKS.values():
            score, _ = score_playbook(pb, "medium", "neutral", 0.5, 0.3)
            assert 0.0 <= score <= 1.0

    def test_high_vol_suppresses_trend_following(self):
        trend = PLAYBOOKS["trend_following"]
        low, _ = score_playbook(trend, "low", "bullish", 0.8, 0.3)
        high, _ = score_playbook(trend, "high", "bullish", 0.8, 0.3)
        assert high < low

    def test_high_vol_favours_volatility_based(self):
        vol = PLAYBOOKS["volatility_based"]
        low, _ = score_playbook(vol, "low", "neutral", 0.5, 0.3)
        high, _ = score_playbook(vol, "high", "neutral", 0.5, 0.3)
        assert high > low

    def test_neutral_bias_favours_mean_reversion_over_trend(self):
        mr, _ = score_playbook(PLAYBOOKS["mean_reversion"], "low", "neutral", 0.5, 0.2)
        tf, _ = score_playbook(PLAYBOOKS["trend_following"], "low", "neutral", 0.5, 0.2)
        assert mr > tf

    def test_high_correlation_favours_pairs(self):
        pairs = PLAYBOOKS["pairs_relative_value"]
        low_corr, _ = score_playbook(pairs, "medium", "neutral", 0.5, 0.1)
        high_corr, _ = score_playbook(pairs, "medium", "neutral", 0.5, 0.7, has_corr_warnings=True)
        assert high_corr > low_corr

    def test_rank_is_sorted_desc_and_complete(self):
        ranked = rank_playbooks("medium", "bullish", 0.7, 0.3)
        assert len(ranked) == len(PLAYBOOKS)
        scores = [s for _, s, _ in ranked]
        assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# Macro bias classification
# --------------------------------------------------------------------------- #
class TestBias:
    def test_fallback_detects_bearish(self):
        agent = StrategyAgent(llm=None)
        text = "Hawkish Fed, tightening, recession risk and a sharp selloff with sticky inflation."
        bias = agent._classify_bias_fallback(text)
        assert bias.direction == "bearish"
        assert bias.source == "fallback"
        assert bias.strength > 0.3

    def test_fallback_detects_bullish(self):
        agent = StrategyAgent(llm=None)
        text = "Dovish pivot with rate cuts, resilient growth and a broad rally; risk-on."
        bias = agent._classify_bias_fallback(text)
        assert bias.direction == "bullish"

    def test_fallback_neutral_when_no_signal(self):
        agent = StrategyAgent(llm=None)
        bias = agent._classify_bias_fallback("The report describes committee attendance.")
        assert bias.direction == "neutral"

    def test_llm_bias_used_when_available(self):
        llm = FakeLLM(responses=[{"direction": "bearish", "strength": 0.8, "rationale": "hawkish"}])
        agent = StrategyAgent(llm=llm)
        briefing = MacroBriefing(query="q", summary="Some macro text.")
        bias = agent._classify_bias(briefing)
        assert bias.direction == "bearish"
        assert bias.source == "llm"
        assert bias.strength == 0.8

    def test_no_briefing_is_neutral(self):
        agent = StrategyAgent(llm=None)
        bias = agent._classify_bias(None)
        assert bias.direction == "neutral"


# --------------------------------------------------------------------------- #
# Agent orchestration
# --------------------------------------------------------------------------- #
def test_decide_deterministic_fallback_no_llm():
    agent = StrategyAgent(llm=None)
    report = agent.decide(_risk(regime="high"), briefing=None)
    assert isinstance(report, StrategyReport)
    assert report.llm_used is False
    assert 1 <= len(report.setups) <= 3
    assert len(report.candidate_scores) == len(PLAYBOOKS)
    # Every fallback setup is bound to a known playbook and grounded fit.
    for s in report.setups:
        assert s.strategy in PLAYBOOKS
        assert s.playbook_fit >= 0.0


def test_decide_uses_llm_setups_and_grounds_confidence():
    bias_resp = {"direction": "bullish", "strength": 0.7, "rationale": "dovish"}
    strat_resp = {
        "setups": [
            {
                "strategy": "trend_following",
                "instrument": "NVDA",
                "direction": "long",
                "rationale": "clear uptrend",
                "confidence": 1.0,
                "horizon": "swing",
            }
        ],
        "suppressed": ["Carry: vol too high"],
        "narrative": "Lean into trend.",
    }
    agent = StrategyAgent(llm=FakeLLM(responses=[bias_resp, strat_resp]))
    report = agent.decide(_risk(regime="medium"), briefing=MacroBriefing(query="q", summary="text"))
    assert report.llm_used is True
    assert len(report.setups) == 1
    s = report.setups[0]
    assert s.instrument == "NVDA"
    assert s.strategy_name == "Trend Following"
    # LLM confidence 1.0 blended 50/50 with the (sub-1.0) deterministic fit.
    assert s.confidence < 1.0
    assert s.risk_note  # sizing constraint attached for NVDA


def test_llm_invalid_instrument_is_dropped():
    bias_resp = {"direction": "neutral", "strength": 0.5, "rationale": ""}
    strat_resp = {
        "setups": [
            {"strategy": "mean_reversion", "instrument": "TSLA", "direction": "neutral", "confidence": 0.6}
        ],
        "narrative": "n",
    }
    agent = StrategyAgent(llm=FakeLLM(responses=[bias_resp, strat_resp]))
    report = agent.decide(_risk(), briefing=MacroBriefing(query="q", summary="text"))
    # TSLA is not in the universe -> instrument cleared, setup still kept.
    assert report.setups[0].instrument is None


def test_llm_unknown_playbook_dropped_then_fallback():
    bias_resp = {"direction": "neutral", "strength": 0.5, "rationale": ""}
    strat_resp = {"setups": [{"strategy": "made_up_strategy", "instrument": "AAPL"}], "narrative": "n"}
    agent = StrategyAgent(llm=FakeLLM(responses=[bias_resp, strat_resp]))
    report = agent.decide(_risk(), briefing=MacroBriefing(query="q", summary="text"))
    # No valid LLM setups -> deterministic fallback kicks in.
    assert report.llm_used is False
    assert report.setups


def test_decide_llm_outage_falls_back():
    agent = StrategyAgent(llm=FakeLLM(available=True, raise_error=True))
    report = agent.decide(_risk(), briefing=MacroBriefing(query="q", summary="text"))
    assert report.llm_used is False
    assert report.setups


def test_analyst_context_threaded():
    agent = StrategyAgent(llm=None)
    briefing = MacroBriefing(query="rates path", summary="Fed on hold.", confidence=0.6)
    report = agent.decide(_risk(), briefing=briefing)
    assert report.analyst_query == "rates path"
    assert report.analyst_confidence == 0.6


def test_render_smoke():
    agent = StrategyAgent(llm=None)
    report = agent.decide(_risk(regime="low"), briefing=None)
    text = report.render()
    assert "STRATEGY VIEW" in text
    assert "PLAYBOOK SUITABILITY" in text
    assert "SUGGESTED SETUPS" in text
