"""Unit tests for the LangGraph orchestration layer (fake agents, no network)."""

import pytest

from agents.execution_schemas import SimulationStats, TradeCard
from agents.orchestrator import RiskPipeline
from agents.pipeline_schemas import PipelineResult
from agents.risk_schemas import ConcentrationRisk, PositionSizingConstraint, RiskSummary
from agents.schemas import MacroBriefing
from agents.strategy_schemas import MacroBias, StrategyReport, StrategySetup


class FakeAnalyst:
    """Returns briefings with a scripted confidence per attempt."""

    llm = None  # forces the deterministic broaden fallback

    def __init__(self, confidences):
        self._confidences = list(confidences)
        self.calls = 0
        self.queries = []

    def brief(self, query, doc_type=None, date_after=None, conversation_context=None, use_llm=True):
        self.queries.append(query)
        conf = self._confidences[min(self.calls, len(self._confidences) - 1)]
        self.calls += 1
        return MacroBriefing(query=query, summary="briefing text", confidence=conf)


class FakeRisk:
    def __init__(self, regime="medium", tickers=("NVDA",), raise_error=False):
        self.regime = regime
        self.tickers = list(tickers)
        self.lookback_days = 252
        self._raise = raise_error

    def assess(self, universe=None, briefing=None):
        if self._raise:
            raise RuntimeError("risk boom")
        return RiskSummary(
            universe=self.tickers,
            vol_regime=self.regime,
            vix_level=18.0,
            concentration=ConcentrationRisk(mean_pairwise_correlation=0.25, n_assets=len(self.tickers)),
            position_sizing=[
                PositionSizingConstraint(ticker=t, max_position_pct=0.1, risk_per_trade_pct=0.01)
                for t in self.tickers
            ],
        )


class FakeStrategy:
    def __init__(self, setups, raise_error=False):
        self._setups = setups
        self._raise = raise_error

    def decide(self, risk, briefing=None):
        if self._raise:
            raise RuntimeError("strategy boom")
        return StrategyReport(
            macro_bias=MacroBias(direction="bullish", strength=0.7),
            vol_regime=risk.vol_regime,
            universe=risk.universe,
            setups=self._setups,
        )


class FakeExecution:
    def simulate_report(self, setups, risk=None):
        from agents.execution_schemas import (
            BacktestMetrics,
            BacktestResult,
            ExecutionComparison,
            ExecutionComparisonEntry,
        )

        cards = []
        for s in setups:
            cards.append(
                TradeCard(
                    instrument=s.instrument,
                    strategy=s.strategy,
                    strategy_name=s.strategy_name or s.strategy,
                    direction=s.direction,
                    simulated=True,
                    stats=SimulationStats(prob_tp_before_sl=0.5, expected_r=0.3, n_sims=100),
                    backtest=BacktestResult(
                        metrics=BacktestMetrics(
                            n_trades=10,
                            total_pnl=5000.0,
                            sharpe_ratio=1.2,
                            max_drawdown_pct=0.05,
                        )
                    ),
                    expectancy_amount=100.0,
                )
            )
        comparison = ExecutionComparison(
            ranked=[
                ExecutionComparisonEntry(
                    rank=1,
                    instrument=cards[0].instrument or "",
                    strategy=cards[0].strategy_name,
                    composite_score=0.8,
                    sharpe_ratio=1.2,
                    total_pnl=5000.0,
                    max_drawdown_pct=0.05,
                    expected_r_forward=0.3,
                )
            ],
            best_sharpe="Trend Following | NVDA",
        )
        return cards, comparison


def _setup(conf=0.7, instrument="NVDA", direction="long"):
    return StrategySetup(
        strategy="trend_following", strategy_name="Trend Following",
        instrument=instrument, direction=direction, confidence=conf, playbook_fit=conf,
    )


def _pipeline(analyst, risk, strategy, execution):
    return RiskPipeline(analyst=analyst, risk=risk, strategy=strategy, execution=execution)


# --------------------------------------------------------------------------- #
def test_happy_path_trade():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(), FakeStrategy([_setup(0.7)]), FakeExecution())
    result = p.run("fed outlook", tickers=["NVDA"], use_llm=False)
    assert isinstance(result, PipelineResult)
    assert result.decision == "trade"
    assert result.analyst_attempts == 1
    assert result.route_log == ["analyst(attempt=1)", "risk", "strategy", "execution"]
    assert len(result.tradeable_cards) == 1


def test_broaden_loop_then_recovers():
    # Low confidence first, high on the second attempt.
    analyst = FakeAnalyst([0.2, 0.8])
    p = _pipeline(analyst, FakeRisk(), FakeStrategy([_setup(0.7)]), FakeExecution())
    result = p.run("very narrow query", use_llm=False)
    assert result.analyst_attempts == 2
    assert "broaden" in result.route_log
    assert result.decision == "trade"
    # The query was broadened (differs from the original).
    assert result.query != result.original_query
    assert analyst.queries[1] != analyst.queries[0]


def test_retries_exhausted_then_proceeds():
    analyst = FakeAnalyst([0.1])  # always low
    p = _pipeline(analyst, FakeRisk(), FakeStrategy([_setup(0.7)]), FakeExecution())
    result = p.run("q", use_llm=False)
    # 1 initial + 2 retries = 3 attempts, then proceeds to risk.
    assert result.analyst_attempts == 3
    assert result.route_log.count("broaden") == 2
    assert "risk" in result.route_log


def test_no_trade_when_setups_below_floor():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(), FakeStrategy([_setup(0.30)]), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "no_trade"
    assert "execution" not in result.route_log
    assert "floor" in result.no_trade_reason.lower()


def test_no_trade_when_no_setups():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(), FakeStrategy([]), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "no_trade"
    assert "no setups" in result.no_trade_reason.lower()


def test_no_trade_high_vol_weak_setup():
    # Setup clears the normal floor (0.45) but not the high-vol floor (0.55).
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(regime="high"), FakeStrategy([_setup(0.50)]), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "no_trade"
    assert "high-vol" in result.no_trade_reason.lower() or "high vol" in result.no_trade_reason.lower()


def test_high_vol_strong_setup_trades():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(regime="high"), FakeStrategy([_setup(0.70)]), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "trade"
    assert result.execution_comparison is not None
    assert len(result.execution_comparison.ranked) >= 1


def test_neutral_only_setups_no_trade():
    setups = [_setup(0.8, instrument="NVDA/AAPL", direction="neutral")]
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(), FakeStrategy(setups), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "no_trade"


def test_risk_failure_degrades_to_no_trade():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(raise_error=True), FakeStrategy([_setup()]), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "no_trade"
    assert any("risk" in e for e in result.errors)


def test_strategy_failure_degrades_to_no_trade():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(), FakeStrategy([_setup()], raise_error=True), FakeExecution())
    result = p.run("q", use_llm=False)
    assert result.decision == "no_trade"
    assert any("strategy" in e for e in result.errors)


def test_render_smoke():
    p = _pipeline(FakeAnalyst([0.8]), FakeRisk(), FakeStrategy([_setup(0.7)]), FakeExecution())
    text = p.run("fed outlook", tickers=["NVDA"], use_llm=False).render()
    assert "MAFAS PIPELINE" in text
    assert "Route:" in text
    assert "Decision: TRADE" in text


def test_progress_callback_reports_stages_and_completion():
    events = []
    p = _pipeline(
        FakeAnalyst([0.8]),
        FakeRisk(),
        FakeStrategy([_setup(0.7)]),
        FakeExecution(),
    )
    result = p.run(
        "fed outlook",
        use_llm=False,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert result.decision == "trade"
    names = [event for event, _ in events]
    assert names[0] == "pipeline_started"
    assert ("stage_started", {"stage": "analyst", "attempt": 1}) in events
    assert names[-1] == "pipeline_completed"


def test_conversation_context_is_forwarded_to_analyst():
    class ContextAnalyst(FakeAnalyst):
        def brief(
            self,
            query,
            doc_type=None,
            date_after=None,
            conversation_context=None,
            use_llm=True,
        ):
            self.context = conversation_context
            return super().brief(query, doc_type=doc_type, date_after=date_after, use_llm=use_llm)

    analyst = ContextAnalyst([0.8])
    p = _pipeline(
        analyst,
        FakeRisk(),
        FakeStrategy([_setup(0.7)]),
        FakeExecution(),
    )
    p.run("What changed?", conversation_context="Earlier we discussed inflation.")
    assert analyst.context == "Earlier we discussed inflation."
