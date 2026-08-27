"""Tests for the evaluation harness."""

from agents.orchestrator import strategy_route
from agents.risk_schemas import ConcentrationRisk, RiskSummary
from agents.schemas import KeyPoint, MacroBriefing, SourceCitation
from agents.strategy_schemas import StrategyReport, StrategySetup
from eval.calibration import brier_score, reliability_bins
from eval.faithfulness import evaluate_faithfulness
from eval.gates_eval import AttemptSnapshot, QueryTrace, sweep_precision
from eval.ir_metrics import labelled_query_metrics
from eval.json_util import load_gold
from eval.runner import run_evaluation
from eval.schemas import EvaluationReport
from eval.simulation_eval import _aligned_case, _trending_df


def test_simulation_calibration_case_has_aligned_metrics():
    frame = _trending_df(300, 0.003)
    case = _aligned_case(
        "trend",
        "Trend long",
        frame,
        direction="long",
        stop_pct=0.03,
        target_pct=0.06,
        horizon=20,
        barrier_mode="close",
        n_sims=400,
    )
    names = {metric.name for metric in case.metrics}
    assert "calibration_error" in names
    assert "empirical_tp_rate" in names
    assert "mc_prob_tp_before_sl" in names
    assert "brier_score" in names


def test_run_evaluation_simulation_only():
    report = run_evaluation(suites=["simulation"])
    assert isinstance(report, EvaluationReport)
    assert len(report.suites) == 1
    assert report.suites[0].suite == "simulation"
    assert report.suites[0].status == "completed"
    assert report.suites[0].metrics


def test_all_does_not_include_opt_in_suites():
    from eval.runner import _resolve_suites

    assert _resolve_suites(["all"]) == ["rag", "simulation", "risk"]
    assert _resolve_suites(["analyst", "gates"]) == ["analyst", "gates"]


def test_gold_files_load():
    rag = load_gold("rag_queries.json")
    gates = load_gold("gates_queries.json")
    assert len(rag) >= 20
    assert len(gates) >= 20
    assert any(q.get("failure_case") for q in rag)
    assert {q["expected"] for q in gates} == {"trade", "no_trade"}


def test_labelled_ir_metrics_distinguish_off_topic():
    spec = {"doc_types": ["sec_filing"], "source_includes": ["aapl"]}
    hits = [
        {"doc_type": "fomc_minutes", "source": "https://federalreserve.gov/x"},
        {"doc_type": "sec_filing", "source": "https://sec.gov/aapl-10k.htm"},
    ]
    corpus = hits + [
        {"doc_type": "sec_filing", "source": "https://sec.gov/msft-10k.htm"},
        {"doc_type": "sec_filing", "source": "https://sec.gov/aapl-10q.htm"},
    ]
    metrics = labelled_query_metrics(hits, corpus, spec, k=2)
    assert metrics["labelled_hit"] == 1.0
    assert metrics["precision_at_k"] == 0.5
    assert metrics["corpus_relevant"] == 3.0


def test_faithfulness_scores_supported_citation():
    briefing = MacroBriefing(
        query="fed",
        summary="The FOMC held rates steady [1].",
        key_points=[
            KeyPoint(
                point="The FOMC held rates steady amid inflation [1]",
                citations=[1],
                confidence=0.8,
            )
        ],
        citations=[
            SourceCitation(
                index=1,
                source="fed",
                doc_type="fomc_minutes",
                date="2024-01-01",
                score=0.7,
                excerpt="The FOMC held rates steady amid moderating inflation.",
            )
        ],
        confidence=0.6,
    )
    faith = evaluate_faithfulness(briefing)
    assert faith["citation_validity"] == 1.0
    assert faith["groundedness"] > 0
    assert faith["n_claims"] >= 1


def test_brier_and_reliability():
    assert brier_score([0.2, 0.8], [0, 1]) == 0.04
    rows = reliability_bins([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 0], n_bins=2)
    assert len(rows) == 2
    assert rows[0]["count"] == 2


def test_strategy_baseline_stats_on_synthetic_trend():
    from eval.strategy_eval import buy_and_hold_stats, sma50_long_flat_stats

    df = _trending_df(120, 0.002)
    bh = buy_and_hold_stats(df)
    sma = sma50_long_flat_stats(df)
    assert bh["total_return_pct"] > 0
    assert sma["n_days"] == 120.0
    setups_trade = [
        StrategySetup(
            strategy="trend_following",
            instrument="AAPL",
            direction="long",
            confidence=0.7,
        )
    ]
    setups_weak = [
        StrategySetup(
            strategy="trend_following",
            instrument="AAPL",
            direction="long",
            confidence=0.3,
        )
    ]

    def _trace(expected: str, confidence: float, setups: list[StrategySetup]) -> QueryTrace:
        briefing = MacroBriefing(query="q", summary="s", confidence=confidence)
        risk = RiskSummary(
            universe=["AAPL"],
            vol_regime="medium",
            concentration=ConcentrationRisk(n_assets=1),
        )
        strategy = StrategyReport(setups=setups, vol_regime="medium", universe=["AAPL"])
        snap = AttemptSnapshot(
            attempt=1,
            query="q",
            confidence=confidence,
            briefing=briefing,
            risk=risk,
            strategy=strategy,
        )
        return QueryTrace(spec={"expected": expected}, use_llm=False, attempts=[snap])

    traces = [
        _trace("trade", 0.8, setups_trade),
        _trace("no_trade", 0.8, setups_weak),
        _trace("no_trade", 0.2, setups_weak),
    ]
    low_floor = sweep_precision(traces, 0.30, 0.25, 0.55)
    high_floor = sweep_precision(traces, 0.30, 0.80, 0.55)
    assert low_floor["trade_rate"] > high_floor["trade_rate"]
    assert strategy_route(setups_trade, "medium", 0.45, 0.55) == "execution"
    assert strategy_route(setups_weak, "medium", 0.45, 0.55) == "no_trade"
