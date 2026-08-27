"""Evaluation suite orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eval.analyst_eval import run_analyst_eval
from eval.gates_eval import run_gates_eval
from eval.rag_eval import run_rag_eval
from eval.risk_eval import run_risk_eval
from eval.schemas import EvalMetric, EvalSuiteName, EvaluationReport, SuiteResult
from eval.simulation_eval import run_simulation_eval
from eval.strategy_eval import run_strategy_eval

ProgressCallback = Callable[[str, dict[str, Any]], None]

_SUITE_RUNNERS = {
    "rag": run_rag_eval,
    "simulation": run_simulation_eval,
    "risk": run_risk_eval,
    "analyst": run_analyst_eval,
    "gates": run_gates_eval,
    "strategy": run_strategy_eval,
}

# Dashboard "all" stays the original three lightweight-ish suites.
_DEFAULT_ALL: list[EvalSuiteName] = ["rag", "simulation", "risk"]


def _resolve_suites(requested: list[str] | None) -> list[EvalSuiteName]:
    if not requested or "all" in requested:
        return list(_DEFAULT_ALL)
    suites: list[EvalSuiteName] = []
    for item in requested:
        if item in _SUITE_RUNNERS and item not in suites:
            suites.append(item)  # type: ignore[arg-type]
    return suites or list(_DEFAULT_ALL)


def run_evaluation(
    *,
    suites: list[str] | None = None,
    top_k: int = 8,
    lookback_days: int = 252,
    tickers: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EvaluationReport:
    """Run one or more evaluation suites and return a structured report."""
    selected = _resolve_suites(suites)
    results: list[SuiteResult] = []

    for suite in selected:
        if progress_callback:
            progress_callback("progress", {"stage": "evaluation", "suite": suite})

        if suite == "rag":
            results.append(run_rag_eval(top_k=top_k, progress=progress_callback))
        elif suite == "simulation":
            results.append(run_simulation_eval(progress=progress_callback))
        elif suite == "risk":
            results.append(
                run_risk_eval(
                    tickers=tickers,
                    lookback_days=lookback_days,
                    progress=progress_callback,
                )
            )
        elif suite == "analyst":
            results.append(run_analyst_eval(progress=progress_callback))
        elif suite == "gates":
            results.append(run_gates_eval(progress=progress_callback))
        elif suite == "strategy":
            results.append(run_strategy_eval(progress=progress_callback))

    summary: list[EvalMetric] = []
    for result in results:
        if result.status != "completed":
            continue
        for metric in result.metrics:
            if metric.name in {
                "retrieval_hit_rate",
                "mean_top_similarity",
                "mean_precision_at_k",
                "mean_recall_at_k",
                "mean_ndcg_at_k",
                "mean_calibration_error",
                "max_calibration_error",
                "live_mean_brier",
                "metric_completeness",
                "mean_realised_vol",
                "mean_citation_validity",
                "mean_groundedness",
                "default_precision_trade",
                "with_llm_precision_trade",
                "mean_excess_sharpe_vs_buy_hold",
                "llm_used_rate",
            }:
                summary.append(
                    EvalMetric(
                        name=f"{result.suite}_{metric.name}",
                        label=f"{result.label}: {metric.label}",
                        value=metric.value,
                        unit=metric.unit,
                        detail=metric.detail,
                    )
                )

    notes = [
        "RAG labelled metrics use source/doc_type gold — not passage-level relevance. Operational cosine hit rate is not relevance.",
        "Analyst routing confidence excludes llm_self (uncalibrated). Faithfulness is rule-based overlap; LLM-as-judge is optional and same-model.",
        "Orchestrator floors 0.40 / 0.45 / 0.55 are unchanged product policy. The gates suite reports operating points; it does not retune them.",
        "Monte Carlo: close-to-close MC is scored on close walks; OHLC bar-bootstrap is scored on high/low walks. Live names use a 70/30 walk-forward split.",
        "Sharpe / Sortino / Calmar on trade cards are calendar-time (session equity), not per-trade annualisation.",
        "Risk metrics are computed from live market data without LLM narration.",
        "'all' runs rag + simulation + risk. analyst, gates, and strategy are opt-in (heavier: Ollama / yfinance / pipeline traces).",
    ]

    return EvaluationReport(suites=results, summary_metrics=summary, notes=notes)
