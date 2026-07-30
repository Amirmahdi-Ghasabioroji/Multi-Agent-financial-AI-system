"""Evaluation suite orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from eval.rag_eval import run_rag_eval
from eval.risk_eval import run_risk_eval
from eval.schemas import EvalMetric, EvalSuiteName, EvaluationReport, SuiteResult
from eval.simulation_eval import run_simulation_eval

ProgressCallback = Callable[[str, dict[str, Any]], None]

_SUITE_RUNNERS = {
    "rag": run_rag_eval,
    "simulation": run_simulation_eval,
    "risk": run_risk_eval,
}


def _resolve_suites(requested: list[str] | None) -> list[EvalSuiteName]:
    if not requested or "all" in requested:
        return ["rag", "simulation", "risk"]
    suites: list[EvalSuiteName] = []
    for item in requested:
        if item in _SUITE_RUNNERS and item not in suites:
            suites.append(item)  # type: ignore[arg-type]
    return suites or ["rag", "simulation", "risk"]


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
            results.append(
                run_rag_eval(top_k=top_k, progress=progress_callback)
            )
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

    summary: list[EvalMetric] = []
    for result in results:
        if result.status != "completed":
            continue
        for metric in result.metrics:
            if metric.name in {
                "retrieval_hit_rate",
                "mean_top_similarity",
                "mean_calibration_error",
                "max_calibration_error",
                "metric_completeness",
                "mean_realised_vol",
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
        "RAG metrics are live retrieval quality probes — not labelled recall@k.",
        "Monte Carlo calibration compares bootstrap probabilities to walk-forward outcomes.",
        "Risk metrics are computed from live market data without LLM narration.",
    ]

    return EvaluationReport(suites=results, summary_metrics=summary, notes=notes)
