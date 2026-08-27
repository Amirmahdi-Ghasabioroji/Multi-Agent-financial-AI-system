"""Analyst briefing evaluation: citation faithfulness and groundedness."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from dotenv import load_dotenv

from eval.faithfulness import evaluate_faithfulness, judge_briefing
from eval.json_util import load_gold
from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def run_analyst_eval(
    progress: Callable[[str, dict], None] | None = None,
    *,
    max_queries: int = 10,
) -> SuiteResult:
    started = time.perf_counter()
    label = "Analyst briefing faithfulness"
    try:
        if progress:
            progress("stage_started", {"stage": "analyst"})

        load_dotenv()
        from agents.analyst import build_analyst_agent
        from agents.llm import OllamaClient

        agent = build_analyst_agent()
        gold = load_gold("rag_queries.json")[:max_queries]
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
        judge_llm = OllamaClient(url=ollama_url, model=ollama_model)
        judge_available = judge_llm.is_available()

        cases: list[EvalCaseResult] = []
        validity: list[float] = []
        overlap: list[float] = []
        grounded: list[float] = []
        contradictions: list[float] = []
        llm_selfs: list[float] = []
        routing: list[float] = []
        judge_scores: list[float] = []

        for spec in gold:
            case_id = str(spec.get("id", "q"))
            query = str(spec.get("query", ""))
            if progress:
                progress("progress", {"stage": "analyst", "case": case_id})
            briefing = agent.brief(query, use_llm=True)
            hits = [
                {
                    "text": c.excerpt,
                    "source": c.source,
                    "doc_type": c.doc_type,
                }
                for c in briefing.citations
            ]
            # Prefer full chunk text when the retriever is reachable.
            try:
                hits = agent.retriever.search(query, top_k=agent.top_k, score_threshold=agent.score_threshold)
            except Exception:  # noqa: BLE001
                pass

            faith = evaluate_faithfulness(briefing, hits)
            judged = judge_briefing(judge_llm, briefing, hits) if judge_available else None

            validity.append(faith["citation_validity"])
            overlap.append(faith["mean_overlap"])
            grounded.append(faith["groundedness"])
            contradictions.append(faith["contradiction_rate"])
            routing.append(float(briefing.confidence))
            if briefing.llm_self_confidence is not None:
                llm_selfs.append(float(briefing.llm_self_confidence))
            if judged:
                judge_scores.append(judged["judge_support_rate"])

            metrics = [
                EvalMetric(name="citation_validity", label="Citation validity", value=faith["citation_validity"], unit="ratio"),
                EvalMetric(name="mean_overlap", label="Claim–chunk overlap", value=faith["mean_overlap"], unit="ratio"),
                EvalMetric(name="groundedness", label="Groundedness", value=faith["groundedness"], unit="ratio"),
                EvalMetric(name="contradiction_rate", label="Contradiction rate", value=faith["contradiction_rate"], unit="ratio"),
                EvalMetric(
                    name="routing_confidence",
                    label="Routing confidence (no llm_self)",
                    value=round(briefing.confidence, 4),
                    unit="ratio",
                ),
                EvalMetric(
                    name="llm_self",
                    label="LLM self-confidence (uncalibrated)",
                    value=briefing.llm_self_confidence,
                    unit="ratio",
                    detail="Displayed only — not used for orchestrator gates.",
                ),
            ]
            if judged:
                metrics.extend(
                    [
                        EvalMetric(
                            name="judge_support_rate",
                            label="LLM-as-judge support rate",
                            value=judged["judge_support_rate"],
                            unit="ratio",
                            detail="Same local model family as the generator — treat as optional.",
                        ),
                        EvalMetric(
                            name="judge_mean_score",
                            label="LLM-as-judge mean score",
                            value=judged["judge_mean_score"],
                            unit="ratio",
                        ),
                    ]
                )
            cases.append(
                EvalCaseResult(
                    id=case_id,
                    label=query,
                    metrics=metrics,
                    notes="failure_case" if spec.get("failure_case") else briefing.model,
                )
            )

        aggregate = [
            EvalMetric(name="mean_citation_validity", label="Mean citation validity", value=_mean(validity), unit="ratio"),
            EvalMetric(name="mean_overlap", label="Mean claim–chunk overlap", value=_mean(overlap), unit="ratio"),
            EvalMetric(name="mean_groundedness", label="Mean groundedness", value=_mean(grounded), unit="ratio"),
            EvalMetric(name="mean_contradiction_rate", label="Mean contradiction rate", value=_mean(contradictions), unit="ratio"),
            EvalMetric(
                name="mean_routing_confidence",
                label="Mean routing confidence",
                value=_mean(routing),
                unit="ratio",
                detail="Evidence-only. llm_self is isolated from routing.",
            ),
            EvalMetric(
                name="mean_llm_self",
                label="Mean llm_self (uncalibrated)",
                value=_mean(llm_selfs),
                unit="ratio",
            ),
            EvalMetric(
                name="mean_judge_support_rate",
                label="Mean LLM-as-judge support",
                value=_mean(judge_scores),
                unit="ratio",
                detail="Skipped when Ollama is down.",
            ),
            EvalMetric(name="queries_scored", label="Queries scored", value=len(cases), unit="count"),
        ]

        if progress:
            progress("stage_completed", {"stage": "analyst", "cases": len(cases)})

        return SuiteResult(
            suite="analyst",
            label=label,
            status="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            metrics=aggregate,
            cases=cases,
        )
    except Exception as exc:  # noqa: BLE001
        return SuiteResult(
            suite="analyst",
            label=label,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
