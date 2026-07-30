"""Live RAG retrieval evaluation against the Qdrant corpus.

Without labeled relevance judgements this suite reports operational retrieval
quality: hit rate, similarity scores, and source diversity for probe queries.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from dotenv import load_dotenv

from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult
from rag.embedder import TextEmbedder
from rag.retriever import VectorRetriever

# Probe queries exercised against the live corpus (not golden-labelled).
PROBE_QUERIES: list[tuple[str, str]] = [
    ("fed_policy", "Federal Reserve interest rate policy and inflation outlook"),
    ("sec_filings", "SEC filing revenue operating margin and cash flow"),
    ("macro_growth", "macroeconomic growth employment and monetary conditions"),
    ("market_volatility", "equity market volatility risk sentiment and positioning"),
    ("mega_cap", "mega-cap technology earnings guidance and valuation"),
]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run_rag_eval(
    *,
    top_k: int = 8,
    score_threshold: float = 0.40,
    progress: Callable[[str, dict], None] | None = None,
) -> SuiteResult:
    """Evaluate live semantic retrieval against Qdrant."""
    started = time.perf_counter()
    label = "RAG retrieval (live corpus)"
    try:
        load_dotenv()
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection = os.getenv("QDRANT_COLLECTION", "financial_docs")

        if progress:
            progress("stage_started", {"stage": "rag", "detail": "Connecting to Qdrant"})

        embedder = TextEmbedder()
        retriever = VectorRetriever(qdrant_url, collection, embedder)
        corpus_size = retriever.count()

        if corpus_size == 0:
            return SuiteResult(
                suite="rag",
                label=label,
                status="failed",
                error="Corpus is empty — build the document index before running RAG evaluation.",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )

        cases: list[EvalCaseResult] = []
        hit_flags: list[float] = []
        top_scores: list[float] = []
        mean_scores: list[float] = []
        source_counts: list[float] = []
        doc_type_counts: list[float] = []

        for case_id, query in PROBE_QUERIES:
            if progress:
                progress("progress", {"stage": "rag", "case": case_id})

            hits = retriever.search(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            hit = 1.0 if hits else 0.0
            top_score = float(hits[0]["score"]) if hits else 0.0
            avg_score = _mean([float(item["score"]) for item in hits])
            sources = {str(item.get("source", "")) for item in hits if item.get("source")}
            doc_types = {str(item.get("doc_type", "")) for item in hits if item.get("doc_type")}

            hit_flags.append(hit)
            top_scores.append(top_score)
            mean_scores.append(avg_score)
            source_counts.append(float(len(sources)))
            doc_type_counts.append(float(len(doc_types)))

            cases.append(
                EvalCaseResult(
                    id=case_id,
                    label=query,
                    metrics=[
                        EvalMetric(
                            name="retrieval_hit",
                            label="Retrieval hit",
                            value=hit,
                            unit="ratio",
                            detail="1 when at least one chunk cleared the score threshold.",
                        ),
                        EvalMetric(
                            name="top_similarity",
                            label="Top similarity",
                            value=top_score,
                            unit="cosine",
                        ),
                        EvalMetric(
                            name="mean_similarity",
                            label=f"Mean similarity (top {top_k})",
                            value=avg_score,
                            unit="cosine",
                        ),
                        EvalMetric(
                            name="unique_sources",
                            label="Unique sources",
                            value=len(sources),
                            unit="count",
                        ),
                        EvalMetric(
                            name="unique_doc_types",
                            label="Unique doc types",
                            value=len(doc_types),
                            unit="count",
                        ),
                    ],
                    notes=", ".join(sorted(doc_types)) if doc_types else "No hits",
                )
            )

        aggregate = [
            EvalMetric(
                name="corpus_points",
                label="Corpus size",
                value=corpus_size,
                unit="chunks",
            ),
            EvalMetric(
                name="retrieval_hit_rate",
                label="Retrieval hit rate",
                value=_mean(hit_flags),
                unit="ratio",
                detail=(
                    "Share of probe queries returning at least one chunk above "
                    f"{score_threshold:.0%} similarity. Not labelled recall@k."
                ),
            ),
            EvalMetric(
                name="mean_top_similarity",
                label="Mean top-1 similarity",
                value=_mean(top_scores),
                unit="cosine",
            ),
            EvalMetric(
                name="mean_query_similarity",
                label=f"Mean top-{top_k} similarity",
                value=_mean(mean_scores),
                unit="cosine",
            ),
            EvalMetric(
                name="mean_unique_sources",
                label="Mean unique sources",
                value=_mean(source_counts),
                unit="count",
            ),
            EvalMetric(
                name="mean_unique_doc_types",
                label="Mean unique doc types",
                value=_mean(doc_type_counts),
                unit="count",
            ),
        ]

        if progress:
            progress("stage_completed", {"stage": "rag", "cases": len(cases)})

        return SuiteResult(
            suite="rag",
            label=label,
            status="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            metrics=aggregate,
            cases=cases,
        )
    except Exception as exc:  # noqa: BLE001 - eval must return structured failure
        return SuiteResult(
            suite="rag",
            label=label,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
