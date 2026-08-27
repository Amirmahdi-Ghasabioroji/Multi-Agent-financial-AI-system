"""Live RAG retrieval evaluation against the Qdrant corpus.

Reports both operational probes (hit rate, cosine) and labelled IR metrics
(Recall@k, Precision@k, nDCG@k) on a source / doc_type gold set. Labels are
coarse: they are not passage-level relevance judgements.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from dotenv import load_dotenv

from eval.ir_metrics import labelled_query_metrics
from eval.json_util import load_gold
from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult
from rag.embedder import TextEmbedder
from rag.retriever import VectorRetriever


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
    label = "RAG retrieval (labelled + operational)"
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

        gold = load_gold("rag_queries.json")
        corpus = retriever.scroll_payloads()

        cases: list[EvalCaseResult] = []
        hit_flags: list[float] = []
        top_scores: list[float] = []
        mean_scores: list[float] = []
        source_counts: list[float] = []
        doc_type_counts: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        labelled_hits: list[float] = []
        failure_recalls: list[float] = []

        for spec in gold:
            case_id = str(spec.get("id", "query"))
            query = str(spec.get("query", ""))
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
            labelled = labelled_query_metrics(hits, corpus, spec, top_k)

            hit_flags.append(hit)
            top_scores.append(top_score)
            mean_scores.append(avg_score)
            source_counts.append(float(len(sources)))
            doc_type_counts.append(float(len(doc_types)))
            precisions.append(labelled["precision_at_k"])
            recalls.append(labelled["recall_at_k"])
            ndcgs.append(labelled["ndcg_at_k"])
            labelled_hits.append(labelled["labelled_hit"])
            if spec.get("failure_case"):
                failure_recalls.append(labelled["recall_at_k"])

            notes = (
                f"failure_case labelled recall@{top_k}={labelled['recall_at_k']}"
                if spec.get("failure_case")
                else ", ".join(sorted(doc_types)) if doc_types else "No hits"
            )
            cases.append(
                EvalCaseResult(
                    id=case_id,
                    label=query,
                    metrics=[
                        EvalMetric(
                            name="retrieval_hit",
                            label="Operational hit",
                            value=hit,
                            unit="ratio",
                            detail="1 when any chunk cleared the cosine threshold — not relevance.",
                        ),
                        EvalMetric(
                            name="labelled_hit",
                            label="Labelled hit",
                            value=labelled["labelled_hit"],
                            unit="ratio",
                            detail="1 when at least one retrieved chunk matches gold doc_type/source.",
                        ),
                        EvalMetric(
                            name="precision_at_k",
                            label=f"Precision@{top_k}",
                            value=labelled["precision_at_k"],
                            unit="ratio",
                        ),
                        EvalMetric(
                            name="recall_at_k",
                            label=f"Recall@{top_k}",
                            value=labelled["recall_at_k"],
                            unit="ratio",
                            detail=f"{int(labelled['retrieved_relevant'])} / {int(labelled['corpus_relevant'])} gold-relevant chunks.",
                        ),
                        EvalMetric(
                            name="ndcg_at_k",
                            label=f"nDCG@{top_k}",
                            value=labelled["ndcg_at_k"],
                            unit="ratio",
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
                    notes=notes,
                )
            )

        aggregate = [
            EvalMetric(name="corpus_points", label="Corpus size", value=corpus_size, unit="chunks"),
            EvalMetric(
                name="mean_precision_at_k",
                label=f"Mean precision@{top_k}",
                value=_mean(precisions),
                unit="ratio",
                detail="Gold is source/doc_type patterns, not passage-level relevance.",
            ),
            EvalMetric(
                name="mean_recall_at_k",
                label=f"Mean recall@{top_k}",
                value=_mean(recalls),
                unit="ratio",
            ),
            EvalMetric(
                name="mean_ndcg_at_k",
                label=f"Mean nDCG@{top_k}",
                value=_mean(ndcgs),
                unit="ratio",
            ),
            EvalMetric(
                name="labelled_hit_rate",
                label="Labelled hit rate",
                value=_mean(labelled_hits),
                unit="ratio",
            ),
            EvalMetric(
                name="failure_case_mean_recall",
                label="Failure-case mean recall",
                value=_mean(failure_recalls) if failure_recalls else None,
                unit="ratio",
                detail="Mega-cap earnings / off-corpus queries. Low recall is expected on a Fed-heavy index.",
            ),
            EvalMetric(
                name="retrieval_hit_rate",
                label="Operational hit rate (cosine)",
                value=_mean(hit_flags),
                unit="ratio",
                detail=(
                    "Share of queries returning at least one chunk above "
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
            EvalMetric(
                name="gold_queries",
                label="Gold queries",
                value=len(gold),
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
