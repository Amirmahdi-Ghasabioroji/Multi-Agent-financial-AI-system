"""Labelled IR metrics for coarse source / doc_type gold."""

from __future__ import annotations

import math
from typing import Any


def grade_hit(hit: dict[str, Any], spec: dict[str, Any]) -> int:
    """Return 0 / 1 / 2. 2 = doc_type + source pattern, 1 = doc_type only."""
    doc_type = str(hit.get("doc_type") or "")
    source = str(hit.get("source") or "").lower()
    allowed = list(spec.get("doc_types") or [])
    if allowed and doc_type not in allowed:
        return 0
    patterns = [str(p).lower() for p in (spec.get("source_includes") or []) if p]
    if patterns:
        if any(p in source for p in patterns):
            return 2
        return 1 if (not allowed or doc_type in allowed) else 0
    return 1 if (not allowed or doc_type in allowed) else 0


def dcg(grades: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def precision_at_k(grades: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = grades[:k]
    if not top:
        return 0.0
    return sum(1 for g in top if g > 0) / k


def recall_at_k(retrieved_relevant: int, corpus_relevant: int) -> float:
    if corpus_relevant <= 0:
        return 0.0
    return retrieved_relevant / corpus_relevant


def ndcg_at_k(retrieved_grades: list[int], corpus_grades: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    actual = dcg([float(g) for g in retrieved_grades[:k]])
    ideal = dcg([float(g) for g in sorted(corpus_grades, reverse=True)[:k]])
    if ideal <= 0:
        return 0.0
    return actual / ideal


def labelled_query_metrics(
    hits: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    spec: dict[str, Any],
    k: int,
) -> dict[str, float]:
    retrieved_grades = [grade_hit(h, spec) for h in hits[:k]]
    corpus_grades = [grade_hit(p, spec) for p in corpus]
    relevant_grades = [g for g in corpus_grades if g > 0]
    retrieved_relevant = sum(1 for g in retrieved_grades if g > 0)
    return {
        "precision_at_k": round(precision_at_k(retrieved_grades, k), 4),
        "recall_at_k": round(recall_at_k(retrieved_relevant, len(relevant_grades)), 4),
        "ndcg_at_k": round(ndcg_at_k(retrieved_grades, relevant_grades, k), 4),
        "labelled_hit": 1.0 if retrieved_relevant else 0.0,
        "corpus_relevant": float(len(relevant_grades)),
        "retrieved_relevant": float(retrieved_relevant),
    }
