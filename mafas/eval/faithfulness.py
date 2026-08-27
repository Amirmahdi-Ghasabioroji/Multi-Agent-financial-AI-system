"""Rule-based (and optional LLM-as-judge) citation faithfulness."""

from __future__ import annotations

import re
from typing import Any

from agents.schemas import MacroBriefing

_CITE_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "its", "their", "they", "will", "would",
    "could", "should", "into", "over", "under", "about", "than", "then", "also",
    "only", "more", "most", "such", "been", "being",
}
_OPPOSITES = (
    ("increase", "decrease"),
    ("higher", "lower"),
    ("hawkish", "dovish"),
    ("tightening", "easing"),
    ("hike", "cut"),
    ("bullish", "bearish"),
)

JUDGE_SYSTEM = (
    "You are a strict financial fact checker. Decide whether the CLAIM is "
    "supported by the CHUNK. Do not use outside knowledge. Respond with JSON "
    "only: {\"supported\": true|false, \"score\": 0.0-1.0}."
)


def citation_indices(text: str) -> list[int]:
    return [int(m) for m in _CITE_RE.findall(text or "")]


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _chunk_text(briefing: MacroBriefing, index: int, hits: list[dict] | None) -> str:
    if hits and 1 <= index <= len(hits):
        return str(hits[index - 1].get("text") or "")
    for cite in briefing.citations:
        if cite.index == index:
            return cite.excerpt or ""
    return ""


def _contradicts(claim: str, chunk: str) -> bool:
    cl, ch = claim.lower(), chunk.lower()
    for a, b in _OPPOSITES:
        if a in cl and b in ch and a not in ch:
            return True
        if b in cl and a in ch and b not in ch:
            return True
    return False


def evaluate_faithfulness(
    briefing: MacroBriefing,
    hits: list[dict] | None = None,
) -> dict[str, float]:
    """Rule-based citation validity, overlap, and contradiction rate."""
    n_sources = len(briefing.citations)
    claims: list[tuple[str, list[int]]] = []
    if briefing.summary:
        cites = citation_indices(briefing.summary)
        claims.append((briefing.summary, cites))
    for kp in briefing.key_points:
        cites = list(kp.citations) or citation_indices(kp.point)
        claims.append((kp.point, cites))

    total_cites = 0
    valid_cites = 0
    overlaps: list[float] = []
    contradictions = 0
    unsupported = 0
    grounded_claims = 0

    for text, cites in claims:
        if not cites:
            unsupported += 1
            continue
        claim_ok = False
        for idx in cites:
            total_cites += 1
            if not (1 <= idx <= n_sources):
                continue
            valid_cites += 1
            chunk = _chunk_text(briefing, idx, hits)
            ov = jaccard(tokenize(text), tokenize(chunk))
            overlaps.append(ov)
            if _contradicts(text, chunk):
                contradictions += 1
            elif ov >= 0.08:
                claim_ok = True
        if claim_ok:
            grounded_claims += 1
        else:
            unsupported += 1

    n_claims = max(len(claims), 1)
    return {
        "citation_validity": round(valid_cites / total_cites, 4) if total_cites else 0.0,
        "mean_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
        "contradiction_rate": round(contradictions / total_cites, 4) if total_cites else 0.0,
        "groundedness": round(grounded_claims / n_claims, 4),
        "unsupported_rate": round(unsupported / n_claims, 4),
        "n_claims": float(len(claims)),
        "n_citations": float(total_cites),
    }


def judge_claim(llm: Any, claim: str, chunk: str) -> dict[str, Any] | None:
    """Optional LLM-as-judge. Returns None on failure."""
    if llm is None or not getattr(llm, "is_available", lambda: False)():
        return None
    try:
        parsed = llm.chat_json(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"CLAIM:\n{claim}\n\nCHUNK:\n{chunk[:1800]}",
                },
            ]
        )
    except Exception:  # noqa: BLE001
        return None
    supported = parsed.get("supported")
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if isinstance(supported, str):
        supported = supported.lower() in {"true", "yes", "1"}
    return {"supported": bool(supported), "score": max(0.0, min(1.0, score))}


def judge_briefing(
    llm: Any,
    briefing: MacroBriefing,
    hits: list[dict] | None = None,
    max_claims: int = 8,
) -> dict[str, float] | None:
    """Score key points with an LLM judge; skip if the model is down."""
    if llm is None or not getattr(llm, "is_available", lambda: False)():
        return None
    scores: list[float] = []
    supported_flags: list[float] = []
    for kp in briefing.key_points[:max_claims]:
        cites = list(kp.citations) or citation_indices(kp.point)
        if not cites:
            scores.append(0.0)
            supported_flags.append(0.0)
            continue
        chunk = _chunk_text(briefing, cites[0], hits)
        judged = judge_claim(llm, kp.point, chunk)
        if judged is None:
            continue
        scores.append(float(judged["score"]))
        supported_flags.append(1.0 if judged["supported"] else 0.0)
    if not scores:
        return None
    return {
        "judge_mean_score": round(sum(scores) / len(scores), 4),
        "judge_support_rate": round(sum(supported_flags) / len(supported_flags), 4),
        "judge_n_claims": float(len(scores)),
    }
