"""Composite confidence scoring for sourced analyst briefings.

The score combines four signals, each normalised to [0, 1]:

    retrieval   — mean semantic similarity of the supporting evidence
    diversity   — breadth of independent sources and document types
    recency     — how fresh the supporting evidence is
    llm_self    — the model's self-reported confidence

A higher score means the briefing rests on relevant, corroborated, fresh
evidence that the model is itself confident about.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from loguru import logger

WEIGHTS: dict[str, float] = {
    "retrieval": 0.35,
    "diversity": 0.25,
    "recency": 0.15,
    "llm_self": 0.25,
}

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %Y",
    "%b %Y",
    "%Y%m%d",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def retrieval_confidence(scores: list[float]) -> float:
    """Mean of retrieval similarities, clamped to [0, 1]."""
    if not scores:
        return 0.0
    return _clamp(sum(scores) / len(scores))


def source_diversity(sources: list[str], doc_types: list[str]) -> float:
    """Reward corroboration across distinct sources and document types."""
    if not sources:
        return 0.0
    n_sources = len(set(sources))
    n_types = len(set(dt for dt in doc_types if dt))
    source_component = _clamp(n_sources / 5.0)
    type_component = _clamp(n_types / 3.0)
    return _clamp(0.5 * source_component + 0.5 * type_component)


def _parse_date(value: str) -> datetime | None:
    if not value or value.lower() == "unknown":
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError, IndexError):
        return None


def recency_factor(dates: list[str], now: datetime | None = None) -> float:
    """Score based on the freshest parseable source date.

    <=30 days -> 1.0, decaying linearly to 0.3 at >=365 days. Unknown -> 0.5.
    """
    now = now or datetime.now(timezone.utc)
    parsed = [d for d in (_parse_date(x) for x in dates) if d is not None]
    if not parsed:
        return 0.5
    newest = max(parsed)
    age_days = (now - newest).days
    if age_days <= 30:
        return 1.0
    if age_days >= 365:
        return 0.3
    return _clamp(1.0 - 0.7 * (age_days - 30) / (365 - 30))


def composite_confidence(
    scores: list[float],
    sources: list[str],
    doc_types: list[str],
    dates: list[str],
    llm_self_confidence: float | None,
) -> tuple[float, dict[str, float]]:
    """Combine all signals into one score plus a per-component breakdown."""
    components = {
        "retrieval": retrieval_confidence(scores),
        "diversity": source_diversity(sources, doc_types),
        "recency": recency_factor(dates),
        "llm_self": _clamp(llm_self_confidence)
        if llm_self_confidence is not None
        else 0.5,
    }
    overall = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    logger.debug("Confidence components: {} -> {:.3f}", components, overall)
    return _clamp(overall), components
