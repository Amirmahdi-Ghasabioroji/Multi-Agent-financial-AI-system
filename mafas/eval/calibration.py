"""Brier score and reliability bins for probability forecasts."""

from __future__ import annotations


def brier_score(probs: list[float], outcomes: list[int]) -> float | None:
    if not probs or len(probs) != len(outcomes):
        return None
    total = sum((p - y) ** 2 for p, y in zip(probs, outcomes))
    return round(total / len(probs), 4)


def reliability_bins(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Equal-width bins on [0, 1]. Each row is predicted vs observed frequency."""
    if n_bins <= 0 or not probs or len(probs) != len(outcomes):
        return []
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        idx = min(n_bins - 1, max(0, int(p * n_bins)))
        buckets[idx].append((p, y))
    rows: list[dict[str, float]] = []
    for i, items in enumerate(buckets):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if not items:
            rows.append(
                {
                    "bin_left": round(lo, 4),
                    "bin_right": round(hi, 4),
                    "count": 0.0,
                    "mean_predicted": round((lo + hi) / 2, 4),
                    "mean_observed": 0.0,
                }
            )
            continue
        mean_p = sum(p for p, _ in items) / len(items)
        mean_y = sum(y for _, y in items) / len(items)
        rows.append(
            {
                "bin_left": round(lo, 4),
                "bin_right": round(hi, 4),
                "count": float(len(items)),
                "mean_predicted": round(mean_p, 4),
                "mean_observed": round(mean_y, 4),
            }
        )
    return rows
