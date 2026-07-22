"""Rank simulated trade cards across strategy setups."""

from __future__ import annotations

from agents.execution_schemas import ExecutionComparison, ExecutionComparisonEntry, TradeCard


def _norm(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if hi == lo:
        return {i: 0.5 for i in range(len(values))}
    return {i: (v - lo) / (hi - lo) for i, v in enumerate(values)}


def rank_trade_cards(cards: list[TradeCard]) -> ExecutionComparison | None:
    """Rank cards that have backtest data; return None if none qualify."""
    eligible = [
        c for c in cards
        if c.simulated and c.backtest is not None and c.backtest.metrics.n_trades > 0
    ]
    if not eligible:
        return None

    sharpes = [
        c.backtest.metrics.sharpe_ratio if c.backtest and c.backtest.metrics.sharpe_ratio is not None else 0.0
        for c in eligible
    ]
    pnls = [c.backtest.metrics.total_pnl if c.backtest else 0.0 for c in eligible]
    drawdowns = [
        -(c.backtest.metrics.max_drawdown_pct if c.backtest else 0.0) for c in eligible
    ]
    forward_r = [c.stats.expected_r for c in eligible]

    n_sharpe = _norm(sharpes)
    n_pnl = _norm(pnls)
    n_dd = _norm(drawdowns)
    n_fwd = _norm(forward_r)

    scored: list[tuple[float, TradeCard]] = []
    for i, card in enumerate(eligible):
        composite = (
            0.35 * n_sharpe.get(i, 0.0)
            + 0.25 * n_pnl.get(i, 0.0)
            + 0.20 * n_dd.get(i, 0.0)
            + 0.20 * n_fwd.get(i, 0.0)
        )
        scored.append((composite, card))

    scored.sort(key=lambda t: t[0], reverse=True)

    ranked: list[ExecutionComparisonEntry] = []
    for rank, (score, card) in enumerate(scored, start=1):
        bt = card.backtest
        assert bt is not None
        ranked.append(
            ExecutionComparisonEntry(
                rank=rank,
                instrument=card.instrument or "—",
                strategy=card.strategy_name or card.strategy,
                composite_score=round(score, 4),
                sharpe_ratio=bt.metrics.sharpe_ratio,
                total_pnl=bt.metrics.total_pnl,
                max_drawdown_pct=bt.metrics.max_drawdown_pct,
                expected_r_forward=card.stats.expected_r,
            )
        )

    best_sharpe = max(eligible, key=lambda c: c.backtest.metrics.sharpe_ratio or -999)  # type: ignore[union-attr]
    best_pnl = max(eligible, key=lambda c: c.backtest.metrics.total_pnl)  # type: ignore[union-attr]
    lowest_dd = min(eligible, key=lambda c: c.backtest.metrics.max_drawdown_pct)  # type: ignore[union-attr]

    def _label(c: TradeCard) -> str:
        return f"{c.strategy_name or c.strategy} | {c.instrument or '—'}"

    return ExecutionComparison(
        ranked=ranked,
        best_sharpe=_label(best_sharpe),
        best_pnl=_label(best_pnl),
        lowest_drawdown=_label(lowest_dd),
    )
