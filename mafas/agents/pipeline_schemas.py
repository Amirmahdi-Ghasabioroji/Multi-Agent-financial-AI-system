"""State and result models for the LangGraph orchestration layer.

`PipelineState` is the mutable dict that flows between graph nodes. `PipelineResult`
is the immutable, structured artifact returned to the caller once the graph
terminates — it bundles every agent's output plus the decision and the route the
graph actually took (useful for debugging the conditional logic).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

from pydantic import BaseModel, Field

from agents.execution_schemas import TradeCard
from agents.risk_schemas import RiskSummary
from agents.schemas import MacroBriefing
from agents.strategy_schemas import StrategyReport


class PipelineState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes."""

    # Inputs
    query: str
    original_query: str
    tickers: list[str]
    use_llm: bool
    conversation_context: str

    # Per-stage outputs
    briefing: MacroBriefing | None
    risk: RiskSummary | None
    strategy: StrategyReport | None
    cards: list[TradeCard]

    # Control / bookkeeping
    analyst_attempts: int
    decision: str  # "trade" | "no_trade" | ""
    no_trade_reason: str
    route_log: list[str]
    errors: list[str]


class PipelineResult(BaseModel):
    """Aggregated final output of the orchestrated pipeline."""

    query: str
    original_query: str
    tickers: list[str] = Field(default_factory=list)
    decision: str = "no_trade"
    no_trade_reason: str = ""

    briefing: MacroBriefing | None = None
    risk: RiskSummary | None = None
    strategy: StrategyReport | None = None
    cards: list[TradeCard] = Field(default_factory=list)

    analyst_attempts: int = 0
    route_log: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def tradeable_cards(self) -> list[TradeCard]:
        return [c for c in self.cards if c.simulated]

    def render(self) -> str:
        """Compact executive summary of the whole run."""
        lines: list[str] = []
        lines.append("#" * 72)
        lines.append(f"# MAFAS PIPELINE — {self.query}")
        lines.append("#" * 72)
        if self.original_query != self.query:
            lines.append(f"(original query broadened from: '{self.original_query}')")
        lines.append(f"Route: {' -> '.join(self.route_log)}")
        lines.append(f"Decision: {self.decision.upper().replace('_', ' ')}")
        if self.decision == "no_trade" and self.no_trade_reason:
            lines.append(f"Reason: {self.no_trade_reason}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)} (see .errors)")
        lines.append("")

        if self.briefing is not None:
            lines.append(f"ANALYST  — confidence {self.briefing.confidence:.0%}, "
                         f"{len(self.briefing.citations)} sources "
                         f"(attempts: {self.analyst_attempts})")
        if self.risk is not None:
            lines.append(f"RISK     — {self.risk.vol_regime.upper()} vol regime, "
                         f"VIX {self.risk.vix_level:.1f}, "
                         f"mean corr {self.risk.concentration.mean_pairwise_correlation:.2f}")
        if self.strategy is not None:
            b = self.strategy.macro_bias
            lines.append(f"STRATEGY — bias {b.direction.upper()} ({b.strength:.0%}), "
                         f"{len(self.strategy.setups)} setup(s)")
        if self.decision == "trade":
            lines.append("")
            lines.append("TRADE CARDS")
            lines.append("-" * 72)
            for c in self.cards:
                inst = c.instrument or "—"
                if c.simulated:
                    lines.append(
                        f"• {c.strategy_name} | {inst} {c.direction.upper()} | "
                        f"P(TP<SL)={c.stats.prob_tp_before_sl:.0%} | "
                        f"E[R]={c.stats.expected_r:+.2f} | E[P/L]=${c.expectancy_amount:+,.0f}"
                    )
                else:
                    lines.append(f"• {c.strategy_name} | {inst} — not simulated ({c.skip_reason[:40]})")
        return "\n".join(lines)
