"""Pydantic models for the Strategy Agent's output.

Structured like MacroBriefing / RiskSummary so the downstream Execution Agent
can consume strategy setups directly. Each StrategySetup is a *reasoned
suggestion*, not a trade signal — the Execution Agent stress-tests it against
historical data and may vary the signal/risk profile.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MacroBias(BaseModel):
    """Directional read of the macro environment used for conditional logic."""

    direction: str = Field("neutral", description="bullish | bearish | neutral")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Conviction 0-1")
    rationale: str = ""
    source: str = Field("fallback", description="'llm' or 'fallback'")


class PlaybookScore(BaseModel):
    """Deterministic suitability of a single playbook for the environment."""

    key: str
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class StrategySetup(BaseModel):
    """A reasoned strategy suggestion bound to an instrument and direction."""

    strategy: str = Field(..., description="Playbook key, e.g. 'trend_following'")
    strategy_name: str = ""
    instrument: str | None = Field(None, description="Ticker from the risk universe")
    direction: str = Field("long", description="long | short | neutral")
    rationale: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    playbook_fit: float = Field(0.0, ge=0.0, le=1.0, description="Deterministic score")
    horizon: str = Field("swing", description="intraday | swing | position")
    risk_note: str = Field("", description="Ties back to Risk Agent sizing constraints")


class StrategyReport(BaseModel):
    """The Strategy Agent's structured reasoning output."""

    macro_bias: MacroBias = Field(default_factory=MacroBias)
    vol_regime: str = "medium"
    universe: list[str] = Field(default_factory=list)

    candidate_scores: list[PlaybookScore] = Field(default_factory=list)
    setups: list[StrategySetup] = Field(default_factory=list)
    suppressed: list[str] = Field(
        default_factory=list, description="Playbooks explicitly ruled out + why"
    )
    narrative: str = ""

    # Context inherited from the upstream agents.
    analyst_query: str | None = None
    analyst_confidence: float | None = None

    model: str = "mistral"
    llm_used: bool = False
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self) -> str:
        """Format the strategy report as a human-readable text report."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"STRATEGY VIEW — {', '.join(self.universe)}")
        lines.append("=" * 70)
        b = self.macro_bias
        lines.append(
            f"Macro bias: {b.direction.upper()} (strength {b.strength:.0%}, "
            f"via {b.source})   |   Vol regime: {self.vol_regime.upper()}"
        )
        if self.analyst_query is not None:
            conf = (
                f"{self.analyst_confidence:.0%}"
                if self.analyst_confidence is not None
                else "n/a"
            )
            lines.append(f"Analyst context: '{self.analyst_query}' (confidence {conf})")
        if b.rationale:
            lines.append(f"Bias rationale: {b.rationale}")
        lines.append("")

        if self.candidate_scores:
            lines.append("PLAYBOOK SUITABILITY (deterministic)")
            lines.append("-" * 70)
            for c in self.candidate_scores:
                lines.append(f"• {c.name:<34} {c.score:.2f}   ({c.reason})")
            lines.append("")

        lines.append("SUGGESTED SETUPS")
        lines.append("-" * 70)
        if not self.setups:
            lines.append("(none — environment did not support a confident setup)")
        for i, s in enumerate(self.setups, start=1):
            inst = s.instrument or "—"
            lines.append(
                f"{i}. {s.strategy_name} | {inst} {s.direction.upper()} "
                f"| conf {s.confidence:.0%} | fit {s.playbook_fit:.2f} | {s.horizon}"
            )
            if s.rationale:
                lines.append(f"   rationale: {s.rationale}")
            if s.risk_note:
                lines.append(f"   risk: {s.risk_note}")
        lines.append("")

        if self.suppressed:
            lines.append("SUPPRESSED / NOT NOW")
            lines.append("-" * 70)
            for item in self.suppressed:
                lines.append(f"• {item}")
            lines.append("")

        if self.narrative:
            lines.append("REASONING")
            lines.append("-" * 70)
            lines.append(self.narrative)

        return "\n".join(lines)
