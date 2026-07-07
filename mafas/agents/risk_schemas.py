"""Pydantic models for the Risk Agent's risk-environment assessment.

These mirror the structured-output style of the Analyst's MacroBriefing so the
downstream Strategy Agent can consume both agents' outputs uniformly.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AssetVolMetrics(BaseModel):
    """Per-instrument volatility snapshot."""

    ticker: str
    last_price: float = 0.0
    atr: float = Field(0.0, description="Wilder ATR in price units")
    atr_pct: float = Field(0.0, description="ATR as a fraction of last price")
    realised_vol: float = Field(0.0, description="Annualised realised volatility")
    regime: str = Field("medium", description="low | medium | high")


class CorrelationWarning(BaseModel):
    """A flagged pair of highly correlated instruments."""

    pair: list[str] = Field(..., description="[ticker_a, ticker_b]")
    correlation: float
    note: str = ""


class PositionSizingConstraint(BaseModel):
    """Advisory position-size cap for a single instrument (not a signal)."""

    ticker: str
    max_position_pct: float = Field(..., description="Cap as fraction of portfolio")
    risk_per_trade_pct: float = Field(..., description="Suggested risk per trade")
    rationale: str = ""


class ConcentrationRisk(BaseModel):
    """Basket-level diversification / concentration diagnostics."""

    mean_pairwise_correlation: float = 0.0
    effective_number_of_bets: float = 0.0
    n_assets: int = 0
    flagged: bool = False
    note: str = ""


class RiskSummary(BaseModel):
    """The Risk Agent's structured assessment of the current risk environment."""

    universe: list[str] = Field(default_factory=list)
    as_of: str = ""
    lookback_days: int = 0

    vol_regime: str = Field("medium", description="Overall market regime: low|medium|high")
    vix_level: float = 0.0
    mean_realised_vol: float = 0.0

    per_asset: list[AssetVolMetrics] = Field(default_factory=list)
    correlation_warnings: list[CorrelationWarning] = Field(default_factory=list)
    concentration: ConcentrationRisk = Field(default_factory=ConcentrationRisk)
    position_sizing: list[PositionSizingConstraint] = Field(default_factory=list)

    # Context inherited from the upstream Analyst Agent.
    analyst_query: str | None = None
    analyst_confidence: float | None = None
    macro_context: str = ""

    # LLM-authored interpretation of the quantitative picture.
    narrative: str = ""
    watch_items: list[str] = Field(default_factory=list)

    model: str = "mistral"
    llm_used: bool = False
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self) -> str:
        """Format the risk summary as a human-readable text report."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"RISK ENVIRONMENT — {', '.join(self.universe)}")
        lines.append("=" * 70)
        lines.append(
            f"Vol regime: {self.vol_regime.upper()}   "
            f"VIX: {self.vix_level:.1f}   "
            f"mean realised vol: {self.mean_realised_vol:.0%}   "
            f"(as of {self.as_of}, {self.lookback_days}d lookback)"
        )
        if self.analyst_query is not None:
            conf = (
                f"{self.analyst_confidence:.0%}"
                if self.analyst_confidence is not None
                else "n/a"
            )
            lines.append(f"Analyst context: '{self.analyst_query}' (confidence {conf})")
        lines.append("")

        lines.append("PER-ASSET VOLATILITY")
        lines.append("-" * 70)
        for a in self.per_asset:
            lines.append(
                f"• {a.ticker:<6} regime={a.regime:<6} "
                f"realised_vol={a.realised_vol:6.1%}  "
                f"ATR={a.atr_pct:5.2%}  last={a.last_price:.2f}"
            )
        lines.append("")

        lines.append("CONCENTRATION")
        lines.append("-" * 70)
        c = self.concentration
        lines.append(
            f"mean pairwise corr={c.mean_pairwise_correlation:.2f}  "
            f"effective bets={c.effective_number_of_bets:.2f} of {c.n_assets}"
            + ("  [FLAGGED]" if c.flagged else "")
        )
        if c.note:
            lines.append(f"  {c.note}")
        lines.append("")

        if self.correlation_warnings:
            lines.append("CORRELATION WARNINGS")
            lines.append("-" * 70)
            for w in self.correlation_warnings:
                lines.append(
                    f"• {w.pair[0]}–{w.pair[1]}: {w.correlation:+.2f}  {w.note}"
                )
            lines.append("")

        lines.append("SUGGESTED POSITION-SIZING CONSTRAINTS")
        lines.append("-" * 70)
        for p in self.position_sizing:
            lines.append(
                f"• {p.ticker:<6} max_size={p.max_position_pct:5.1%}  "
                f"risk/trade={p.risk_per_trade_pct:4.1%}"
            )
        lines.append("")

        if self.narrative:
            lines.append("INTERPRETATION")
            lines.append("-" * 70)
            lines.append(self.narrative)
            lines.append("")

        if self.watch_items:
            lines.append("WATCH ITEMS")
            lines.append("-" * 70)
            for item in self.watch_items:
                lines.append(f"• {item}")

        return "\n".join(lines)
