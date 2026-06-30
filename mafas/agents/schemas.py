"""Pydantic models for the Analyst Agent's macro briefing output."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """A single retrieved evidence chunk referenced by the briefing."""

    index: int = Field(..., description="1-based citation marker used in the text")
    source: str
    doc_type: str
    date: str
    score: float = Field(..., description="Retrieval similarity (cosine, 0-1)")
    excerpt: str = Field(..., description="Short snippet of the cited chunk")


class KeyPoint(BaseModel):
    """A single synthesised finding with supporting citations."""

    point: str
    citations: list[int] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class MacroBriefing(BaseModel):
    """A sourced macroeconomic briefing produced by the Analyst Agent."""

    query: str
    summary: str
    key_points: list[KeyPoint] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    citations: list[SourceCitation] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    llm_self_confidence: float | None = None
    model: str = "mistral"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self) -> str:
        """Format the briefing as a human-readable text report."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"MACRO BRIEFING — {self.query}")
        lines.append("=" * 70)
        lines.append(
            f"Overall confidence: {self.confidence:.0%}   "
            f"(model: {self.model})"
        )
        if self.confidence_breakdown:
            parts = ", ".join(
                f"{k}={v:.2f}" for k, v in self.confidence_breakdown.items()
            )
            lines.append(f"  breakdown: {parts}")
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(self.summary)
        lines.append("")

        if self.key_points:
            lines.append("KEY POINTS")
            lines.append("-" * 70)
            for kp in self.key_points:
                cites = "".join(f"[{c}]" for c in kp.citations)
                lines.append(f"• ({kp.confidence:.0%}) {kp.point} {cites}")
            lines.append("")

        if self.risks:
            lines.append("RISKS / CAVEATS")
            lines.append("-" * 70)
            for risk in self.risks:
                lines.append(f"• {risk}")
            lines.append("")

        if self.citations:
            lines.append("SOURCES")
            lines.append("-" * 70)
            for c in self.citations:
                lines.append(
                    f"[{c.index}] ({c.doc_type}, {c.date}, sim={c.score:.2f}) "
                    f"{c.source}"
                )
        return "\n".join(lines)
