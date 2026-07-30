"""Structured evaluation report models for dashboard and API export."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EvalSuiteName = Literal["rag", "simulation", "risk"]


class EvalMetric(BaseModel):
    """Single scored observation from an evaluation run."""

    name: str
    label: str
    value: float | int | str | None
    unit: str | None = None
    detail: str | None = None


class EvalCaseResult(BaseModel):
    """Per-scenario breakdown within a suite."""

    id: str
    label: str
    metrics: list[EvalMetric] = Field(default_factory=list)
    notes: str = ""


class SuiteResult(BaseModel):
    """Results for one evaluation suite."""

    suite: EvalSuiteName
    label: str
    status: Literal["completed", "failed"] = "completed"
    error: str | None = None
    duration_ms: float = 0.0
    metrics: list[EvalMetric] = Field(default_factory=list)
    cases: list[EvalCaseResult] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Top-level evaluation artefact stored on evaluation jobs."""

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    suites: list[SuiteResult] = Field(default_factory=list)
    summary_metrics: list[EvalMetric] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
