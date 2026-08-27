"""Typed API contracts for jobs, conversations, and existing MAFAS agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JobKind = Literal[
    "pipeline",
    "analyst",
    "risk",
    "strategy",
    "execution",
    "corpus_refresh",
    "corpus_reset",
    "demo",
    "evaluation",
]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class StrictRequest(BaseModel):
    """Base request that rejects accidental or misspelled fields."""

    model_config = ConfigDict(extra="forbid")


class SourceCitationDTO(BaseModel):
    """Analyst evidence citation."""

    index: int
    source: str
    doc_type: str
    date: str
    score: float
    excerpt: str


class KeyPointDTO(BaseModel):
    """Confidence-scored analyst finding."""

    point: str
    citations: list[int] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class MacroBriefingDTO(BaseModel):
    """API representation of ``agents.schemas.MacroBriefing``."""

    query: str
    summary: str
    key_points: list[KeyPointDTO] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    citations: list[SourceCitationDTO] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    llm_self_confidence: float | None = None
    model: str = "mistral"
    generated_at: str = ""


class AssetVolMetricsDTO(BaseModel):
    """Per-instrument volatility metrics."""

    ticker: str
    last_price: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    realised_vol: float = 0.0
    regime: str = "medium"


class CorrelationWarningDTO(BaseModel):
    """Highly correlated instrument pair."""

    pair: list[str]
    correlation: float
    note: str = ""


class PositionSizingConstraintDTO(BaseModel):
    """Risk Agent position-size cap."""

    ticker: str
    max_position_pct: float
    risk_per_trade_pct: float
    rationale: str = ""


class ConcentrationRiskDTO(BaseModel):
    """Basket concentration diagnostics."""

    mean_pairwise_correlation: float = 0.0
    effective_number_of_bets: float = 0.0
    n_assets: int = 0
    flagged: bool = False
    note: str = ""


class RiskSummaryDTO(BaseModel):
    """API representation of ``agents.risk_schemas.RiskSummary``."""

    universe: list[str] = Field(default_factory=list)
    as_of: str = ""
    lookback_days: int = 0
    vol_regime: str = "medium"
    vix_level: float = 0.0
    mean_realised_vol: float = 0.0
    per_asset: list[AssetVolMetricsDTO] = Field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    correlation_warnings: list[CorrelationWarningDTO] = Field(default_factory=list)
    concentration: ConcentrationRiskDTO = Field(default_factory=ConcentrationRiskDTO)
    position_sizing: list[PositionSizingConstraintDTO] = Field(default_factory=list)
    analyst_query: str | None = None
    analyst_confidence: float | None = None
    macro_context: str = ""
    narrative: str = ""
    watch_items: list[str] = Field(default_factory=list)
    model: str = "mistral"
    llm_used: bool = False
    generated_at: str = ""


class MacroBiasDTO(BaseModel):
    """Directional macro read used by Strategy."""

    direction: str = "neutral"
    strength: float = Field(0.5, ge=0.0, le=1.0)
    rationale: str = ""
    source: str = "fallback"


class PlaybookScoreDTO(BaseModel):
    """Strategy playbook suitability score."""

    key: str
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class StrategySetupDTO(BaseModel):
    """API representation of ``agents.strategy_schemas.StrategySetup``."""

    strategy: str
    strategy_name: str = ""
    instrument: str | None = None
    direction: Literal["long", "short", "neutral"] = "long"
    rationale: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    playbook_fit: float = Field(0.0, ge=0.0, le=1.0)
    horizon: Literal["intraday", "swing", "position"] = "swing"
    risk_note: str = ""


class StrategyReportDTO(BaseModel):
    """API representation of ``agents.strategy_schemas.StrategyReport``."""

    macro_bias: MacroBiasDTO = Field(default_factory=MacroBiasDTO)
    vol_regime: str = "medium"
    universe: list[str] = Field(default_factory=list)
    candidate_scores: list[PlaybookScoreDTO] = Field(default_factory=list)
    setups: list[StrategySetupDTO] = Field(default_factory=list)
    suppressed: list[str] = Field(default_factory=list)
    narrative: str = ""
    analyst_query: str | None = None
    analyst_confidence: float | None = None
    model: str = "mistral"
    llm_used: bool = False
    generated_at: str = ""


class PipelineRunRequest(StrictRequest):
    """Input for a complete Analyst → Risk → Strategy → Execution run."""

    query: str = Field(..., min_length=3, max_length=2000)
    tickers: list[str] = Field(default_factory=list, max_length=50)
    lookback_days: int = Field(252, ge=20, le=2500)
    use_llm: bool = True
    demo_mode: bool = False
    conversation_id: str | None = None
    context: str | list[dict[str, Any]] | dict[str, Any] | None = None

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip().upper() for item in value if item.strip()))


class AnalystRunRequest(StrictRequest):
    """Input for the existing Analyst Agent."""

    query: str = Field(..., min_length=3, max_length=2000)
    doc_type: str | None = None
    date_after: str | None = None
    demo_mode: bool = False
    conversation_id: str | None = None
    context: str | list[dict[str, Any]] | dict[str, Any] | None = None


class RiskRunRequest(StrictRequest):
    """Input for the existing Risk Agent."""

    tickers: list[str] = Field(default_factory=list, max_length=50)
    lookback_days: int = Field(252, ge=20, le=2500)
    use_llm: bool = True
    briefing: MacroBriefingDTO | None = None
    demo_mode: bool = False
    conversation_id: str | None = None

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip().upper() for item in value if item.strip()))


class StrategyRunRequest(StrictRequest):
    """Input for the existing Strategy Agent."""

    risk: RiskSummaryDTO
    briefing: MacroBriefingDTO | None = None
    use_llm: bool = True
    demo_mode: bool = False
    conversation_id: str | None = None


class ExecutionRunRequest(StrictRequest):
    """Input for the existing Execution Agent."""

    setups: list[StrategySetupDTO] = Field(..., min_length=1, max_length=25)
    risk: RiskSummaryDTO | None = None
    use_llm: bool = True
    demo_mode: bool = False
    conversation_id: str | None = None


class CorpusRefreshRequest(StrictRequest):
    """Corpus ingestion controls."""

    n_fomc: int = Field(5, ge=0, le=50)
    max_news_per_feed: int = Field(10, ge=0, le=100)
    tickers: list[str] | None = None
    filings_per_ticker: int = Field(1, ge=0, le=10)


class CorpusResetRequest(CorpusRefreshRequest):
    """Destructive corpus reset requiring an exact confirmation phrase."""

    confirmation: Literal["RESET FINANCIAL DOCS"]


class DemoRunRequest(StrictRequest):
    """Optional custom labels for a deterministic dashboard demo."""

    query: str = "How is the rate outlook affecting mega-cap equities?"
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    conversation_id: str | None = None
    demo_mode: Literal[True] = True


class EvaluationRunRequest(StrictRequest):
    """On-demand evaluation. ``all`` is rag + simulation + risk; other suites are opt-in."""

    suites: list[Literal["rag", "simulation", "risk", "analyst", "gates", "strategy", "all"]] = Field(
        default_factory=lambda: ["all"]
    )
    top_k: int = Field(8, ge=1, le=50)
    lookback_days: int = Field(252, ge=20, le=2500)
    tickers: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip().upper() for item in value if item.strip()))


class JobSubmitRequest(StrictRequest):
    """Generic job submission contract."""

    kind: JobKind
    payload: dict[str, Any] = Field(default_factory=dict)


class JobAccepted(BaseModel):
    """Immediate response for accepted asynchronous work."""

    id: str
    status: JobStatus = "queued"


class JobEventRead(BaseModel):
    """Persisted job event."""

    sequence: int
    event_type: str
    message: str
    data: dict[str, Any]
    created_at: datetime


class JobRead(BaseModel):
    """Persisted job state and result."""

    id: str
    kind: JobKind
    status: JobStatus
    payload: dict[str, Any]
    metadata: dict[str, Any]
    result: Any = None
    error: str | None = None
    conversation_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ConversationCreate(StrictRequest):
    """Create a conversation."""

    title: str = Field("New conversation", min_length=1, max_length=255)


class ConversationUpdate(StrictRequest):
    """Rename a conversation."""

    title: str = Field(..., min_length=1, max_length=255)


class MessageCreate(StrictRequest):
    """Append a message to a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=100_000)
    job_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRead(BaseModel):
    """Persisted conversation message."""

    id: str
    conversation_id: str
    role: str
    content: str
    job_id: str | None
    metadata: dict[str, Any]
    created_at: datetime


class ConversationRead(BaseModel):
    """Conversation summary."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationRead):
    """Conversation with all messages."""

    messages: list[MessageRead] = Field(default_factory=list)


class ServiceCheck(BaseModel):
    """A single dependency health result."""

    status: Literal["ok", "unavailable", "not_configured"]
    detail: str | None = None
    count: int | None = None


class HealthResponse(BaseModel):
    """API and dependency readiness snapshot."""

    status: Literal["ok", "degraded"]
    services: dict[str, ServiceCheck]


class SystemSummary(BaseModel):
    """Dashboard-wide persistence and service summary."""

    conversations: int
    messages: int
    jobs: int
    active_jobs: int
    corpus_count: int | None = None
    health_status: Literal["ok", "degraded"]


class PlaybookDTO(BaseModel):
    """Public, deterministic strategy-playbook metadata."""

    key: str
    name: str
    description: str
    directional: bool
    tags: list[str]


class DefaultsResponse(BaseModel):
    """Default dashboard configuration."""

    watchlist: list[str]
    playbooks: list[PlaybookDTO]
    job_kinds: list[str]
