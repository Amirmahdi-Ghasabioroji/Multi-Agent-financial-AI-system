"""MAFAS agents package."""

from agents.analyst import AnalystAgent, build_analyst_agent
from agents.llm import OllamaClient, OllamaError
from agents.risk import DEFAULT_WATCHLIST, RiskAgent, build_risk_agent
from agents.risk_schemas import (
    AssetVolMetrics,
    ConcentrationRisk,
    CorrelationWarning,
    PositionSizingConstraint,
    RiskSummary,
)
from agents.schemas import KeyPoint, MacroBriefing, SourceCitation

__all__ = [
    "AnalystAgent",
    "build_analyst_agent",
    "RiskAgent",
    "build_risk_agent",
    "DEFAULT_WATCHLIST",
    "RiskSummary",
    "AssetVolMetrics",
    "CorrelationWarning",
    "ConcentrationRisk",
    "PositionSizingConstraint",
    "OllamaClient",
    "OllamaError",
    "MacroBriefing",
    "KeyPoint",
    "SourceCitation",
]
