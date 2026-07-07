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
from agents.strategy import StrategyAgent, build_strategy_agent
from agents.strategy_playbooks import PLAYBOOKS, Playbook, rank_playbooks
from agents.strategy_schemas import (
    MacroBias,
    PlaybookScore,
    StrategyReport,
    StrategySetup,
)

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
    "StrategyAgent",
    "build_strategy_agent",
    "StrategyReport",
    "StrategySetup",
    "MacroBias",
    "PlaybookScore",
    "PLAYBOOKS",
    "Playbook",
    "rank_playbooks",
    "OllamaClient",
    "OllamaError",
    "MacroBriefing",
    "KeyPoint",
    "SourceCitation",
]
