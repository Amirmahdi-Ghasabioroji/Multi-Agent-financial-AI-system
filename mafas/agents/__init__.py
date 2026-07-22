"""MAFAS agents package.

Heavy agent modules (analyst, orchestrator, etc.) are loaded lazily so lightweight
imports — e.g. ``agents.execution_schemas`` or ``agents.simulation`` — do not
pull sentence-transformers / scipy during unit tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    "ExecutionAgent",
    "build_execution_agent",
    "TradeCard",
    "TradeLevels",
    "SimulationStats",
    "SizingInfo",
    "RiskPipeline",
    "build_pipeline",
    "PipelineResult",
    "PipelineState",
    "OllamaClient",
    "OllamaError",
    "MacroBriefing",
    "KeyPoint",
    "SourceCitation",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalystAgent": ("agents.analyst", "AnalystAgent"),
    "build_analyst_agent": ("agents.analyst", "build_analyst_agent"),
    "RiskAgent": ("agents.risk", "RiskAgent"),
    "build_risk_agent": ("agents.risk", "build_risk_agent"),
    "DEFAULT_WATCHLIST": ("agents.risk", "DEFAULT_WATCHLIST"),
    "RiskSummary": ("agents.risk_schemas", "RiskSummary"),
    "AssetVolMetrics": ("agents.risk_schemas", "AssetVolMetrics"),
    "CorrelationWarning": ("agents.risk_schemas", "CorrelationWarning"),
    "ConcentrationRisk": ("agents.risk_schemas", "ConcentrationRisk"),
    "PositionSizingConstraint": ("agents.risk_schemas", "PositionSizingConstraint"),
    "StrategyAgent": ("agents.strategy", "StrategyAgent"),
    "build_strategy_agent": ("agents.strategy", "build_strategy_agent"),
    "StrategyReport": ("agents.strategy_schemas", "StrategyReport"),
    "StrategySetup": ("agents.strategy_schemas", "StrategySetup"),
    "MacroBias": ("agents.strategy_schemas", "MacroBias"),
    "PlaybookScore": ("agents.strategy_schemas", "PlaybookScore"),
    "PLAYBOOKS": ("agents.strategy_playbooks", "PLAYBOOKS"),
    "Playbook": ("agents.strategy_playbooks", "Playbook"),
    "rank_playbooks": ("agents.strategy_playbooks", "rank_playbooks"),
    "ExecutionAgent": ("agents.execution", "ExecutionAgent"),
    "build_execution_agent": ("agents.execution", "build_execution_agent"),
    "TradeCard": ("agents.execution_schemas", "TradeCard"),
    "TradeLevels": ("agents.execution_schemas", "TradeLevels"),
    "SimulationStats": ("agents.execution_schemas", "SimulationStats"),
    "SizingInfo": ("agents.execution_schemas", "SizingInfo"),
    "RiskPipeline": ("agents.orchestrator", "RiskPipeline"),
    "build_pipeline": ("agents.orchestrator", "build_pipeline"),
    "PipelineResult": ("agents.pipeline_schemas", "PipelineResult"),
    "PipelineState": ("agents.pipeline_schemas", "PipelineState"),
    "OllamaClient": ("agents.llm", "OllamaClient"),
    "OllamaError": ("agents.llm", "OllamaError"),
    "MacroBriefing": ("agents.schemas", "MacroBriefing"),
    "KeyPoint": ("agents.schemas", "KeyPoint"),
    "SourceCitation": ("agents.schemas", "SourceCitation"),
}

if TYPE_CHECKING:
    from agents.analyst import AnalystAgent, build_analyst_agent
    from agents.execution import ExecutionAgent, build_execution_agent
    from agents.execution_schemas import SimulationStats, SizingInfo, TradeCard, TradeLevels
    from agents.llm import OllamaClient, OllamaError
    from agents.orchestrator import RiskPipeline, build_pipeline
    from agents.pipeline_schemas import PipelineResult, PipelineState
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
    from agents.strategy_schemas import MacroBias, PlaybookScore, StrategyReport, StrategySetup


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
