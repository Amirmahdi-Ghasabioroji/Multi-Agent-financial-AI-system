"""Lazy adapters from API job payloads to the synchronous MAFAS core."""

from __future__ import annotations

import inspect
import json
from typing import Any

from backend.app.config import ensure_core_import_path
from backend.app.jobs import EventEmitter, JobRunner, json_safe

DEMO_TIMESTAMP = "2026-01-15T12:00:00+00:00"


def _demo_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic, schema-shaped data for dashboard demonstrations."""
    query = str(
        payload.get("query")
        or "How is the rate outlook affecting mega-cap equities?"
    )
    tickers = list(payload.get("tickers") or ["AAPL", "MSFT", "NVDA"])
    briefing = {
        "query": query,
        "summary": (
            "Inflation is moderating while policy remains restrictive, supporting "
            "quality equities but leaving duration-sensitive assets exposed to rate repricing."
        ),
        "key_points": [
            {
                "point": "Disinflation keeps eventual easing in view.",
                "citations": [1],
                "confidence": 0.84,
            },
            {
                "point": "Mega-cap balance-sheet quality offers relative resilience.",
                "citations": [2],
                "confidence": 0.78,
            },
        ],
        "risks": ["A renewed inflation impulse could delay expected policy easing."],
        "citations": [
            {
                "index": 1,
                "source": "Federal Reserve demo fixture",
                "doc_type": "fomc_minutes",
                "date": "2026-01-01",
                "score": 0.91,
                "excerpt": "Participants noted continued progress toward price stability.",
            },
            {
                "index": 2,
                "source": "SEC filings demo fixture",
                "doc_type": "sec_filing",
                "date": "2025-12-31",
                "score": 0.86,
                "excerpt": "Liquidity and operating cash flow remained strong.",
            },
        ],
        "confidence": 0.82,
        "confidence_breakdown": {
            "retrieval_quality": 0.88,
            "source_diversity": 0.76,
            "llm_self_confidence": 0.82,
        },
        "llm_self_confidence": 0.82,
        "model": "demo-deterministic",
        "generated_at": DEMO_TIMESTAMP,
    }
    per_asset = [
        {
            "ticker": ticker,
            "last_price": round(180.0 + index * 55.0, 2),
            "atr": round(4.2 + index * 0.8, 2),
            "atr_pct": round(0.023 + index * 0.003, 3),
            "realised_vol": round(0.22 + index * 0.025, 3),
            "regime": "medium",
        }
        for index, ticker in enumerate(tickers)
    ]
    risk = {
        "universe": tickers,
        "as_of": "2026-01-15",
        "lookback_days": int(payload.get("lookback_days", 252)),
        "vol_regime": "medium",
        "vix_level": 18.4,
        "mean_realised_vol": 0.245,
        "per_asset": per_asset,
        "correlation_matrix": {
            ticker: {
                other: (1.0 if ticker == other else 0.58)
                for other in tickers
            }
            for ticker in tickers
        },
        "correlation_warnings": [],
        "concentration": {
            "mean_pairwise_correlation": 0.58,
            "effective_number_of_bets": 2.1,
            "n_assets": len(tickers),
            "flagged": False,
            "note": "Moderate common factor exposure; sizing remains important.",
        },
        "position_sizing": [
            {
                "ticker": ticker,
                "max_position_pct": 0.2,
                "risk_per_trade_pct": 0.01,
                "rationale": "Medium-vol regime demo constraint.",
            }
            for ticker in tickers
        ],
        "analyst_query": query,
        "analyst_confidence": 0.82,
        "macro_context": briefing["summary"],
        "narrative": (
            "Volatility is moderate and correlations are elevated but not extreme. "
            "Favor staged entries and maintain explicit per-position risk caps."
        ),
        "watch_items": ["VIX above 24", "Two-year yield breakout"],
        "model": "demo-deterministic",
        "llm_used": False,
        "generated_at": DEMO_TIMESTAMP,
    }
    primary = tickers[0] if tickers else "AAPL"
    setup = {
        "strategy": "trend_following",
        "strategy_name": "Trend Following",
        "instrument": primary,
        "direction": "long",
        "rationale": "Constructive macro bias and contained volatility support trend exposure.",
        "confidence": 0.74,
        "playbook_fit": 0.81,
        "horizon": "swing",
        "risk_note": "Cap risk at 1% of equity and reassess if VIX exceeds 24.",
    }
    strategy = {
        "macro_bias": {
            "direction": "bullish",
            "strength": 0.68,
            "rationale": "Disinflation and resilient cash flows support a measured risk-on bias.",
            "source": "fallback",
        },
        "vol_regime": "medium",
        "universe": tickers,
        "candidate_scores": [
            {
                "key": "trend_following",
                "name": "Trend Following",
                "score": 0.81,
                "reason": "Good fit for a directional, medium-vol environment.",
            }
        ],
        "setups": [setup],
        "suppressed": ["Carry: insufficient low-vol support in the current regime."],
        "narrative": "Use a liquid directional setup with disciplined risk sizing.",
        "analyst_query": query,
        "analyst_confidence": 0.82,
        "model": "demo-deterministic",
        "llm_used": False,
        "generated_at": DEMO_TIMESTAMP,
    }
    card = {
        "instrument": primary,
        "strategy": "trend_following",
        "strategy_name": "Trend Following",
        "direction": "long",
        "horizon": "swing",
        "simulated": True,
        "skip_reason": "",
        "levels": {
            "entry": 185.0,
            "stop_loss": 176.6,
            "take_profit": 201.8,
            "atr": 4.2,
            "risk_per_unit": 8.4,
            "planned_rr": 2.0,
        },
        "stats": {
            "n_sims": 5000,
            "horizon_bars": 20,
            "prob_tp_before_sl": 0.57,
            "prob_sl_before_tp": 0.31,
            "prob_timeout": 0.12,
            "expected_r": 0.42,
            "win_rate": 0.57,
            "avg_bars_to_exit": 9.4,
            "mae_mean_r": 0.48,
            "mae_p95_r": 0.94,
            "seed": 42,
        },
        "sizing": {
            "account_equity": 100000.0,
            "units": 119.0,
            "notional": 22015.0,
            "notional_pct": 0.22015,
            "risk_amount": 999.6,
            "risk_pct": 0.009996,
            "max_position_pct": 0.25,
            "capped": False,
        },
        "expectancy_amount": 419.83,
        "data_source": "demo",
        "bars_used": 504,
        "strategy_confidence": 0.74,
        "playbook_fit": 0.81,
        "verdict": "Positive simulated expectancy with risk contained by the proposed stop.",
        "llm_used": False,
        "model": "demo-deterministic",
        "generated_at": DEMO_TIMESTAMP,
    }
    return {
        "demo_mode": True,
        "query": query,
        "original_query": query,
        "tickers": tickers,
        "decision": "trade",
        "no_trade_reason": "",
        "briefing": briefing,
        "risk": risk,
        "strategy": strategy,
        "cards": [card],
        "analyst_attempts": 1,
        "route_log": ["analyst(attempt=1)", "risk", "strategy", "execution"],
        "errors": [],
        "generated_at": DEMO_TIMESTAMP,
    }


def _context_text(context: Any) -> str | None:
    if context is None:
        return None
    if isinstance(context, str):
        return context
    return json.dumps(context, ensure_ascii=False)


def _pipeline_runner(payload: dict[str, Any], emit: EventEmitter) -> Any:
    if payload.get("demo_mode"):
        emit("progress", "Loading deterministic pipeline fixture", {"stage": "demo"})
        return _demo_fixture(payload)

    ensure_core_import_path()
    from agents.orchestrator import build_pipeline

    emit("progress", "Building pipeline agents", {"stage": "initialise"})
    pipeline = build_pipeline(with_llm=bool(payload.get("use_llm", True)))
    pipeline.risk.lookback_days = int(payload.get("lookback_days", 252))
    run_kwargs: dict[str, Any] = {
        "query": payload["query"],
        "tickers": payload.get("tickers") or [],
        "use_llm": bool(payload.get("use_llm", True)),
    }
    signature = inspect.signature(pipeline.run)
    context = _context_text(payload.get("context"))
    if context is not None:
        if "conversation_context" in signature.parameters:
            run_kwargs["conversation_context"] = context
        elif "context" in signature.parameters:
            run_kwargs["context"] = context
    if "progress_callback" in signature.parameters:
        run_kwargs["progress_callback"] = lambda event, data: emit(
            event,
            f"{str(data.get('stage', 'pipeline')).replace('_', ' ').title()}: "
            f"{event.replace('_', ' ')}",
            data,
        )
    emit("progress", "Running synchronous agent pipeline", {"stage": "pipeline"})
    result = pipeline.run(**run_kwargs)
    emit("progress", "Pipeline returned a result", {"stage": "complete"})
    return json_safe(result)


def _analyst_runner(payload: dict[str, Any], emit: EventEmitter) -> Any:
    if payload.get("demo_mode"):
        emit("progress", "Loading deterministic analyst fixture", {"stage": "demo"})
        return _demo_fixture(payload)["briefing"] | {"demo_mode": True}

    ensure_core_import_path()
    from agents.analyst import build_analyst_agent

    emit("progress", "Building Analyst Agent", {"stage": "initialise"})
    agent = build_analyst_agent()
    emit("progress", "Retrieving evidence and preparing briefing", {"stage": "analyst"})
    result = agent.brief(
        query=payload["query"],
        doc_type=payload.get("doc_type"),
        date_after=payload.get("date_after"),
        conversation_context=_context_text(payload.get("context")),
    )
    return json_safe(result)


def _risk_runner(payload: dict[str, Any], emit: EventEmitter) -> Any:
    if payload.get("demo_mode"):
        emit("progress", "Loading deterministic risk fixture", {"stage": "demo"})
        return _demo_fixture(payload)["risk"] | {"demo_mode": True}

    ensure_core_import_path()
    from agents.risk import build_risk_agent
    from agents.schemas import MacroBriefing

    emit("progress", "Building Risk Agent", {"stage": "initialise"})
    agent = build_risk_agent(with_llm=bool(payload.get("use_llm", True)))
    agent.lookback_days = int(payload.get("lookback_days", 252))
    briefing_data = payload.get("briefing")
    briefing = MacroBriefing.model_validate(briefing_data) if briefing_data else None
    emit("progress", "Calculating risk metrics", {"stage": "risk"})
    return json_safe(agent.assess(payload.get("tickers") or [], briefing=briefing))


def _strategy_runner(payload: dict[str, Any], emit: EventEmitter) -> Any:
    if payload.get("demo_mode"):
        emit("progress", "Loading deterministic strategy fixture", {"stage": "demo"})
        return _demo_fixture(payload)["strategy"] | {"demo_mode": True}

    ensure_core_import_path()
    from agents.risk_schemas import RiskSummary
    from agents.schemas import MacroBriefing
    from agents.strategy import build_strategy_agent

    emit("progress", "Building Strategy Agent", {"stage": "initialise"})
    agent = build_strategy_agent(with_llm=bool(payload.get("use_llm", True)))
    risk = RiskSummary.model_validate(payload["risk"])
    briefing_data = payload.get("briefing")
    briefing = MacroBriefing.model_validate(briefing_data) if briefing_data else None
    emit("progress", "Ranking strategy playbooks", {"stage": "strategy"})
    return json_safe(agent.decide(risk, briefing=briefing))


def _execution_runner(payload: dict[str, Any], emit: EventEmitter) -> Any:
    if payload.get("demo_mode"):
        emit("progress", "Loading deterministic execution fixture", {"stage": "demo"})
        return {"demo_mode": True, "cards": _demo_fixture(payload)["cards"]}

    ensure_core_import_path()
    from agents.execution import build_execution_agent
    from agents.risk_schemas import RiskSummary
    from agents.strategy_schemas import StrategySetup

    emit("progress", "Building Execution Agent", {"stage": "initialise"})
    agent = build_execution_agent(with_llm=bool(payload.get("use_llm", True)))
    setups = [StrategySetup.model_validate(item) for item in payload["setups"]]
    risk_data = payload.get("risk")
    risk = RiskSummary.model_validate(risk_data) if risk_data else None
    emit("progress", "Simulating strategy setups", {"stage": "execution"})
    return {"cards": json_safe(agent.simulate_report(setups, risk=risk))}


def _corpus_runner(
    payload: dict[str, Any], emit: EventEmitter, *, reset: bool
) -> dict[str, Any]:
    ensure_core_import_path()
    from rag.corpus_builder import build_initial_corpus

    emit(
        "progress",
        "Loading financial source documents",
        {"stage": "corpus", "reset": reset},
    )
    def progress(event: str, data: dict[str, Any]) -> None:
        stage = str(data.get("stage", "corpus")).replace("_", " ").title()
        emit(event, f"{stage}: {event.replace('_', ' ')}", data)

    result = build_initial_corpus(
        n_fomc=int(payload.get("n_fomc", 5)),
        max_news_per_feed=int(payload.get("max_news_per_feed", 10)),
        tickers=payload.get("tickers"),
        filings_per_ticker=int(payload.get("filings_per_ticker", 1)),
        reset=reset,
        progress_callback=progress,
    )
    return {
        "operation": "reset" if reset else "refresh",
        "completed": True,
        **json_safe(result),
    }


def _demo_runner(payload: dict[str, Any], emit: EventEmitter) -> dict[str, Any]:
    emit("progress", "Loading deterministic dashboard fixture", {"stage": "demo"})
    return _demo_fixture(payload)


def build_default_runners() -> dict[str, JobRunner]:
    """Build the unified runner registry without constructing core agents."""
    return {
        "pipeline": _pipeline_runner,
        "analyst": _analyst_runner,
        "risk": _risk_runner,
        "strategy": _strategy_runner,
        "execution": _execution_runner,
        "corpus_refresh": lambda payload, emit: _corpus_runner(
            payload, emit, reset=False
        ),
        "corpus_reset": lambda payload, emit: _corpus_runner(
            payload, emit, reset=True
        ),
        "demo": _demo_runner,
    }
