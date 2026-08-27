"""Orchestrator gate sweep: broaden / no-trade / trade vs labelled queries.

Each query is recorded once per LLM mode. Thresholds are applied post-hoc so a
3×3×3 grid does not re-run agents. Production defaults stay 0.40 / 0.45 / 0.55.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agents.orchestrator import (
    ANALYST_CONFIDENCE_THRESHOLD,
    HIGH_VOL_CONFIDENCE_FLOOR,
    MAX_ANALYST_RETRIES,
    SETUP_CONFIDENCE_FLOOR,
    analyst_route,
    broaden_query_text,
    strategy_route,
)
from agents.risk_schemas import RiskSummary
from agents.schemas import MacroBriefing
from agents.strategy_schemas import StrategyReport
from eval.json_util import load_gold
from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult

ANALYST_GRID = (0.30, 0.40, 0.50)
SETUP_GRID = (0.35, 0.45, 0.55)
HIGH_VOL_GRID = (0.45, 0.55, 0.65)


@dataclass
class AttemptSnapshot:
    attempt: int
    query: str
    confidence: float
    briefing: MacroBriefing | None
    risk: RiskSummary | None
    strategy: StrategyReport | None


@dataclass
class QueryTrace:
    spec: dict
    use_llm: bool
    attempts: list[AttemptSnapshot] = field(default_factory=list)
    error: str = ""


def _select_attempt(
    attempts: list[AttemptSnapshot],
    analyst_floor: float,
    max_retries: int = MAX_ANALYST_RETRIES,
) -> AttemptSnapshot | None:
    if not attempts:
        return None
    for snap in attempts:
        route = analyst_route(snap.confidence, snap.attempt, analyst_floor, max_retries)
        if route == "risk":
            return snap
    return attempts[-1]


def decision_for_floors(
    trace: QueryTrace,
    analyst_floor: float,
    setup_floor: float,
    high_vol_floor: float,
) -> str:
    snap = _select_attempt(trace.attempts, analyst_floor)
    if snap is None or snap.strategy is None or snap.risk is None:
        return "no_trade"
    return strategy_route(
        snap.strategy.setups,
        snap.risk.vol_regime,
        setup_floor,
        high_vol_floor,
    )


def sweep_precision(
    traces: list[QueryTrace],
    analyst_floor: float,
    setup_floor: float,
    high_vol_floor: float,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    broaden = 0
    trade = 0
    n = 0
    for trace in traces:
        if not trace.attempts:
            continue
        n += 1
        first = trace.attempts[0]
        if analyst_route(first.confidence, first.attempt, analyst_floor, MAX_ANALYST_RETRIES) == "broaden":
            broaden += 1
        predicted = decision_for_floors(trace, analyst_floor, setup_floor, high_vol_floor)
        expected = str(trace.spec.get("expected", "no_trade"))
        if predicted == "execution":
            trade += 1
            if expected == "trade":
                tp += 1
            else:
                fp += 1
        else:
            if expected == "no_trade":
                tn += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": float(n),
        "trade_rate": trade / n if n else 0.0,
        "broaden_rate": broaden / n if n else 0.0,
        "no_trade_rate": (n - trade) / n if n else 0.0,
        "precision_trade": precision,
        "recall_trade": recall,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _record_query(
    spec: dict,
    *,
    use_llm: bool,
    analyst,
    risk_agent,
    strategy_agent,
    progress: Callable[[str, dict], None] | None = None,
) -> QueryTrace:
    trace = QueryTrace(spec=spec, use_llm=use_llm)
    query = str(spec.get("query", ""))
    tickers = list(spec.get("tickers") or [])
    max_analyst = max(ANALYST_GRID)
    current = query
    try:
        for attempt in range(1, MAX_ANALYST_RETRIES + 2):
            if progress:
                progress("progress", {"stage": "gates", "case": spec.get("id"), "attempt": attempt, "llm": use_llm})
            briefing = analyst.brief(current, use_llm=use_llm)
            snap = AttemptSnapshot(
                attempt=attempt,
                query=current,
                confidence=float(briefing.confidence),
                briefing=briefing,
                risk=None,
                strategy=None,
            )
            risk = risk_agent.assess(universe=tickers or None, briefing=briefing)
            report = strategy_agent.decide(risk, briefing=briefing)
            snap.risk = risk
            snap.strategy = report
            trace.attempts.append(snap)
            if briefing.confidence >= max_analyst:
                break
            if attempt > MAX_ANALYST_RETRIES:
                break
            current = broaden_query_text(current, getattr(analyst, "llm", None), use_llm)
    except Exception as exc:  # noqa: BLE001
        trace.error = f"{type(exc).__name__}: {exc}"
    return trace


def _default_row(traces: list[QueryTrace]) -> dict[str, float]:
    return sweep_precision(
        traces,
        ANALYST_CONFIDENCE_THRESHOLD,
        SETUP_CONFIDENCE_FLOOR,
        HIGH_VOL_CONFIDENCE_FLOOR,
    )


def _grid_cases(traces: list[QueryTrace], prefix: str) -> list[EvalCaseResult]:
    cases: list[EvalCaseResult] = []
    for a in ANALYST_GRID:
        for s in SETUP_GRID:
            for h in HIGH_VOL_GRID:
                stats = sweep_precision(traces, a, s, h)
                is_default = (
                    a == ANALYST_CONFIDENCE_THRESHOLD
                    and s == SETUP_CONFIDENCE_FLOOR
                    and h == HIGH_VOL_CONFIDENCE_FLOOR
                )
                cases.append(
                    EvalCaseResult(
                        id=f"{prefix}_a{a:.2f}_s{s:.2f}_h{h:.2f}",
                        label=f"{prefix} analyst={a:.2f} setup={s:.2f} high_vol={h:.2f}",
                        metrics=[
                            EvalMetric(name="precision_trade", label="Precision (trade)", value=round(stats["precision_trade"], 4), unit="ratio"),
                            EvalMetric(name="recall_trade", label="Recall (trade)", value=round(stats["recall_trade"], 4), unit="ratio"),
                            EvalMetric(name="trade_rate", label="Trade rate", value=round(stats["trade_rate"], 4), unit="ratio"),
                            EvalMetric(name="broaden_rate", label="Broaden rate", value=round(stats["broaden_rate"], 4), unit="ratio"),
                            EvalMetric(name="no_trade_rate", label="No-trade rate", value=round(stats["no_trade_rate"], 4), unit="ratio"),
                        ],
                        notes="production defaults" if is_default else "",
                    )
                )
    return cases


def run_gates_eval(
    progress: Callable[[str, dict], None] | None = None,
) -> SuiteResult:
    started = time.perf_counter()
    label = "Orchestrator gates (threshold sweep)"
    try:
        if progress:
            progress("stage_started", {"stage": "gates"})

        from agents.analyst import build_analyst_agent
        from agents.risk import build_risk_agent
        from agents.strategy import build_strategy_agent

        gold = load_gold("gates_queries.json")
        analyst = build_analyst_agent()
        risk_nollm = build_risk_agent(with_llm=False)
        strat_nollm = build_strategy_agent(with_llm=False)

        nollm_traces = [
            _record_query(
                spec,
                use_llm=False,
                analyst=analyst,
                risk_agent=risk_nollm,
                strategy_agent=strat_nollm,
                progress=progress,
            )
            for spec in gold
        ]

        llm_available = bool(getattr(getattr(analyst, "llm", None), "is_available", lambda: False)())
        llm_traces: list[QueryTrace] = []
        if llm_available:
            risk_llm = build_risk_agent(with_llm=True)
            strat_llm = build_strategy_agent(with_llm=True)
            llm_traces = [
                _record_query(
                    spec,
                    use_llm=True,
                    analyst=analyst,
                    risk_agent=risk_llm,
                    strategy_agent=strat_llm,
                    progress=progress,
                )
                for spec in gold
            ]

        nollm_default = _default_row(nollm_traces)
        cases = _grid_cases(nollm_traces, "no_llm")
        if llm_traces:
            llm_default = _default_row(llm_traces)
            cases.extend(_grid_cases(llm_traces, "with_llm"))
        else:
            llm_default = None
            cases.append(
                EvalCaseResult(
                    id="with_llm_skipped",
                    label="with-LLM sweep skipped",
                    notes="Ollama unavailable — with-LLM operating points not measured.",
                )
            )

        recorded = sum(1 for t in nollm_traces if t.attempts)
        failed = sum(1 for t in nollm_traces if t.error)

        aggregate = [
            EvalMetric(
                name="default_precision_trade",
                label="Default-floor precision (no-LLM)",
                value=round(nollm_default["precision_trade"], 4),
                unit="ratio",
                detail="Floors 0.40 / 0.45 / 0.55 are uncalibrated product policy; this is the measured operating point.",
            ),
            EvalMetric(
                name="default_trade_rate",
                label="Default-floor trade rate (no-LLM)",
                value=round(nollm_default["trade_rate"], 4),
                unit="ratio",
            ),
            EvalMetric(
                name="default_broaden_rate",
                label="Default-floor broaden rate (no-LLM)",
                value=round(nollm_default["broaden_rate"], 4),
                unit="ratio",
            ),
            EvalMetric(
                name="default_no_trade_rate",
                label="Default-floor no-trade rate (no-LLM)",
                value=round(nollm_default["no_trade_rate"], 4),
                unit="ratio",
            ),
            EvalMetric(
                name="with_llm_precision_trade",
                label="Default-floor precision (with-LLM)",
                value=round(llm_default["precision_trade"], 4) if llm_default else None,
                unit="ratio",
            ),
            EvalMetric(
                name="with_llm_trade_rate",
                label="Default-floor trade rate (with-LLM)",
                value=round(llm_default["trade_rate"], 4) if llm_default else None,
                unit="ratio",
            ),
            EvalMetric(name="queries_recorded", label="Queries recorded (no-LLM)", value=recorded, unit="count"),
            EvalMetric(name="record_errors", label="Record errors", value=failed, unit="count"),
            EvalMetric(name="grid_points", label="Grid points per mode", value=len(ANALYST_GRID) * len(SETUP_GRID) * len(HIGH_VOL_GRID), unit="count"),
        ]

        if progress:
            progress("stage_completed", {"stage": "gates", "cases": len(cases)})

        return SuiteResult(
            suite="gates",
            label=label,
            status="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            metrics=aggregate,
            cases=cases,
        )
    except Exception as exc:  # noqa: BLE001
        return SuiteResult(
            suite="gates",
            label=label,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
