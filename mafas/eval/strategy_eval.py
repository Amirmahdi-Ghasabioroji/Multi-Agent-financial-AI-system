"""Strategy / playbook outcome evaluation versus simple baselines."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from agents.risk_metrics import TRADING_DAYS
from agents.simulation.historical import BacktestConfig, run_historical_backtest
from agents.simulation.metrics import sharpe_from_equity, sortino_from_equity
from agents.simulation.signals.base import sma
from agents.strategy_playbooks import PLAYBOOKS
from agents.strategy_schemas import StrategySetup
from eval.market_data import load_ohlcv_frame
from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult

BASELINE_POSITION_PCT = 0.10


def _sleeve_equity(df: pd.DataFrame, weights: np.ndarray, initial: float, sleeve: float) -> np.ndarray:
    close = df["close"].astype(float).to_numpy()
    rets = np.zeros(len(close))
    rets[1:] = close[1:] / close[:-1] - 1.0
    w = np.asarray(weights, dtype=float)
    if w.size != rets.size:
        w = np.resize(w, rets.size)
    strat = rets * w * sleeve
    equity = [initial]
    running = initial
    for r in strat:
        running *= 1.0 + r
        equity.append(running)
    return np.array(equity, dtype=float)


def buy_and_hold_stats(df: pd.DataFrame, initial: float = 100_000.0, sleeve: float = BASELINE_POSITION_PCT) -> dict:
    weights = np.ones(len(df), dtype=float)
    equity = _sleeve_equity(df, weights, initial, sleeve)
    total_return = float(equity[-1] / initial - 1.0)
    return {
        "sharpe": sharpe_from_equity(equity, TRADING_DAYS),
        "sortino": sortino_from_equity(equity, TRADING_DAYS),
        "total_return_pct": total_return,
        "n_days": float(len(df)),
    }


def sma50_long_flat_stats(df: pd.DataFrame, initial: float = 100_000.0, sleeve: float = BASELINE_POSITION_PCT) -> dict:
    close = df["close"].astype(float)
    signal = (close > sma(close, 50)).astype(float).shift(1).fillna(0.0).to_numpy()
    equity = _sleeve_equity(df, signal, initial, sleeve)
    total_return = float(equity[-1] / initial - 1.0)
    return {
        "sharpe": sharpe_from_equity(equity, TRADING_DAYS),
        "sortino": sortino_from_equity(equity, TRADING_DAYS),
        "total_return_pct": total_return,
        "n_days": float(len(df)),
    }


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def run_strategy_eval(
    progress: Callable[[str, dict], None] | None = None,
) -> SuiteResult:
    started = time.perf_counter()
    label = "Strategy vs baselines"
    try:
        if progress:
            progress("stage_started", {"stage": "strategy"})

        from agents.risk import build_risk_agent
        from agents.strategy import build_strategy_agent

        risk_agent = build_risk_agent(with_llm=False)
        risk = risk_agent.assess()
        llm_agent = build_strategy_agent(with_llm=True)
        det_agent = build_strategy_agent(with_llm=False)
        llm_report = llm_agent.decide(risk, briefing=None)
        det_report = det_agent.decide(risk, briefing=None)

        llm_setups = llm_report.setups
        n_llm = len(llm_setups)
        missing_instrument = sum(1 for s in llm_setups if not s.instrument)
        dropped_to_fallback = 0.0 if llm_report.llm_used else 1.0

        cases: list[EvalCaseResult] = []
        chosen_sharpes: list[float] = []
        excess_vs_bh: list[float] = []
        excess_vs_sma: list[float] = []
        beat_random: list[float] = []

        rng = random.Random(42)
        playbook_keys = list(PLAYBOOKS.keys())

        for i, setup in enumerate(llm_setups or det_report.setups):
            if progress:
                progress("progress", {"stage": "strategy", "case": setup.strategy})
            ticker = (setup.instrument or "").split("/")[0]
            if not ticker:
                cases.append(
                    EvalCaseResult(
                        id=f"setup_{i}_{setup.strategy}",
                        label=f"{setup.strategy_name} (no instrument)",
                        notes="Invalid / missing instrument — skipped backtest.",
                    )
                )
                continue
            df = load_ohlcv_frame(ticker, days=504)
            if df is None:
                cases.append(
                    EvalCaseResult(
                        id=f"setup_{i}_{setup.strategy}",
                        label=f"{setup.strategy_name} | {ticker}",
                        notes="OHLCV unavailable.",
                    )
                )
                continue

            config = BacktestConfig(account_equity=100_000.0, seed=42)
            chosen = run_historical_backtest(df, setup, config)
            bh = buy_and_hold_stats(df)
            sma_stats = sma50_long_flat_stats(df)

            other_keys = [k for k in playbook_keys if k != setup.strategy] or playbook_keys
            random_key = rng.choice(other_keys)
            random_setup = StrategySetup(
                strategy=random_key,
                strategy_name=PLAYBOOKS[random_key].name,
                instrument=setup.instrument,
                direction=setup.direction if setup.direction != "neutral" else "long",
                horizon=setup.horizon,
                confidence=setup.confidence,
                playbook_fit=setup.playbook_fit,
            )
            random_bt = run_historical_backtest(df, random_setup, config)

            c_sh = chosen.metrics.sharpe_ratio
            r_sh = random_bt.metrics.sharpe_ratio
            if c_sh is not None:
                chosen_sharpes.append(c_sh)
                if bh["sharpe"] is not None:
                    excess_vs_bh.append(c_sh - bh["sharpe"])
                if sma_stats["sharpe"] is not None:
                    excess_vs_sma.append(c_sh - sma_stats["sharpe"])
            if c_sh is not None and r_sh is not None:
                beat_random.append(1.0 if c_sh > r_sh else 0.0)

            cases.append(
                EvalCaseResult(
                    id=f"setup_{i}_{setup.strategy}_{ticker}",
                    label=f"{setup.strategy_name} | {ticker} {setup.direction}",
                    metrics=[
                        EvalMetric(
                            name="chosen_sharpe",
                            label="Chosen playbook Sharpe (calendar)",
                            value=_round_or_none(c_sh),
                        ),
                        EvalMetric(
                            name="chosen_total_return",
                            label="Chosen total return",
                            value=round(chosen.metrics.total_return_pct, 4),
                            unit="ratio",
                            detail="Per-trade 1% risk sizing — not comparable as P/L to 10% sleeve baselines.",
                        ),
                        EvalMetric(
                            name="buy_hold_sharpe",
                            label="Buy-and-hold Sharpe (10% sleeve)",
                            value=_round_or_none(bh["sharpe"]),
                        ),
                        EvalMetric(
                            name="sma50_sharpe",
                            label="SMA50 long/flat Sharpe (10% sleeve)",
                            value=_round_or_none(sma_stats["sharpe"]),
                        ),
                        EvalMetric(
                            name="random_playbook_sharpe",
                            label=f"Random playbook Sharpe ({random_key})",
                            value=_round_or_none(r_sh),
                        ),
                        EvalMetric(
                            name="n_trades",
                            label="Chosen n trades",
                            value=chosen.metrics.n_trades,
                            unit="count",
                        ),
                        EvalMetric(
                            name="low_sample",
                            label="Low sample",
                            value=1 if chosen.metrics.low_sample else 0,
                            unit="ratio",
                        ),
                    ],
                    notes=f"llm_used={llm_report.llm_used}",
                )
            )

        def _mean(xs: list[float]) -> float | None:
            return round(sum(xs) / len(xs), 4) if xs else None

        aggregate = [
            EvalMetric(
                name="llm_used_rate",
                label="LLM used (this run)",
                value=1.0 if llm_report.llm_used else 0.0,
                unit="ratio",
                detail="1 if LLM setups were accepted; 0 if deterministic fallback.",
            ),
            EvalMetric(
                name="fallback_rate",
                label="Fallback rate",
                value=dropped_to_fallback,
                unit="ratio",
            ),
            EvalMetric(
                name="invalid_instrument_rate",
                label="Invalid-instrument rate (LLM setups)",
                value=round(missing_instrument / n_llm, 4) if n_llm else 0.0,
                unit="ratio",
            ),
            EvalMetric(
                name="mean_chosen_sharpe",
                label="Mean chosen calendar Sharpe",
                value=_mean(chosen_sharpes),
            ),
            EvalMetric(
                name="mean_excess_sharpe_vs_buy_hold",
                label="Mean excess Sharpe vs buy-and-hold",
                value=_mean(excess_vs_bh),
                detail="Positive = chosen playbook Sharpe beat a 10% B&H sleeve on the same window.",
            ),
            EvalMetric(
                name="mean_excess_sharpe_vs_sma50",
                label="Mean excess Sharpe vs SMA50",
                value=_mean(excess_vs_sma),
            ),
            EvalMetric(
                name="beat_random_rate",
                label="Beat random playbook rate",
                value=_mean(beat_random),
                unit="ratio",
            ),
            EvalMetric(name="setups_evaluated", label="Setups evaluated", value=len(cases), unit="count"),
            EvalMetric(
                name="weights_unvalidated",
                label="Playbook weights still unvalidated",
                value=1,
                unit="count",
                detail="0.45 / 0.35 / 0.20 regime/bias/corr weights were not retuned.",
            ),
        ]

        if progress:
            progress("stage_completed", {"stage": "strategy", "cases": len(cases)})

        return SuiteResult(
            suite="strategy",
            label=label,
            status="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            metrics=aggregate,
            cases=cases,
        )
    except Exception as exc:  # noqa: BLE001
        return SuiteResult(
            suite="strategy",
            label=label,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
