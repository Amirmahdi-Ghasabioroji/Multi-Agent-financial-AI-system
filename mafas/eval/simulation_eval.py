"""Monte Carlo calibration evaluation.

Compares bootstrap barrier probabilities to empirical walk-forward outcomes on
synthetic OHLCV paths. Calibration error is |empirical TP rate − MC P(TP before SL)|.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from agents.risk_metrics import daily_returns
from agents.simulation.barrier import simulate_barrier_bootstrap, walk_forward_barrier
from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult


def _trending_df(n: int, step: float) -> pd.DataFrame:
    closes = [50.0 * (1 + step) ** i for i in range(n)]
    series = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": series * 0.999,
            "high": series * 1.015,
            "low": series * 0.985,
            "close": series,
            "volume": [1_000_000] * n,
        }
    )


def _choppy_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, 0.012, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    series = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": series * 0.999,
            "high": series * 1.008,
            "low": series * 0.992,
            "close": series,
            "volume": [1_000_000] * n,
        }
    )


def _empirical_tp_rate(
    df: pd.DataFrame,
    *,
    direction: str,
    stop_pct: float,
    target_pct: float,
    horizon: int,
    entry_stride: int = 25,
    warmup: int = 60,
) -> tuple[float, int]:
    """Walk forward on real bars and measure TP-before-timeout rate."""
    is_long = direction.lower() != "short"
    outcomes = 0
    tp_hits = 0

    for entry_bar in range(warmup, len(df) - horizon - 1, entry_stride):
        entry_price = float(df["close"].iloc[entry_bar])
        if is_long:
            stop = entry_price * (1 - stop_pct)
            target = entry_price * (1 + target_pct)
        else:
            stop = entry_price * (1 + stop_pct)
            target = entry_price * (1 - target_pct)

        walked = walk_forward_barrier(
            df,
            entry_bar,
            entry_price,
            stop,
            target,
            direction,
            horizon,
        )
        if walked is None:
            continue
        outcomes += 1
        if walked.outcome == "tp":
            tp_hits += 1

    rate = tp_hits / outcomes if outcomes else 0.0
    return round(rate, 4), outcomes


def _calibration_case(
    case_id: str,
    label: str,
    df: pd.DataFrame,
    *,
    direction: str,
    stop_pct: float,
    target_pct: float,
    horizon: int,
) -> EvalCaseResult:
    empirical_rate, n_outcomes = _empirical_tp_rate(
        df,
        direction=direction,
        stop_pct=stop_pct,
        target_pct=target_pct,
        horizon=horizon,
    )
    returns = daily_returns(df["close"]).to_numpy()
    entry_price = float(df["close"].iloc[len(df) // 2])
    if direction.lower() == "short":
        stop = entry_price * (1 + stop_pct)
        target = entry_price * (1 - target_pct)
    else:
        stop = entry_price * (1 - stop_pct)
        target = entry_price * (1 + target_pct)

    mc = simulate_barrier_bootstrap(
        returns,
        entry_price,
        stop,
        target,
        direction=direction,
        max_bars=horizon,
        n_sims=4000,
        seed=17,
    )
    calibration_error = round(abs(empirical_rate - mc.prob_tp_before_sl), 4)
    prob_sum = round(
        mc.prob_tp_before_sl + mc.prob_sl_before_tp + mc.prob_timeout,
        4,
    )

    return EvalCaseResult(
        id=case_id,
        label=label,
        metrics=[
            EvalMetric(
                name="empirical_tp_rate",
                label="Empirical TP rate",
                value=empirical_rate,
                unit="ratio",
                detail=f"From {n_outcomes} walk-forward entries.",
            ),
            EvalMetric(
                name="mc_prob_tp_before_sl",
                label="MC P(TP before SL)",
                value=mc.prob_tp_before_sl,
                unit="ratio",
            ),
            EvalMetric(
                name="calibration_error",
                label="Calibration error",
                value=calibration_error,
                unit="ratio",
                detail="|empirical − MC|",
            ),
            EvalMetric(
                name="probability_mass",
                label="Probability mass sum",
                value=prob_sum,
                unit="ratio",
                detail="TP + SL + timeout should equal 1.",
            ),
        ],
        notes=f"{n_outcomes} empirical paths, {mc.n_sims} simulations",
    )


def run_simulation_eval(
    progress: Callable[[str, dict], None] | None = None,
) -> SuiteResult:
    """Evaluate Monte Carlo calibration on synthetic market paths."""
    started = time.perf_counter()
    label = "Monte Carlo calibration"
    try:
        if progress:
            progress("stage_started", {"stage": "simulation"})

        scenarios = [
            ("trend_long", "Uptrend long setup", _trending_df(420, 0.0035), "long", 0.03, 0.06, 20),
            ("trend_short", "Downtrend short setup", _trending_df(420, -0.003), "short", 0.03, 0.06, 20),
            ("choppy_long", "Choppy market long setup", _choppy_df(420), "long", 0.025, 0.05, 15),
        ]

        cases = [
            _calibration_case(case_id, case_label, frame, direction=direction, stop_pct=stop, target_pct=target, horizon=horizon)
            for case_id, case_label, frame, direction, stop, target, horizon in scenarios
        ]

        calibration_errors = [
            float(metric.value)
            for case in cases
            for metric in case.metrics
            if metric.name == "calibration_error" and isinstance(metric.value, (int, float))
        ]
        prob_sums = [
            float(metric.value)
            for case in cases
            for metric in case.metrics
            if metric.name == "probability_mass" and isinstance(metric.value, (int, float))
        ]

        aggregate = [
            EvalMetric(
                name="mean_calibration_error",
                label="Mean calibration error",
                value=round(sum(calibration_errors) / len(calibration_errors), 4)
                if calibration_errors
                else None,
                unit="ratio",
                detail="Average |empirical TP rate − MC P(TP before SL)| across scenarios.",
            ),
            EvalMetric(
                name="max_calibration_error",
                label="Max calibration error",
                value=round(max(calibration_errors), 4) if calibration_errors else None,
                unit="ratio",
            ),
            EvalMetric(
                name="mean_probability_mass",
                label="Mean probability mass",
                value=round(sum(prob_sums) / len(prob_sums), 4) if prob_sums else None,
                unit="ratio",
            ),
            EvalMetric(
                name="scenario_count",
                label="Scenarios evaluated",
                value=len(cases),
                unit="count",
            ),
        ]

        if progress:
            progress("stage_completed", {"stage": "simulation", "cases": len(cases)})

        return SuiteResult(
            suite="simulation",
            label=label,
            status="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            metrics=aggregate,
            cases=cases,
        )
    except Exception as exc:  # noqa: BLE001
        return SuiteResult(
            suite="simulation",
            label=label,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
