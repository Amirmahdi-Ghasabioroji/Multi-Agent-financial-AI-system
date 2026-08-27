"""Monte Carlo calibration evaluation.

Synthetic paths remain an engine unit-check. Live AAPL/NVDA/SPY paths use a
chronological train/test split. Close-to-close MC is scored only against
close-to-close walks; OHLC MC is scored only against OHLC walks.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from agents.risk_metrics import daily_returns
from agents.simulation.barrier import (
    ohlc_relative_bars,
    simulate_barrier_bootstrap,
    simulate_barrier_ohlc_bootstrap,
    walk_forward_barrier,
)
from agents.simulation.levels import compute_levels_at_bar
from eval.calibration import brier_score, reliability_bins
from eval.market_data import load_ohlcv_frame
from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult

LIVE_TICKERS = ("AAPL", "NVDA", "SPY")
TRAIN_FRAC = 0.70


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
    barrier_mode: str,
    entry_stride: int = 25,
    warmup: int = 60,
) -> tuple[float, int, list[int]]:
    """Walk forward and return TP rate, n, and per-entry binary outcomes."""
    is_long = direction.lower() != "short"
    outcomes: list[int] = []

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
            barrier_mode=barrier_mode,
        )
        if walked is None:
            continue
        outcomes.append(1 if walked.outcome == "tp" else 0)

    n = len(outcomes)
    rate = (sum(outcomes) / n) if n else 0.0
    return round(rate, 4), n, outcomes


def _mc_for_mode(
    df: pd.DataFrame,
    *,
    entry_price: float,
    stop: float,
    target: float,
    direction: str,
    horizon: int,
    barrier_mode: str,
    n_sims: int,
    seed: int,
):
    if barrier_mode == "ohlc":
        return simulate_barrier_ohlc_bootstrap(
            ohlc_relative_bars(df),
            entry_price,
            stop,
            target,
            direction=direction,
            max_bars=horizon,
            n_sims=n_sims,
            seed=seed,
        )
    returns = daily_returns(df["close"]).to_numpy()
    return simulate_barrier_bootstrap(
        returns,
        entry_price,
        stop,
        target,
        direction=direction,
        max_bars=horizon,
        n_sims=n_sims,
        seed=seed,
    )


def _aligned_case(
    case_id: str,
    label: str,
    df: pd.DataFrame,
    *,
    direction: str,
    stop_pct: float,
    target_pct: float,
    horizon: int,
    barrier_mode: str,
    n_sims: int = 4000,
) -> EvalCaseResult:
    empirical_rate, n_outcomes, outcomes = _empirical_tp_rate(
        df,
        direction=direction,
        stop_pct=stop_pct,
        target_pct=target_pct,
        horizon=horizon,
        barrier_mode=barrier_mode,
    )
    entry_price = float(df["close"].iloc[len(df) // 2])
    if direction.lower() == "short":
        stop = entry_price * (1 + stop_pct)
        target = entry_price * (1 - target_pct)
    else:
        stop = entry_price * (1 - stop_pct)
        target = entry_price * (1 + target_pct)

    mc = _mc_for_mode(
        df,
        entry_price=entry_price,
        stop=stop,
        target=target,
        direction=direction,
        horizon=horizon,
        barrier_mode=barrier_mode,
        n_sims=n_sims,
        seed=17,
    )
    calibration_error = round(abs(empirical_rate - mc.prob_tp_before_sl), 4)
    probs = [mc.prob_tp_before_sl] * len(outcomes)
    brier = brier_score(probs, outcomes)
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
                detail=f"{n_outcomes} {barrier_mode} walks.",
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
                detail=f"|empirical − MC| ({barrier_mode} aligned)",
            ),
            EvalMetric(
                name="brier_score",
                label="Brier score",
                value=brier,
                unit="ratio",
                detail="Constant-p Brier on synthetic entries (engine check).",
            ),
            EvalMetric(
                name="probability_mass",
                label="Probability mass sum",
                value=prob_sum,
                unit="ratio",
            ),
        ],
        notes=f"{barrier_mode} aligned · {n_outcomes} empirical · {mc.n_sims} sims",
    )


def _live_walk_forward_case(
    ticker: str,
    df: pd.DataFrame,
    *,
    direction: str = "long",
    horizon: int = 20,
    barrier_mode: str,
    n_sims: int = 2000,
    entry_stride: int = 8,
) -> EvalCaseResult | None:
    split = int(len(df) * TRAIN_FRAC)
    if split < 80 or len(df) - split < horizon + 5:
        return None
    train = df.iloc[:split]
    test_start = split
    probs: list[float] = []
    outcomes: list[int] = []

    for entry_bar in range(test_start, len(df) - horizon - 1, entry_stride):
        levels = compute_levels_at_bar(df, entry_bar, direction, 1.5, 3.0)
        if levels is None:
            continue
        walked = walk_forward_barrier(
            df,
            entry_bar,
            levels.entry,
            levels.stop_loss,
            levels.take_profit,
            direction,
            horizon,
            barrier_mode=barrier_mode,
        )
        if walked is None:
            continue
        mc = _mc_for_mode(
            train,
            entry_price=levels.entry,
            stop=levels.stop_loss,
            target=levels.take_profit,
            direction=direction,
            horizon=horizon,
            barrier_mode=barrier_mode,
            n_sims=n_sims,
            seed=17 + entry_bar,
        )
        if mc.n_sims <= 0:
            continue
        probs.append(mc.prob_tp_before_sl)
        outcomes.append(1 if walked.outcome == "tp" else 0)

    if len(outcomes) < 5:
        return None

    empirical = sum(outcomes) / len(outcomes)
    mean_p = sum(probs) / len(probs)
    brier = brier_score(probs, outcomes)
    bins = reliability_bins(probs, outcomes, n_bins=5)
    occupied = [b for b in bins if b["count"] > 0]
    return EvalCaseResult(
        id=f"live_{ticker.lower()}_{barrier_mode}",
        label=f"{ticker} walk-forward {barrier_mode} (train {TRAIN_FRAC:.0%})",
        metrics=[
            EvalMetric(
                name="empirical_tp_rate",
                label="Out-of-sample TP rate",
                value=round(empirical, 4),
                unit="ratio",
                detail=f"{len(outcomes)} test entries; ATR 1.5/3.0 brackets.",
            ),
            EvalMetric(
                name="mean_mc_prob",
                label="Mean MC P(TP)",
                value=round(mean_p, 4),
                unit="ratio",
            ),
            EvalMetric(
                name="calibration_error",
                label="Calibration error",
                value=round(abs(empirical - mean_p), 4),
                unit="ratio",
                detail="|mean y − mean p| on held-out bars",
            ),
            EvalMetric(
                name="brier_score",
                label="Brier score",
                value=brier,
                unit="ratio",
            ),
            EvalMetric(
                name="n_test_entries",
                label="Test entries",
                value=len(outcomes),
                unit="count",
            ),
        ],
        notes="; ".join(
            f"p={b['mean_predicted']:.2f} y={b['mean_observed']:.2f} n={int(b['count'])}"
            for b in occupied
        ),
    )


def _metric_values(cases: list[EvalCaseResult], name: str) -> list[float]:
    return [
        float(metric.value)
        for case in cases
        for metric in case.metrics
        if metric.name == name and isinstance(metric.value, (int, float))
    ]


def run_simulation_eval(
    progress: Callable[[str, dict], None] | None = None,
) -> SuiteResult:
    """Evaluate Monte Carlo calibration on synthetic and live paths."""
    started = time.perf_counter()
    label = "Monte Carlo calibration (aligned engines)"
    try:
        if progress:
            progress("stage_started", {"stage": "simulation"})

        synthetic_specs = [
            ("trend_long", "Uptrend long setup", _trending_df(420, 0.0035), "long", 0.03, 0.06, 20),
            ("trend_short", "Downtrend short setup", _trending_df(420, -0.003), "short", 0.03, 0.06, 20),
            ("choppy_long", "Choppy market long setup", _choppy_df(420), "long", 0.025, 0.05, 15),
        ]

        cases: list[EvalCaseResult] = []
        for case_id, case_label, frame, direction, stop, target, horizon in synthetic_specs:
            for mode in ("close", "ohlc"):
                cases.append(
                    _aligned_case(
                        f"{case_id}_{mode}",
                        f"{case_label} ({mode})",
                        frame,
                        direction=direction,
                        stop_pct=stop,
                        target_pct=target,
                        horizon=horizon,
                        barrier_mode=mode,
                        n_sims=1500,
                    )
                )

        live_notes: list[str] = []
        for ticker in LIVE_TICKERS:
            if progress:
                progress("progress", {"stage": "simulation", "case": ticker})
            frame = load_ohlcv_frame(ticker, days=504)
            if frame is None:
                live_notes.append(f"{ticker}: no OHLCV")
                cases.append(
                    EvalCaseResult(
                        id=f"live_{ticker.lower()}_unavailable",
                        label=f"{ticker} live path",
                        notes="Could not load OHLCV — live calibration skipped for this name.",
                    )
                )
                continue
            for mode in ("close", "ohlc"):
                live_case = _live_walk_forward_case(ticker, frame, barrier_mode=mode)
                if live_case is None:
                    live_notes.append(f"{ticker} {mode}: too few test entries")
                    continue
                cases.append(live_case)

        synthetic_cases = [c for c in cases if not str(c.id).startswith("live_")]
        live_cases = [
            c for c in cases
            if str(c.id).startswith("live_") and any(m.name == "brier_score" for m in c.metrics)
        ]

        synth_err = _metric_values(synthetic_cases, "calibration_error")
        live_brier = _metric_values(live_cases, "brier_score")
        live_err = _metric_values(live_cases, "calibration_error")
        prob_sums = _metric_values(cases, "probability_mass")

        aggregate = [
            EvalMetric(
                name="mean_calibration_error",
                label="Mean aligned calibration error (synthetic)",
                value=round(sum(synth_err) / len(synth_err), 4) if synth_err else None,
                unit="ratio",
                detail="Close vs close and OHLC vs OHLC on toy paths — engine check, not live evidence.",
            ),
            EvalMetric(
                name="max_calibration_error",
                label="Max synthetic calibration error",
                value=round(max(synth_err), 4) if synth_err else None,
                unit="ratio",
            ),
            EvalMetric(
                name="live_mean_brier",
                label="Live mean Brier score",
                value=round(sum(live_brier) / len(live_brier), 4) if live_brier else None,
                unit="ratio",
                detail="Walk-forward on AAPL/NVDA/SPY; train returns only in the bootstrap pool.",
            ),
            EvalMetric(
                name="live_mean_calibration_error",
                label="Live mean |ȳ − p̄|",
                value=round(sum(live_err) / len(live_err), 4) if live_err else None,
                unit="ratio",
            ),
            EvalMetric(
                name="live_cases",
                label="Live aligned cases",
                value=len(live_cases),
                unit="count",
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
