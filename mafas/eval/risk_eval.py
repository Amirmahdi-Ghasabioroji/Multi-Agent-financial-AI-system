"""Live risk-agent metrics evaluation.

Runs the Risk Agent against the configured watchlist and reports computed
metrics plus structural sanity checks on correlations and volatility outputs.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from eval.schemas import EvalCaseResult, EvalMetric, SuiteResult


def _check_correlation_bounds(matrix: dict[str, dict[str, float]]) -> tuple[int, int]:
    valid = 0
    invalid = 0
    for row in matrix.values():
        for value in row.values():
            if -1.0001 <= float(value) <= 1.0001:
                valid += 1
            else:
                invalid += 1
    return valid, invalid


def run_risk_eval(
    *,
    tickers: list[str] | None = None,
    lookback_days: int = 252,
    progress: Callable[[str, dict], None] | None = None,
) -> SuiteResult:
    """Evaluate live risk metrics from market data."""
    started = time.perf_counter()
    label = "Risk agent metrics (live market data)"
    try:
        from agents.risk import build_risk_agent

        if progress:
            progress("stage_started", {"stage": "risk"})

        agent = build_risk_agent(with_llm=False)
        agent.lookback_days = lookback_days
        summary = agent.assess(universe=tickers or None)

        corr_valid, corr_invalid = _check_correlation_bounds(summary.correlation_matrix)
        assets_loaded = len(summary.per_asset)
        sizing_rows = len(summary.position_sizing)
        warnings = len(summary.correlation_warnings)

        per_asset_cases: list[EvalCaseResult] = []
        positive_vol = 0
        for asset in summary.per_asset:
            if asset.realised_vol > 0:
                positive_vol += 1
            per_asset_cases.append(
                EvalCaseResult(
                    id=asset.ticker,
                    label=f"{asset.ticker} volatility profile",
                    metrics=[
                        EvalMetric(
                            name="realised_vol",
                            label="Realised vol",
                            value=round(asset.realised_vol, 4),
                            unit="annualised",
                        ),
                        EvalMetric(
                            name="atr_pct",
                            label="ATR %",
                            value=round(asset.atr_pct, 4),
                            unit="ratio",
                        ),
                        EvalMetric(
                            name="regime",
                            label="Asset regime",
                            value=asset.regime,
                        ),
                    ],
                )
            )

        aggregate = [
            EvalMetric(
                name="assets_loaded",
                label="Assets with price data",
                value=assets_loaded,
                unit="count",
            ),
            EvalMetric(
                name="positive_vol_assets",
                label="Assets with positive vol",
                value=positive_vol,
                unit="count",
            ),
            EvalMetric(
                name="vix_level",
                label="VIX level",
                value=round(summary.vix_level, 2),
            ),
            EvalMetric(
                name="mean_realised_vol",
                label="Mean realised vol",
                value=round(summary.mean_realised_vol, 4),
                unit="annualised",
            ),
            EvalMetric(
                name="vol_regime",
                label="Market vol regime",
                value=summary.vol_regime,
            ),
            EvalMetric(
                name="mean_pairwise_correlation",
                label="Mean pairwise correlation",
                value=summary.concentration.mean_pairwise_correlation,
            ),
            EvalMetric(
                name="effective_number_of_bets",
                label="Effective number of bets",
                value=summary.concentration.effective_number_of_bets,
            ),
            EvalMetric(
                name="correlation_cells_valid",
                label="Valid correlation cells",
                value=corr_valid,
                unit="count",
            ),
            EvalMetric(
                name="correlation_cells_invalid",
                label="Invalid correlation cells",
                value=corr_invalid,
                unit="count",
            ),
            EvalMetric(
                name="correlation_warnings",
                label="High-correlation warnings",
                value=warnings,
                unit="count",
            ),
            EvalMetric(
                name="position_sizing_rows",
                label="Position sizing rows",
                value=sizing_rows,
                unit="count",
            ),
            EvalMetric(
                name="metric_completeness",
                label="Metric completeness",
                value=round(assets_loaded / max(len(summary.universe), 1), 4),
                unit="ratio",
                detail="Share of requested universe with per-asset metrics.",
            ),
        ]

        if progress:
            progress("stage_completed", {"stage": "risk", "assets": assets_loaded})

        return SuiteResult(
            suite="risk",
            label=label,
            status="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            metrics=aggregate,
            cases=per_asset_cases,
        )
    except Exception as exc:  # noqa: BLE001
        return SuiteResult(
            suite="risk",
            label=label,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
