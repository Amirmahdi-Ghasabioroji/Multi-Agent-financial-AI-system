"""ATR-based trade level geometry."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.risk_metrics import average_true_range


@dataclass
class TradeLevelGeometry:
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    risk_per_unit: float
    planned_rr: float


def compute_levels_at_bar(
    df: pd.DataFrame,
    bar_index: int,
    direction: str,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
) -> TradeLevelGeometry | None:
    """Compute entry/stop/target at a specific bar using ATR from data up to that bar."""
    if bar_index < 0 or bar_index >= len(df):
        return None
    slice_df = df.iloc[: bar_index + 1]
    atr = average_true_range(slice_df, period=14)
    entry = float(slice_df["close"].astype(float).iloc[-1])
    if atr <= 0 or entry <= 0:
        return None

    is_long = direction.lower() != "short"
    if is_long:
        stop = entry - sl_atr_mult * atr
        target = entry + tp_atr_mult * atr
    else:
        stop = entry + sl_atr_mult * atr
        target = entry - tp_atr_mult * atr
    risk_per_unit = abs(entry - stop)
    planned_rr = tp_atr_mult / sl_atr_mult if sl_atr_mult > 0 else 0.0

    return TradeLevelGeometry(
        entry=round(entry, 4),
        stop_loss=round(stop, 4),
        take_profit=round(target, 4),
        atr=round(atr, 4),
        risk_per_unit=round(risk_per_unit, 4),
        planned_rr=round(planned_rr, 3),
    )


def compute_levels_latest(
    df: pd.DataFrame,
    direction: str,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
) -> TradeLevelGeometry | None:
    """Compute levels at the last bar of the dataframe."""
    if df.empty:
        return None
    return compute_levels_at_bar(df, len(df) - 1, direction, sl_atr_mult, tp_atr_mult)
