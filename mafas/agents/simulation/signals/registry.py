"""Map playbook keys to signal functions."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from loguru import logger

from agents.simulation.signals.base import SignalDirection, WARMUP_BARS
from agents.simulation.signals.playbooks import (
    signal_carry,
    signal_ma_crossover,
    signal_mean_reversion,
    signal_momentum_breakout,
    signal_pairs_relative_value,
    signal_range_support_resistance,
    signal_trend_following,
    signal_volatility_based,
)

SignalFn = Callable[[pd.DataFrame, int, pd.DataFrame | None], SignalDirection]

_REGISTRY: dict[str, SignalFn] = {
    "trend_following": lambda df, bar, _b=None: signal_trend_following(df, bar),
    "ma_crossover": lambda df, bar, _b=None: signal_ma_crossover(df, bar),
    "momentum_breakout": lambda df, bar, _b=None: signal_momentum_breakout(df, bar),
    "mean_reversion": lambda df, bar, _b=None: signal_mean_reversion(df, bar),
    "range_support_resistance": lambda df, bar, _b=None: signal_range_support_resistance(df, bar),
    "volatility_based": lambda df, bar, _b=None: signal_volatility_based(df, bar),
    "carry": lambda df, bar, _b=None: signal_carry(df, bar),
    "pairs_relative_value": signal_pairs_relative_value,
}


def get_signal_fn(playbook_key: str) -> SignalFn:
    fn = _REGISTRY.get(playbook_key)
    if fn is None:
        logger.warning("Unknown playbook '{}'; falling back to trend_following", playbook_key)
        return _REGISTRY["trend_following"]
    return fn


def signal_at_bar(
    playbook_key: str,
    df: pd.DataFrame,
    bar: int,
    df_b: pd.DataFrame | None = None,
) -> SignalDirection:
    return get_signal_fn(playbook_key)(df, bar, df_b)


def matches_setup_direction(signal: SignalDirection, setup_direction: str) -> bool:
    if setup_direction == "neutral":
        return False
    return signal == setup_direction


def warmup_bars(playbook_key: str) -> int:
    if playbook_key == "carry":
        return 252
    if playbook_key == "pairs_relative_value":
        return WARMUP_BARS
    return WARMUP_BARS
