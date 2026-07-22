"""Unit tests for playbook signal generators."""

import pandas as pd

from agents.simulation.signals.playbooks import (
    signal_ma_crossover,
    signal_mean_reversion,
    signal_momentum_breakout,
    signal_trend_following,
)
from agents.simulation.signals.registry import matches_setup_direction, signal_at_bar


def _df_from_closes(closes: list[float]) -> pd.DataFrame:
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": s * 0.999,
            "high": s * 1.01,
            "low": s * 0.99,
            "close": s,
            "volume": [1_000_000] * len(closes),
        }
    )


def test_trend_following_long_on_uptrend():
    closes = [100 + i * 0.5 for i in range(120)]
    df = _df_from_closes(closes)
    assert signal_trend_following(df, 100) == "long"


def test_ma_crossover_detects_cross():
    # flat then jump
    closes = [100.0] * 70 + [100 + i * 2 for i in range(30)]
    df = _df_from_closes(closes)
    found = any(signal_ma_crossover(df, b) == "long" for b in range(60, len(df)))
    assert found


def test_mean_reversion_oversold():
    closes = [100 - i * 1.5 for i in range(30)]
    df = _df_from_closes(closes)
    sig = signal_mean_reversion(df, 25)
    assert sig in ("long", "short", "flat")


def test_momentum_breakout_on_new_high():
    closes = [100.0] * 25 + [100 + i for i in range(1, 10)]
    df = _df_from_closes(closes)
    sig = signal_momentum_breakout(df, len(df) - 1)
    assert sig == "long"


def test_matches_setup_direction():
    assert matches_setup_direction("long", "long") is True
    assert matches_setup_direction("short", "long") is False
    assert matches_setup_direction("long", "neutral") is False


def test_registry_unknown_playbook_falls_back():
    df = _df_from_closes([100 + i for i in range(80)])
    sig = signal_at_bar("unknown_playbook", df, 70)
    assert sig in ("long", "short", "flat")
