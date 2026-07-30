"""Playbook entry signal generators."""

from agents.simulation.signals.registry import (
    get_signal_fn,
    matches_setup_direction,
    signal_at_bar,
    warmup_bars,
)

__all__ = [
    "get_signal_fn",
    "matches_setup_direction",
    "signal_at_bar",
    "warmup_bars",
]
