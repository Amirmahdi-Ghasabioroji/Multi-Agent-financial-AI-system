"""Simulation engines for historical backtest and forward Monte Carlo."""

from agents.simulation.barrier import (
    HORIZON_BARS,
    SimulationResult,
    ohlc_relative_bars,
    simulate_barrier_bootstrap,
    simulate_barrier_ohlc_bootstrap,
    walk_forward_barrier,
)
from agents.simulation.sizing import PositionSize, compute_position_size

__all__ = [
    "HORIZON_BARS",
    "SimulationResult",
    "PositionSize",
    "compute_position_size",
    "simulate_barrier_bootstrap",
    "simulate_barrier_ohlc_bootstrap",
    "ohlc_relative_bars",
    "walk_forward_barrier",
]
