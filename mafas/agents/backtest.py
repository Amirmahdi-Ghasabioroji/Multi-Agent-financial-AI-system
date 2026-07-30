"""Compatibility re-exports — prefer agents.simulation.* for new code."""

from agents.simulation.barrier import (
    HORIZON_BARS,
    SimulationResult,
    simulate_barrier_bootstrap,
    walk_forward_barrier,
)
from agents.simulation.sizing import PositionSize, compute_position_size

__all__ = [
    "HORIZON_BARS",
    "SimulationResult",
    "PositionSize",
    "compute_position_size",
    "simulate_barrier_bootstrap",
    "walk_forward_barrier",
]
