"""Risk-based position sizing for simulated trades."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionSize:
    """Risk-based position size for a single trade."""

    units: float
    notional: float
    notional_pct: float
    risk_amount: float
    risk_pct: float
    capped: bool


def compute_position_size(
    account_equity: float,
    entry_price: float,
    risk_per_unit: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
) -> PositionSize:
    """Size a trade from risk-per-trade, capped by a max notional weight."""
    if entry_price <= 0 or risk_per_unit <= 0 or account_equity <= 0:
        return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, False)

    target_risk = account_equity * max(0.0, risk_per_trade_pct)
    units = target_risk / risk_per_unit
    notional = units * entry_price
    cap = account_equity * max(0.0, max_position_pct)

    capped = False
    if cap > 0 and notional > cap:
        units = cap / entry_price
        notional = cap
        capped = True

    risk_amount = units * risk_per_unit
    return PositionSize(
        units=round(units, 4),
        notional=round(notional, 2),
        notional_pct=round(notional / account_equity, 4),
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_amount / account_equity, 4),
        capped=capped,
    )
