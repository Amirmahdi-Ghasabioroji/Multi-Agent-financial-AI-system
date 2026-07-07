"""Strategy playbook registry + deterministic suitability scoring.

A *playbook* is a plain description of a strategy family and the environment in
which it tends to work. The Strategy Agent scores every playbook against the
current environment (vol regime, macro bias, cross-asset correlation) using the
pure functions here, then lets the LLM reason over the top candidates.

Keeping the conditional logic here — separate from the LLM — makes the agent's
core reasoning reproducible and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical vocabularies used across the strategy layer.
REGIMES = ("low", "medium", "high")
BIASES = ("bullish", "bearish", "neutral")

# Weights blending the three deterministic suitability signals.
_W_REGIME = 0.45
_W_BIAS = 0.35
_W_CORR = 0.20

# mean pairwise correlation at/above which correlation-dependent strategies
# (e.g. pairs / relative value) are considered well supported.
_CORR_REFERENCE = 0.6


@dataclass(frozen=True)
class Playbook:
    """A strategy family and the conditions under which it performs."""

    key: str
    name: str
    description: str
    # Suitability weight (0-1) of each vol regime.
    regime_fit: dict[str, float]
    # Suitability weight (0-1) of each macro bias.
    bias_fit: dict[str, float]
    # How the strategy relates to cross-asset correlation.
    corr_pref: str = "neutral"  # "high" | "low" | "neutral"
    directional: bool = True
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Registry — 8 playbooks spanning trend, reversion, vol, and relative value.
# --------------------------------------------------------------------------- #
PLAYBOOKS: dict[str, Playbook] = {
    "trend_following": Playbook(
        key="trend_following",
        name="Trend Following",
        description=(
            "Ride established directional moves. Works when a clear macro trend "
            "exists and volatility is contained; whipsaws in choppy, high-vol tape."
        ),
        regime_fit={"low": 0.85, "medium": 0.90, "high": 0.35},
        bias_fit={"bullish": 0.90, "bearish": 0.90, "neutral": 0.20},
        corr_pref="neutral",
        directional=True,
        tags=["directional", "trend"],
    ),
    "mean_reversion": Playbook(
        key="mean_reversion",
        name="Mean Reversion",
        description=(
            "Fade stretched moves back toward a mean. Works in calm, range-bound "
            "markets with no strong directional macro; dangerous in trends."
        ),
        regime_fit={"low": 0.90, "medium": 0.60, "high": 0.30},
        bias_fit={"bullish": 0.45, "bearish": 0.45, "neutral": 0.90},
        corr_pref="neutral",
        directional=False,
        tags=["reversion", "range"],
    ),
    "momentum_breakout": Playbook(
        key="momentum_breakout",
        name="Momentum / Breakout",
        description=(
            "Enter on breaks of consolidation with expanding volatility. Needs a "
            "directional catalyst and rising vol; poor in dead, low-vol ranges."
        ),
        regime_fit={"low": 0.40, "medium": 0.80, "high": 0.75},
        bias_fit={"bullish": 0.85, "bearish": 0.70, "neutral": 0.30},
        corr_pref="neutral",
        directional=True,
        tags=["directional", "momentum", "breakout"],
    ),
    "volatility_based": Playbook(
        key="volatility_based",
        name="Volatility-Based (long vol / vol-target)",
        description=(
            "Position around volatility itself — long optionality or defensive "
            "vol-targeting. Shines in high-vol, risk-off regimes; costly in calm."
        ),
        regime_fit={"low": 0.35, "medium": 0.60, "high": 0.95},
        bias_fit={"bullish": 0.50, "bearish": 0.75, "neutral": 0.60},
        corr_pref="neutral",
        directional=False,
        tags=["volatility", "defensive"],
    ),
    "ma_crossover": Playbook(
        key="ma_crossover",
        name="Moving-Average Crossover",
        description=(
            "Systematic trend proxy using fast/slow MA crosses. Works in trending, "
            "moderate-vol markets; generates false signals when ranging or gapping."
        ),
        regime_fit={"low": 0.80, "medium": 0.85, "high": 0.40},
        bias_fit={"bullish": 0.85, "bearish": 0.85, "neutral": 0.25},
        corr_pref="neutral",
        directional=True,
        tags=["directional", "trend", "systematic"],
    ),
    "range_support_resistance": Playbook(
        key="range_support_resistance",
        name="Range / Support-Resistance",
        description=(
            "Trade bounces off well-defined range boundaries. Best in low-vol, "
            "neutral markets with no trend; breaks down when a trend emerges."
        ),
        regime_fit={"low": 0.90, "medium": 0.55, "high": 0.25},
        bias_fit={"bullish": 0.40, "bearish": 0.40, "neutral": 0.90},
        corr_pref="neutral",
        directional=False,
        tags=["range", "reversion"],
    ),
    "carry": Playbook(
        key="carry",
        name="Carry",
        description=(
            "Harvest yield/roll differentials in calm markets. Works when vol is "
            "low and stable; prone to sharp unwinds when volatility spikes."
        ),
        regime_fit={"low": 0.90, "medium": 0.60, "high": 0.20},
        bias_fit={"bullish": 0.70, "bearish": 0.40, "neutral": 0.70},
        corr_pref="low",
        directional=True,
        tags=["carry", "income"],
    ),
    "pairs_relative_value": Playbook(
        key="pairs_relative_value",
        name="Pairs / Relative Value",
        description=(
            "Long/short two correlated instruments to trade their spread, hedging "
            "market direction. Needs genuinely correlated pairs; robust across "
            "regimes and especially useful when concentration/correlation is high."
        ),
        regime_fit={"low": 0.70, "medium": 0.75, "high": 0.60},
        bias_fit={"bullish": 0.60, "bearish": 0.60, "neutral": 0.75},
        corr_pref="high",
        directional=False,
        tags=["market-neutral", "relative-value", "pairs"],
    ),
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _corr_component(playbook: Playbook, mean_corr: float, has_corr_warnings: bool) -> float:
    """Correlation-fit signal in [0, 1] based on the playbook's preference."""
    scaled = _clamp(mean_corr / _CORR_REFERENCE)
    if playbook.corr_pref == "high":
        base = scaled
        if has_corr_warnings:
            base = _clamp(base + 0.2)  # explicit warned pairs help pairs trades
        return base
    if playbook.corr_pref == "low":
        return _clamp(1.0 - scaled)
    return 0.5


def score_playbook(
    playbook: Playbook,
    regime: str,
    bias: str,
    bias_strength: float,
    mean_corr: float,
    has_corr_warnings: bool = False,
) -> tuple[float, str]:
    """Score one playbook against the environment; return (score, reason).

    Directional playbooks have their bias contribution scaled by how strong the
    macro bias is, so a weak/uncertain macro read does not over-favour trend or
    momentum trades.
    """
    regime_w = playbook.regime_fit.get(regime, 0.4)
    bias_w = playbook.bias_fit.get(bias, 0.4)

    if playbook.directional and bias in ("bullish", "bearish"):
        # Blend the directional fit toward its neutral value when conviction is low.
        neutral_w = playbook.bias_fit.get("neutral", bias_w)
        bias_w = neutral_w + (bias_w - neutral_w) * _clamp(bias_strength)

    corr_w = _corr_component(playbook, mean_corr, has_corr_warnings)

    score = _clamp(_W_REGIME * regime_w + _W_BIAS * bias_w + _W_CORR * corr_w)
    reason = (
        f"regime[{regime}]={regime_w:.2f}, bias[{bias}]={bias_w:.2f}, "
        f"corr({playbook.corr_pref})={corr_w:.2f}"
    )
    return score, reason


def rank_playbooks(
    regime: str,
    bias: str,
    bias_strength: float,
    mean_corr: float,
    has_corr_warnings: bool = False,
) -> list[tuple[Playbook, float, str]]:
    """Return all playbooks scored and sorted best-first."""
    scored = [
        (pb, *score_playbook(pb, regime, bias, bias_strength, mean_corr, has_corr_warnings))
        for pb in PLAYBOOKS.values()
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
