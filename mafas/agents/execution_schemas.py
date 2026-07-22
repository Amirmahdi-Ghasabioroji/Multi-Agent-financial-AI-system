"""Pydantic models for the Execution Agent's simulated trade card.

Structured like the other agents' outputs. A TradeCard stress-tests one
StrategySetup against historical price behaviour — it is a simulation result,
not an order.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class TradeLevels(BaseModel):
    """Entry / stop / target price geometry for the simulated trade."""

    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    risk_per_unit: float = Field(0.0, description="|entry - stop| price distance")
    planned_rr: float = Field(0.0, description="Reward:risk from the levels")


class SimulationStats(BaseModel):
    """Aggregated Monte Carlo barrier-simulation outcomes."""

    n_sims: int = 0
    horizon_bars: int = 0
    prob_tp_before_sl: float = 0.0
    prob_sl_before_tp: float = 0.0
    prob_timeout: float = 0.0
    expected_r: float = Field(0.0, description="Mean realised R across paths")
    win_rate: float = 0.0
    avg_bars_to_exit: float = 0.0
    mae_mean_r: float = Field(0.0, description="Mean max adverse excursion in R")
    mae_p95_r: float = Field(0.0, description="95th-pct max adverse excursion in R")
    seed: int = 0


class TradeRecord(BaseModel):
    """One completed trade from the historical backtest."""

    entry_date: str = ""
    exit_date: str = ""
    direction: str = "long"
    entry_price: float = 0.0
    exit_price: float = 0.0
    outcome: str = ""  # tp | sl | timeout
    pnl_r: float = 0.0
    pnl_amount: float = 0.0
    bars_held: int = 0


class BacktestMetrics(BaseModel):
    """Professional performance statistics from historical replay."""

    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    avg_bars_held: float = 0.0
    low_sample: bool = False


class BacktestResult(BaseModel):
    """Historical backtest output for one strategy setup."""

    period_start: str = ""
    period_end: str = ""
    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    trades: list[TradeRecord] = Field(default_factory=list)
    equity_curve: list[float] = Field(default_factory=list)
    drawdown_curve: list[float] = Field(default_factory=list)
    mc_robustness: dict[str, float] | None = None


class ExecutionComparisonEntry(BaseModel):
    """One row in the cross-strategy ranking table."""

    rank: int = 0
    instrument: str = ""
    strategy: str = ""
    composite_score: float = 0.0
    sharpe_ratio: float | None = None
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    expected_r_forward: float = 0.0


class ExecutionComparison(BaseModel):
    """Ranked comparison across simulated strategy setups."""

    ranked: list[ExecutionComparisonEntry] = Field(default_factory=list)
    best_sharpe: str | None = None
    best_pnl: str | None = None
    lowest_drawdown: str | None = None


class SizingInfo(BaseModel):
    """Risk-based position sizing derived from Risk Agent constraints."""

    account_equity: float = 0.0
    units: float = 0.0
    notional: float = 0.0
    notional_pct: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    max_position_pct: float = 0.0
    capped: bool = False


class TradeCard(BaseModel):
    """A simulated trade card for one strategy setup."""

    instrument: str | None = None
    strategy: str = ""
    strategy_name: str = ""
    direction: str = "long"
    horizon: str = "swing"

    simulated: bool = Field(True, description="False when the setup could not be simulated")
    skip_reason: str = ""

    levels: TradeLevels = Field(default_factory=TradeLevels)
    stats: SimulationStats = Field(default_factory=SimulationStats)
    backtest: BacktestResult | None = None
    sizing: SizingInfo = Field(default_factory=SizingInfo)

    expectancy_amount: float = Field(0.0, description="Expected P/L in account currency")
    data_source: str = Field("", description="twelvedata | yfinance")
    bars_used: int = 0

    # Context inherited from upstream agents.
    strategy_confidence: float | None = None
    playbook_fit: float | None = None

    verdict: str = Field("", description="LLM (or fallback) qualitative read of the card")
    llm_used: bool = False
    model: str = "mistral"
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def render(self) -> str:
        """Format the trade card as a human-readable report."""
        lines: list[str] = []
        lines.append("=" * 70)
        title = self.instrument or "—"
        lines.append(f"TRADE CARD — {self.strategy_name} | {title} {self.direction.upper()}")
        lines.append("=" * 70)

        if not self.simulated:
            lines.append(f"Not simulated: {self.skip_reason}")
            return "\n".join(lines)

        lv, st, sz = self.levels, self.stats, self.sizing
        lines.append(
            f"Data: {self.data_source} ({self.bars_used} bars)   "
            f"Horizon: {self.horizon} ({st.horizon_bars} bars)   "
            f"Sims: {st.n_sims}"
        )
        lines.append("")
        lines.append("LEVELS")
        lines.append("-" * 70)
        lines.append(
            f"entry={lv.entry:.2f}  stop={lv.stop_loss:.2f}  target={lv.take_profit:.2f}  "
            f"ATR={lv.atr:.2f}  risk/unit={lv.risk_per_unit:.2f}  planned R:R={lv.planned_rr:.2f}"
        )
        lines.append("")
        lines.append("SIMULATION")
        lines.append("-" * 70)
        lines.append(
            f"P(TP before SL)={st.prob_tp_before_sl:.0%}   "
            f"P(SL first)={st.prob_sl_before_tp:.0%}   "
            f"P(timeout)={st.prob_timeout:.0%}"
        )
        lines.append(
            f"expected R={st.expected_r:+.2f}   win rate={st.win_rate:.0%}   "
            f"avg bars to exit={st.avg_bars_to_exit:.1f}"
        )
        lines.append(
            f"MAE: mean={st.mae_mean_r:.2f}R  p95={st.mae_p95_r:.2f}R"
        )
        if self.backtest is not None:
            bt = self.backtest
            m = bt.metrics
            lines.append("")
            lines.append("BACKTEST")
            lines.append("-" * 70)
            lines.append(f"period: {bt.period_start} → {bt.period_end}   trades: {m.n_trades}")
            lines.append(
                f"total P/L=${m.total_pnl:+,.0f} ({m.total_return_pct:.1%})   "
                f"max DD={m.max_drawdown_pct:.1%} (${m.max_drawdown_amount:,.0f})"
            )
            sharpe = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "n/a"
            sortino = f"{m.sortino_ratio:.2f}" if m.sortino_ratio is not None else "n/a"
            calmar = f"{m.calmar_ratio:.2f}" if m.calmar_ratio is not None else "n/a"
            lines.append(
                f"Sharpe={sharpe}  Sortino={sortino}  Calmar={calmar}  "
                f"profit factor={m.profit_factor:.2f}  expectancy={m.expectancy_r:+.2f}R"
            )
            if m.low_sample:
                lines.append("(low sample — fewer trades than minimum for robust metrics)")
        lines.append("")
        lines.append("SIZING")
        lines.append("-" * 70)
        lines.append(
            f"equity=${sz.account_equity:,.0f}  units={sz.units:g}  "
            f"notional=${sz.notional:,.0f} ({sz.notional_pct:.1%})  "
            f"risk=${sz.risk_amount:,.0f} ({sz.risk_pct:.2%})"
            + ("  [CAPPED]" if sz.capped else "")
        )
        lines.append(f"expected P/L=${self.expectancy_amount:+,.0f}")

        if self.verdict:
            lines.append("")
            lines.append("VERDICT")
            lines.append("-" * 70)
            lines.append(self.verdict)
        return "\n".join(lines)
