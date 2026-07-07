"""
Smoke-test the end-to-end Analyst -> Risk pipeline (live network + Ollama).

This exercises the full Phase-1 flow:
    1. Analyst Agent runs RAG over Qdrant + Ollama to produce a MacroBriefing.
    2. Risk Agent consumes that briefing, pulls live market data via yfinance,
       computes deterministic risk metrics, and adds an Ollama narrative.

Prerequisites (run from anywhere after: cd mafas && pip install -e .):
    * Qdrant running and the corpus ingested   (docker compose up -d)
    * Ollama running with the model pulled      (ollama serve; ollama pull mistral)

Usage:
    python scripts/smoke_pipeline.py
    python scripts/smoke_pipeline.py --query "How is the Fed framing inflation?"
    python scripts/smoke_pipeline.py --tickers TSLA META --lookback 180
    python scripts/smoke_pipeline.py --no-llm      # deterministic, skips Ollama
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

DEFAULT_QUERY = "What is the Federal Reserve's current stance on interest rates and inflation?"


def ok(label: str, detail: str = "") -> None:
    msg = f"  [OK]   {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def warn(label: str, detail: str = "") -> None:
    msg = f"  [WARN] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def check_ollama(use_llm: bool) -> bool:
    """Report Ollama availability. Not fatal — the pipeline degrades gracefully."""
    print("\n--- Ollama LLM backend ---")
    if not use_llm:
        warn("skipped", "--no-llm set; agents will use deterministic fallbacks")
        return True
    try:
        from agents.llm import OllamaClient

        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "mistral")
        client = OllamaClient(url=url, model=model)
        if client.is_available():
            ok("is_available", f"model '{model}' ready at {url}")
            return True
        warn(
            "is_available",
            f"Ollama/model '{model}' not reachable — agents will fall back",
        )
        return True
    except Exception as exc:
        warn("OllamaClient", f"{type(exc).__name__}: {exc}")
        return True


def run_analyst(query: str, use_llm: bool):
    """Produce a MacroBriefing. Returns the briefing or None on hard failure."""
    print("\n--- Analyst Agent (RAG over Qdrant + Ollama) ---")
    try:
        from agents.analyst import build_analyst_agent

        agent = build_analyst_agent()
        if use_llm and not agent.llm.is_available():
            warn("llm", "Ollama unavailable — briefing will be low-confidence fallback")

        briefing = agent.brief(query)
        ok("brief", f"query='{query[:50]}...'")
        ok("confidence", f"{briefing.confidence:.0%}")
        ok("citations", f"{len(briefing.citations)} source(s)")
        ok("key_points", f"{len(briefing.key_points)}")
        if not briefing.citations:
            warn(
                "corpus",
                "0 citations — is Qdrant up and the corpus ingested? "
                "(docker compose up -d; python -m rag.corpus_builder)",
            )
        return briefing
    except Exception as exc:
        fail("AnalystAgent", f"{type(exc).__name__}: {exc}")
        return None


def run_risk(briefing, tickers: list[str], lookback: int, use_llm: bool):
    """Produce a RiskSummary from the briefing + live market data."""
    print("\n--- Risk Agent (yfinance metrics + Ollama narrative) ---")
    try:
        from agents.risk import build_risk_agent

        agent = build_risk_agent(with_llm=use_llm)
        agent.lookback_days = lookback
        summary = agent.assess(universe=tickers, briefing=briefing)

        if not summary.per_asset:
            fail("assess", "no market data retrieved for any ticker")
            return None

        ok("assess", f"{len(summary.per_asset)} asset(s) assessed")
        ok("vol_regime", f"{summary.vol_regime.upper()} (VIX {summary.vix_level:.1f})")
        ok(
            "concentration",
            f"mean_corr={summary.concentration.mean_pairwise_correlation:.2f}, "
            f"eff_bets={summary.concentration.effective_number_of_bets:.2f}",
        )
        ok("correlation_warnings", f"{len(summary.correlation_warnings)}")
        ok("position_sizing", f"{len(summary.position_sizing)} constraint(s)")
        ok("llm_used", str(summary.llm_used))
        if briefing is not None and summary.analyst_query is None:
            warn("handoff", "analyst context not threaded into risk summary")
        return summary
    except Exception as exc:
        fail("RiskAgent", f"{type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MAFAS end-to-end Analyst -> Risk pipeline smoke test."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Macro question for the Analyst.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=[],
        help="Extra tickers on top of the default watchlist.",
    )
    parser.add_argument("--lookback", type=int, default=252, help="Trading-day history window.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Ollama; use deterministic fallbacks throughout.",
    )
    parser.add_argument(
        "--show-reports",
        action="store_true",
        help="Print the full rendered briefing and risk summary.",
    )
    args = parser.parse_args()
    use_llm = not args.no_llm

    print("MAFAS Phase-1 pipeline smoke test (requires internet)\n")
    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"LLM:    {'enabled' if use_llm else 'disabled (--no-llm)'}")

    check_ollama(use_llm)
    briefing = run_analyst(args.query, use_llm)
    summary = run_risk(briefing, args.tickers, args.lookback, use_llm)

    if args.show_reports:
        if briefing is not None:
            print("\n" + briefing.render())
        if summary is not None:
            print("\n" + summary.render())

    # The pipeline "passes" if the Risk Agent produced a usable summary; the
    # Analyst may legitimately return an empty briefing if the corpus is bare.
    analyst_ok = briefing is not None
    risk_ok = summary is not None and bool(summary.per_asset)
    passed = int(analyst_ok) + int(risk_ok)
    print(f"\n=== Result: {passed}/2 stages passed "
          f"(analyst={'ok' if analyst_ok else 'fail'}, "
          f"risk={'ok' if risk_ok else 'fail'}) ===\n")
    return 0 if passed == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
