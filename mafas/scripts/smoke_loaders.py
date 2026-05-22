"""
Smoke-test all data loaders (live network required).

Run from anywhere after: cd mafas && pip install -e .

    python scripts/smoke_loaders.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def ok(label: str, detail: str = "") -> None:
    msg = f"  [OK]   {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def check_news() -> bool:
    print("\n--- NewsLoader (Reuters RSS) ---")
    try:
        from data.loaders.news import NewsLoader

        articles = NewsLoader().load_all(max_per_feed=3)
        if not articles:
            fail("load_all", "0 articles (RSS blocked or feeds down)")
            return False
        a = articles[0]
        ok("load_all", f"{len(articles)} articles")
        ok("sample", f"{a['metadata'].get('title', '')[:60]}...")
        ok("word_count", str(a["metadata"].get("word_count")))
        return True
    except Exception as exc:
        fail("NewsLoader", str(exc))
        return False


def check_fomc() -> bool:
    print("\n--- FOMCLoader (Federal Reserve PDFs) ---")
    try:
        from data.loaders.fomc import FOMCLoader

        loader = FOMCLoader(cache_dir=os.getenv("CACHE_DIR", "./data/cache"))
        links = loader.fetch_minutes_page()
        if not links:
            fail("fetch_minutes_page", "0 links (Fed site unreachable?)")
            return False
        ok("fetch_minutes_page", f"{len(links)} minutes URLs")

        doc = loader.load_minutes_pdf(links[0]["url"])
        if not doc:
            fail("load_minutes_pdf", f"could not load {links[0]['url']}")
            return False
        ok("load_minutes_pdf", f"words={doc['metadata'].get('word_count')}")
        return True
    except Exception as exc:
        fail("FOMCLoader", str(exc))
        return False


def check_edgar() -> bool:
    print("\n--- EDGARLoader (SEC filings) ---")
    ticker = "AAPL"
    try:
        from data.loaders.edgar import EDGARLoader

        loader = EDGARLoader(cache_dir=os.getenv("CACHE_DIR", "./data/cache"))
        filings = loader.get_recent_filings(ticker, form_type="10-Q", limit=3)
        if not filings:
            fail("get_recent_filings", f"0 filings for {ticker}")
            return False
        ok("get_recent_filings", f"{len(filings)} filing(s)")

        doc = None
        for filing in filings:
            doc = loader.load_filing_text(ticker, filing["accession_number"])
            if doc:
                ok("load_filing_text", f"{filing['accession_number']} words={doc['metadata'].get('word_count')}")
                return True
        fail("load_filing_text", "no filing returned enough text (tried all)")
        return False
    except Exception as exc:
        fail("EDGARLoader", str(exc))
        return False


def check_market() -> bool:
    print("\n--- MarketDataLoader (yfinance + FRED) ---")
    try:
        from data.loaders.market import MarketDataLoader

        key = os.getenv("FRED_API_KEY", "")
        loader = MarketDataLoader(
            fred_api_key=key,
            cache_dir=os.getenv("CACHE_DIR", "./data/cache"),
        )

        df = loader.get_ohlcv("SPY", days=30)
        if df.empty:
            fail("get_ohlcv", "empty DataFrame")
            return False
        cols = {str(c).lower(): c for c in df.columns}
        price_col = cols.get("close") or cols.get("adj close") or df.columns[0]
        last_price = float(df[price_col].iloc[-1])
        ok("get_ohlcv", f"{len(df)} rows, last {price_col} ~ {last_price:.2f}")

        vix = loader.get_vix()
        ok("get_vix", f"{vix:.2f}")

        series = loader.get_fred_series("UNRATE", days=90)
        if series.empty:
            print(
                "  [WARN] get_fred_series empty — set FRED_API_KEY in .env "
                "(OHLCV/VIX still OK)"
            )
            return True
        ok("get_fred_series", f"{len(series)} points for UNRATE")
        return True
    except Exception as exc:
        fail("MarketDataLoader", str(exc))
        return False


def main() -> int:
    print("MAFAS loader smoke test (requires internet)\n")
    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")

    results = [
        check_news(),
        check_fomc(),
        check_edgar(),
        check_market(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n=== Result: {passed}/{total} loaders passed ===\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
