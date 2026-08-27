"""Pytest fixtures for MAFAS prerequisite tests."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MAFAS_EVAL_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text() -> str:
    """200-word paragraph of realistic financial text."""
    return (
        "The Federal Reserve held its policy rate steady at the September meeting "
        "as inflation continued to moderate toward the two percent target. "
        "Chair Powell noted that labor market conditions remain solid, though "
        "hiring has cooled from the pace seen in prior years. "
        "Participants discussed the appropriate path for balance sheet runoff "
        "and the implications of tighter financial conditions for credit "
        "availability to households and businesses. "
        "Several members emphasized data dependence and the need to avoid "
        "premature easing that could re-ignite price pressures. "
        "Markets priced in fewer cuts than at the start of the year, "
        "pushing Treasury yields higher across the curve. "
        "Equity valuations reflected optimism about artificial intelligence "
        "investment while remaining sensitive to macro surprises. "
        "International developments, including energy prices and exchange rates, "
        "were cited as upside risks to the inflation outlook. "
        "Banking sector resilience was judged adequate, with supervisors "
        "monitoring commercial real estate exposures. "
        "The statement retained language noting elevated uncertainty and "
        "the Committee's commitment to maximum employment and price stability. "
        "Analysts expect the next decision in January 2024 to hinge on "
        "upcoming payrolls and consumer spending data releases."
    )


@pytest.fixture
def sample_metadata() -> dict:
    """Valid metadata dict matching DocumentMetadata fields."""
    return {
        "source": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm",
        "doc_type": "fomc_minutes",
        "title": "FOMC Minutes — January 2024",
        "date": "2024-01-31",
        "tickers": [],
        "word_count": 1850,
    }


@pytest.fixture
def mock_chunks(sample_metadata: dict) -> list[dict]:
    """Three chunk dicts for retriever/ingestion tests."""
    return [
        {
            "text": "The Federal Reserve held its policy rate steady.",
            "metadata": {**sample_metadata, "chunk_index": 0},
        },
        {
            "text": "Participants discussed balance sheet runoff implications.",
            "metadata": {**sample_metadata, "chunk_index": 1},
        },
        {
            "text": "Markets priced in fewer cuts than earlier in the year.",
            "metadata": {**sample_metadata, "chunk_index": 2},
        },
    ]
