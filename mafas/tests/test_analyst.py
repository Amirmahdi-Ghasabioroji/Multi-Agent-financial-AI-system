"""Unit tests for the Analyst Agent (mocked LLM and retriever, no network)."""

from datetime import datetime, timezone

import pytest

from agents.analyst import AnalystAgent
from agents.confidence import (
    composite_confidence,
    recency_factor,
    retrieval_confidence,
    source_diversity,
)
from agents.schemas import MacroBriefing


class FakeRetriever:
    """Stand-in for VectorRetriever returning canned hits."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def search(self, query, top_k=8, doc_type=None, date_after=None, score_threshold=0.4):
        return self._hits


class FakeLLM:
    """Stand-in for OllamaClient returning a canned parsed JSON object."""

    model = "mistral"

    def __init__(self, response: dict | None = None, raise_error: bool = False) -> None:
        self._response = response or {}
        self._raise = raise_error
        self.last_messages = []

    def chat_json(self, messages, options=None):
        self.last_messages = messages
        if self._raise:
            from agents.llm import OllamaError

            raise OllamaError("simulated outage")
        return self._response


@pytest.fixture
def sample_hits() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [
        {
            "text": "The FOMC held rates steady amid moderating inflation.",
            "score": 0.72,
            "source": "https://federalreserve.gov/fomcminutes20240131.htm",
            "date": today,
            "doc_type": "fomc_minutes",
            "chunk_index": 0,
        },
        {
            "text": "Apple reported resilient services revenue in its latest 10-Q.",
            "score": 0.65,
            "source": "https://sec.gov/aapl-10q.htm",
            "date": today,
            "doc_type": "sec_filing",
            "chunk_index": 2,
        },
        {
            "text": "Markets expect fewer rate cuts this year, lifting yields.",
            "score": 0.6,
            "source": "https://news.example.com/markets",
            "date": today,
            "doc_type": "news_article",
            "chunk_index": 1,
        },
    ]


class TestConfidence:
    def test_retrieval_confidence_mean(self) -> None:
        assert retrieval_confidence([0.5, 0.7]) == pytest.approx(0.6)

    def test_retrieval_confidence_empty(self) -> None:
        assert retrieval_confidence([]) == 0.0

    def test_source_diversity_rewards_distinct_types(self) -> None:
        low = source_diversity(["a"], ["news_article"])
        high = source_diversity(
            ["a", "b", "c"], ["fomc_minutes", "sec_filing", "news_article"]
        )
        assert high > low

    def test_recency_recent_is_high(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert recency_factor([today]) == 1.0

    def test_recency_unknown_is_neutral(self) -> None:
        assert recency_factor(["unknown"]) == 0.5

    def test_composite_in_range(self) -> None:
        score, breakdown = composite_confidence(
            scores=[0.7, 0.6],
            sources=["a", "b"],
            doc_types=["fomc_minutes", "news_article"],
            dates=[datetime.now(timezone.utc).strftime("%Y-%m-%d")],
            llm_self_confidence=0.8,
        )
        assert 0.0 <= score <= 1.0
        assert set(breakdown) == {"retrieval", "diversity", "recency", "llm_self"}


class TestAnalystAgent:
    def test_brief_assembles_full_briefing(self, sample_hits) -> None:
        llm_response = {
            "summary": "The Fed is on hold while inflation cools [1].",
            "key_points": [
                {"point": "Rates steady [1]", "citations": [1], "confidence": 0.8},
                {"point": "Apple services strong [2]", "citations": [2], "confidence": 0.7},
            ],
            "risks": ["Sticky inflation could delay cuts [3]."],
            "self_confidence": 0.75,
        }
        agent = AnalystAgent(FakeRetriever(sample_hits), FakeLLM(llm_response))
        briefing = agent.brief("What is the Fed's stance?")

        assert isinstance(briefing, MacroBriefing)
        assert briefing.summary.startswith("The Fed is on hold")
        assert len(briefing.key_points) == 2
        assert len(briefing.citations) == 3
        assert briefing.llm_self_confidence == 0.75
        assert 0.0 < briefing.confidence <= 1.0

    def test_brief_handles_no_hits(self) -> None:
        agent = AnalystAgent(FakeRetriever([]), FakeLLM({}))
        briefing = agent.brief("Obscure query with no evidence")
        assert briefing.confidence == 0.0
        assert briefing.citations == []
        assert "Insufficient evidence" in briefing.summary

    def test_brief_handles_llm_outage(self, sample_hits) -> None:
        agent = AnalystAgent(FakeRetriever(sample_hits), FakeLLM(raise_error=True))
        briefing = agent.brief("What is the Fed's stance?")
        assert briefing.confidence == 0.0
        # Citations are still attached even when the LLM fails.
        assert len(briefing.citations) == 3

    def test_citations_have_excerpts(self, sample_hits) -> None:
        agent = AnalystAgent(FakeRetriever(sample_hits), FakeLLM({"summary": "x"}))
        briefing = agent.brief("test")
        for citation in briefing.citations:
            assert citation.excerpt
            assert citation.index >= 1

    def test_conversation_context_is_bounded_and_separate_from_retrieval(
        self, sample_hits
    ) -> None:
        llm = FakeLLM({"summary": "Follow-up answer [1]."})
        retriever = FakeRetriever(sample_hits)
        agent = AnalystAgent(retriever, llm)

        agent.brief(
            "What changed?",
            conversation_context=("old context " * 1_000) + "LATEST TURN",
        )

        prompt = llm.last_messages[-1]["content"]
        assert "<conversation_context>" in prompt
        assert "LATEST TURN" in prompt
        assert len(prompt) < 20_000

    def test_render_produces_report(self, sample_hits) -> None:
        agent = AnalystAgent(FakeRetriever(sample_hits), FakeLLM({"summary": "Hold."}))
        briefing = agent.brief("test")
        report = briefing.render()
        assert "MACRO BRIEFING" in report
        assert "SOURCES" in report
