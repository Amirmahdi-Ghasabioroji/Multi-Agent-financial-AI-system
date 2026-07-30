"""Dashboard-facing corpus refresh tests (all external services mocked)."""

from rag import corpus_builder as module
from rag.corpus_builder import CorpusBuildResult


class FakeRetriever:
    def __init__(self, *args, **kwargs) -> None:
        self.points = 0

    def upsert(self, chunks: list[dict]) -> None:
        self.points += len(chunks)

    def count(self) -> int:
        return self.points


class FakeChunker:
    def chunk_document(self, text: str, metadata: dict) -> list[dict]:
        return [{"text": text, **metadata}]


class FakeFOMC:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def load_recent(self, n: int) -> list[dict]:
        return [{"text": "fomc", "metadata": {"doc_type": "fomc"}}]


class FakeEDGAR:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def load_recent_for_ticker(self, ticker: str, limit: int) -> list[dict]:
        return [{"text": ticker, "metadata": {"ticker": ticker}}]


class FakeNews:
    def load_all(self, max_per_feed: int) -> list[dict]:
        return [{"text": "news", "metadata": {"doc_type": "news"}}]


def test_build_initial_corpus_returns_result_and_emits_progress(monkeypatch) -> None:
    monkeypatch.setattr(module, "TextEmbedder", lambda: object())
    monkeypatch.setattr(module, "VectorRetriever", FakeRetriever)
    monkeypatch.setattr(module, "SemanticChunker", FakeChunker)
    monkeypatch.setattr(module, "FOMCLoader", FakeFOMC)
    monkeypatch.setattr(module, "EDGARLoader", FakeEDGAR)
    monkeypatch.setattr(module, "NewsLoader", FakeNews)
    events: list[tuple[str, dict]] = []

    result = module.build_initial_corpus(
        n_fomc=1,
        tickers=["AAPL", "MSFT"],
        max_news_per_feed=1,
        reset=True,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert isinstance(result, CorpusBuildResult)
    assert result.documents_ingested == 4
    assert result.chunks_created == 4
    assert result.vectors_in_collection == 4
    assert result.reset is True
    assert events[0][0] == "stage_started"
    assert events[-1][0] == "corpus_completed"
