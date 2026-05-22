"""Prerequisite unit tests for MAFAS data and RAG foundation."""

from data.processors.cleaner import TextCleaner
from data.processors.metadata import MetadataExtractor
from rag.chunker import SemanticChunker
from rag.embedder import TextEmbedder


class TestTextCleaner:
    def test_clean_collapses_multiple_newlines(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("line1\n\n\n\nline2") == "line1\n\nline2"

    def test_clean_collapses_multiple_spaces(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.clean("word1   word2") == "word1 word2"

    def test_clean_html_strips_script_and_nav(self) -> None:
        cleaner = TextCleaner()
        html = """
        <html><body>
        <nav>Skip this</nav>
        <script>alert('x')</script>
        <p>Visible policy text about rates.</p>
        </body></html>
        """
        result = cleaner.clean_html(html)
        assert "alert" not in result
        assert "Skip this" not in result
        assert "Visible policy text" in result

    def test_is_meaningful_short_and_long(self) -> None:
        cleaner = TextCleaner()
        assert cleaner.is_meaningful("too short", min_words=50) is False
        long_text = " ".join(["word"] * 60)
        assert cleaner.is_meaningful(long_text, min_words=50) is True


class TestSemanticChunker:
    def test_chunk_multiple_for_long_document(self) -> None:
        chunker = SemanticChunker(max_tokens=100, overlap_tokens=10)
        paragraph = " ".join(["economics"] * 80)
        text = "\n\n".join([paragraph] * 20)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_single_for_short_text(self) -> None:
        chunker = SemanticChunker()
        text = "A brief note on central bank policy and inflation targets."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_chunk_document_has_chunk_index(self, sample_metadata: dict) -> None:
        chunker = SemanticChunker(max_tokens=50, overlap_tokens=5)
        paragraph = " ".join(["markets"] * 40)
        text = "\n\n".join([paragraph] * 5)
        chunks = chunker.chunk_document(text, sample_metadata)
        assert len(chunks) >= 1
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i


class TestMetadataExtractor:
    def test_date_extraction_january_2024(self) -> None:
        extractor = MetadataExtractor()
        text = "The committee met in January 2024 to review policy."
        meta = extractor.extract(text, source="test", doc_type="fomc_minutes")
        assert "January 2024" in meta.date

    def test_date_unknown_when_missing(self) -> None:
        extractor = MetadataExtractor()
        text = "No temporal references in this document at all."
        meta = extractor.extract(text, source="test", doc_type="news_article")
        assert meta.date == "unknown"

    def test_word_count_populated(self) -> None:
        extractor = MetadataExtractor()
        text = "one two three four five"
        meta = extractor.extract(text, source="test", doc_type="news_article")
        assert meta.word_count == 5


class TestTextEmbedder:
    def test_embed_returns_correct_dimension(self) -> None:
        embedder = TextEmbedder()
        vector = embedder.embed("Federal Reserve policy rate decision")
        assert isinstance(vector, list)
        assert all(isinstance(v, float) for v in vector)
        assert len(vector) == embedder.dim

    def test_embed_batch_count(self) -> None:
        embedder = TextEmbedder()
        texts = ["inflation outlook", "labor market conditions", "Treasury yields"]
        vectors = embedder.embed_batch(texts, batch_size=2)
        assert len(vectors) == 3
        assert all(len(v) == embedder.dim for v in vectors)
