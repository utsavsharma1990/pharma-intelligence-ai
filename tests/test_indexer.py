"""
Tests for the embedding provider and the indexing pipeline.

For unit tests we use a FAKE embedder (deterministic, zero-cost) so tests
don't download a 22MB model and don't run torch at import time.
"""

import pytest
from pathlib import Path

from src.core.chroma_store import ChromaStore
from src.core.embeddings import EmbeddingProvider
from src.core.indexer import TrialIndexer
from src.ingestion.fetcher import TrialFetcher


# ---------------------------------------------------------------------------
# Fake embedder — same shape as HuggingFaceEmbeddings, zero dependencies.
# ---------------------------------------------------------------------------

class FakeEmbedder(EmbeddingProvider):
    """
    Deterministic fake embedder for testing.
    Hashes each input character into a fixed-dim vector. Same input → same vector.
    """
    DIM = 16

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            s = sum(ord(c) for c in t)
            out.append([((s + i) % 100) * 0.01 for i in range(self.DIM)])
        return out

    @property
    def dimension(self) -> int:
        return self.DIM


# ---------------------------------------------------------------------------
# Indexer tests
# ---------------------------------------------------------------------------

@pytest.fixture
def indexer(tmp_path):
    fetcher  = TrialFetcher(use_mock=True, cache_dir=tmp_path / "cache")
    embedder = FakeEmbedder()
    store    = ChromaStore(persist_dir=tmp_path / "chroma", collection_name="test")
    return TrialIndexer(fetcher=fetcher, embedder=embedder, store=store), store


def test_indexer_processes_all_mock_trials(indexer):
    idxr, store = indexer
    stats = idxr.index()
    assert stats.trials_processed == 3
    assert stats.chunks_indexed > 0
    assert store.count() == stats.chunks_indexed


def test_indexer_respects_max_studies(indexer):
    idxr, store = indexer
    stats = idxr.index(max_studies=1)
    assert stats.trials_processed == 1


def test_indexer_idempotent(indexer):
    """Running twice should produce the same final state — re-ingestion is safe."""
    idxr, store = indexer
    stats_1 = idxr.index()
    count_1 = store.count()

    stats_2 = idxr.index()
    count_2 = store.count()

    assert count_1 == count_2, "Running twice changed the index size"
    assert stats_2.chunks_deleted == count_1, "Should have deleted old chunks before re-inserting"


def test_indexer_deletes_old_chunks_on_reingest(indexer):
    idxr, store = indexer
    idxr.index()
    stats = idxr.index()  # second run
    # Every trial's old chunks should have been wiped
    assert stats.chunks_deleted > 0


def test_indexer_uses_correct_embedding_dim(indexer):
    idxr, store = indexer
    idxr.index(max_studies=1)
    # Sanity: the store has chunks, and we can search them back
    query_vec = FakeEmbedder().embed_text("anything")
    hits = store.search(query_embedding=query_vec, top_k=3)
    assert len(hits) > 0