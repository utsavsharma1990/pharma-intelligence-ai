"""
Unit tests for ChromaStore.

We use deterministic fake embeddings (not real model output) so tests are:
  - Fast (no embedding model load)
  - Deterministic (same input → same output, no GPU/CPU drift)
  - Network-free (no model download on CI)
"""

from pathlib import Path
import pytest

from src.core.chroma_store import ChromaStore
from src.ingestion.models import TrialChunk


def _fake_embedding(seed: int, dim: int = 16) -> list[float]:
    """Generate a deterministic vector — close to itself, far from others."""
    return [(seed + i) * 0.01 for i in range(dim)]


def _make_chunks() -> tuple[list[TrialChunk], list[list[float]]]:
    chunks = [
        TrialChunk(
            chunk_id="NCT001::overview::0",
            nct_id="NCT001",
            section_type="overview",
            content="Phase 3 study of pembrolizumab in NSCLC",
            phase="PHASE3",
            sponsor="Merck",
            overall_status="RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT001::adverse_events::0",
            nct_id="NCT001",
            section_type="adverse_events",
            content="Pneumonitis 3%, colitis 2%",
            phase="PHASE3",
            sponsor="Merck",
            overall_status="RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT002::overview::0",
            nct_id="NCT002",
            section_type="overview",
            content="Phase 2 ibrutinib in CLL",
            phase="PHASE2",
            sponsor="AbbVie",
            overall_status="ACTIVE_NOT_RECRUITING",
        ),
    ]
    embeddings = [_fake_embedding(i) for i in range(len(chunks))]
    return chunks, embeddings


@pytest.fixture
def store(tmp_path) -> ChromaStore:
    return ChromaStore(persist_dir=tmp_path, collection_name="test_collection")


# ---------- Add + count ----------

def test_add_chunks_increases_count(store):
    chunks, embeddings = _make_chunks()
    assert store.count() == 0
    store.add_chunks(chunks, embeddings)
    assert store.count() == 3


def test_add_chunks_empty_is_noop(store):
    store.add_chunks([], [])
    assert store.count() == 0


def test_add_chunks_mismatched_lengths_raises(store):
    chunks, _ = _make_chunks()
    with pytest.raises(ValueError, match="must have the same length"):
        store.add_chunks(chunks, [_fake_embedding(0)])  # only 1 embedding for 3 chunks


def test_add_chunks_upserts_duplicates(store):
    """Re-adding the same chunk_id should not crash; it should overwrite."""
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)
    store.add_chunks(chunks, embeddings)  # same ids again
    assert store.count() == 3  # still 3, not 6


# ---------- Search ----------

def test_search_returns_top_k(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)

    # Query with same embedding as chunk 0 → should hit chunk 0 first
    hits = store.search(query_embedding=embeddings[0], top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "NCT001::overview::0"
    assert hits[0].score > hits[1].score  # ordered by similarity


def test_search_score_in_range(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)
    hits = store.search(query_embedding=embeddings[0], top_k=3)
    for h in hits:
        assert 0.0 <= h.score <= 1.0


def test_search_with_metadata_filter(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)

    # Filter to only AE chunks — should return exactly one (chunk 1)
    hits = store.search(
        query_embedding=embeddings[0],
        top_k=10,
        filters={"section_type": "adverse_events"},
    )
    assert len(hits) == 1
    assert hits[0].chunk.section_type == "adverse_events"


def test_search_with_multi_field_filter(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)

    # Phase 3 + Merck → only NCT001 chunks (2 of them)
    hits = store.search(
        query_embedding=embeddings[0],
        top_k=10,
        filters={"phase": "PHASE3", "sponsor": "Merck"},
    )
    assert len(hits) == 2
    for h in hits:
        assert h.chunk.nct_id == "NCT001"


def test_search_filter_with_no_matches(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)
    hits = store.search(
        query_embedding=embeddings[0],
        top_k=5,
        filters={"phase": "PHASE4"},  # nothing matches
    )
    assert hits == []


# ---------- Delete ----------

def test_delete_by_nct_id_removes_all_chunks(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)
    n = store.delete_by_nct_id("NCT001")
    assert n == 2
    assert store.count() == 1


def test_delete_by_unknown_nct_id_returns_zero(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)
    n = store.delete_by_nct_id("NCT_NONEXISTENT")
    assert n == 0
    assert store.count() == 3


# ---------- Reset ----------

def test_reset_wipes_collection(store):
    chunks, embeddings = _make_chunks()
    store.add_chunks(chunks, embeddings)
    store.reset()
    assert store.count() == 0


# ---------- Persistence (the whole point of ChromaDB) ----------

def test_data_persists_across_instances(tmp_path):
    """Closing and reopening should preserve data — this is what 'persistent' means."""
    chunks, embeddings = _make_chunks()

    store1 = ChromaStore(persist_dir=tmp_path, collection_name="persist_test")
    store1.add_chunks(chunks, embeddings)
    assert store1.count() == 3

    # Simulate process restart: new instance, same path
    store2 = ChromaStore(persist_dir=tmp_path, collection_name="persist_test")
    assert store2.count() == 3