"""Tests for the Retriever — uses the FakeEmbedder + ChromaStore from earlier tests."""

import pytest
from pathlib import Path

from src.core.chroma_store import ChromaStore
from src.core.embeddings import EmbeddingProvider
from src.core.retriever import Retriever, RetrievalQuery
from src.ingestion.models import TrialChunk


# Reuse the deterministic fake embedder pattern from test_indexer.py
class FakeEmbedder(EmbeddingProvider):
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
# Test fixtures
# ---------------------------------------------------------------------------

def _seed_store(tmp_path) -> tuple[Retriever, ChromaStore]:
    """Build a populated retriever + store for testing."""
    chunks = [
        TrialChunk(chunk_id="NCT001::overview::0",       nct_id="NCT001",
                   section_type="overview",       content="Phase 3 NSCLC pembrolizumab",
                   phase="PHASE3", sponsor="Merck",   overall_status="RECRUITING"),
        TrialChunk(chunk_id="NCT001::adverse_events::0", nct_id="NCT001",
                   section_type="adverse_events", content="Pneumonitis colitis hepatitis",
                   phase="PHASE3", sponsor="Merck",   overall_status="RECRUITING"),
        TrialChunk(chunk_id="NCT001::eligibility::0",    nct_id="NCT001",
                   section_type="eligibility",    content="Adults 18+ ECOG 0-1",
                   phase="PHASE3", sponsor="Merck",   overall_status="RECRUITING"),
        TrialChunk(chunk_id="NCT002::overview::0",       nct_id="NCT002",
                   section_type="overview",       content="Phase 2 CLL ibrutinib",
                   phase="PHASE2", sponsor="AbbVie",  overall_status="ACTIVE_NOT_RECRUITING"),
        TrialChunk(chunk_id="NCT002::adverse_events::0", nct_id="NCT002",
                   section_type="adverse_events", content="Atrial fibrillation bleeding",
                   phase="PHASE2", sponsor="AbbVie",  overall_status="ACTIVE_NOT_RECRUITING"),
    ]
    embedder = FakeEmbedder()
    store    = ChromaStore(persist_dir=tmp_path, collection_name="ret_test")
    store.add_chunks(chunks, embedder.embed_texts([c.content for c in chunks]))
    return Retriever(embedder, store), store


@pytest.fixture
def retriever(tmp_path):
    r, _ = _seed_store(tmp_path)
    return r


# ---------------------------------------------------------------------------
# Basic search
# ---------------------------------------------------------------------------

def test_plain_search_returns_results(retriever):
    hits = retriever.search_text("lung cancer", top_k=3)
    assert len(hits) > 0
    assert len(hits) <= 3


def test_top_k_is_respected(retriever):
    hits = retriever.search_text("anything", top_k=2)
    assert len(hits) <= 2


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def test_filter_by_nct_id(retriever):
    hits = retriever.search(RetrievalQuery(text="anything", nct_id="NCT001", top_k=10))
    assert len(hits) == 3
    for h in hits:
        assert h.chunk.nct_id == "NCT001"


def test_filter_by_section_type(retriever):
    hits = retriever.search(RetrievalQuery(
        text="anything", section_type="adverse_events", top_k=10
    ))
    assert len(hits) == 2
    for h in hits:
        assert h.chunk.section_type == "adverse_events"


def test_combined_filters(retriever):
    hits = retriever.search(RetrievalQuery(
        text="anything", phase="PHASE3", sponsor="Merck", top_k=10,
    ))
    assert len(hits) == 3
    for h in hits:
        assert h.chunk.phase == "PHASE3"
        assert h.chunk.sponsor == "Merck"


def test_filter_with_no_matches(retriever):
    hits = retriever.search(RetrievalQuery(text="x", phase="PHASE4", top_k=10))
    assert hits == []


# ---------------------------------------------------------------------------
# Auto NCT extraction
# ---------------------------------------------------------------------------

def test_auto_extracts_nct_id_from_query_text(retriever):
    hits = retriever.search(RetrievalQuery(
        text="What are eligibility criteria for NCT001?",
        top_k=10,
    ))
    # All hits should be from NCT001 because we auto-filtered
    assert len(hits) > 0
    for h in hits:
        assert h.chunk.nct_id == "NCT001"


def test_auto_extracted_ids_recorded(retriever):
    q = RetrievalQuery(text="Compare NCT001 and NCT002 safety profiles")
    retriever.search(q)
    assert "NCT001" in q.extracted_nct_ids
    assert "NCT002" in q.extracted_nct_ids


def test_explicit_nct_id_overrides_extraction(retriever):
    """If the caller sets nct_id explicitly, don't override it from the text."""
    hits = retriever.search(RetrievalQuery(
        text="Tell me about NCT001",   # mentions NCT001
        nct_id="NCT002",                # but caller wants NCT002
        top_k=10,
    ))
    for h in hits:
        assert h.chunk.nct_id == "NCT002"


def test_nct_pattern_case_insensitive(retriever):
    hits = retriever.search(RetrievalQuery(text="info about nct001", top_k=10))
    assert len(hits) > 0
    for h in hits:
        assert h.chunk.nct_id == "NCT001"


# ---------------------------------------------------------------------------
# Convenience methods (the API agents will use)
# ---------------------------------------------------------------------------

def test_search_safety_filters_to_aes(retriever):
    hits = retriever.search_safety("any AEs", top_k=10)
    for h in hits:
        assert h.chunk.section_type == "adverse_events"


def test_search_safety_with_nct_filter(retriever):
    hits = retriever.search_safety("any AEs", nct_id="NCT001", top_k=10)
    assert len(hits) == 1
    assert hits[0].chunk.nct_id == "NCT001"


def test_search_eligibility(retriever):
    hits = retriever.search_eligibility("inclusion", top_k=10)
    for h in hits:
        assert h.chunk.section_type == "eligibility"


def test_get_trial_chunks_returns_all_sections(retriever):
    hits = retriever.get_trial_chunks("NCT001", top_k=20)
    # NCT001 has 3 sections seeded above
    assert len(hits) == 3
    sections = {h.chunk.section_type for h in hits}
    assert sections == {"overview", "adverse_events", "eligibility"}