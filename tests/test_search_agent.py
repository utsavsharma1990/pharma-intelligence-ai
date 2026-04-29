"""Tests for the Trial Search agent."""

import pytest
from pathlib import Path
import json

from src.agents.search_agent import search_agent_node, make_search_node
from src.agents.base_agent import format_context, extract_citations
from src.agents.state import GraphState
from src.core.chroma_store import ChromaStore
from src.core.embeddings import EmbeddingProvider
from src.core.llm import EchoProvider
from src.core.retriever import Retriever
from src.ingestion.models import TrialChunk
from src.core.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeEmbedder(EmbeddingProvider):
    DIM = 16
    def embed_texts(self, texts):
        return [[(sum(ord(c) for c in t) + i) % 100 * 0.01
                 for i in range(self.DIM)] for t in texts]
    @property
    def dimension(self): return self.DIM


@pytest.fixture
def populated_retriever(tmp_path):
    """Retriever seeded with mock trial chunks."""
    chunks = [
        TrialChunk(
            chunk_id="NCT05123456::overview::0",
            nct_id="NCT05123456",
            section_type="overview",
            content="Trial NCT05123456: Pembrolizumab Plus Chemotherapy in NSCLC. Phase: PHASE3. Status: RECRUITING. Sponsor: Merck Sharp & Dohme LLC. Planned enrollment: 800 participants.",
            phase="PHASE3",
            sponsor="Merck Sharp & Dohme LLC",
            overall_status="RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT05123456::eligibility::0",
            nct_id="NCT05123456",
            section_type="eligibility",
            content="Trial NCT05123456 ELIGIBILITY CRITERIA\nInclusion: Adults 18+, NSCLC confirmed, ECOG 0-1.\nExclusion: Prior anti-PD-1 therapy, active brain metastases.",
            phase="PHASE3",
            sponsor="Merck Sharp & Dohme LLC",
            overall_status="RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT04567890::overview::0",
            nct_id="NCT04567890",
            section_type="overview",
            content="Trial NCT04567890: Nivolumab vs Docetaxel in NSCLC. Phase: PHASE3. Sponsor: Bristol-Myers Squibb.",
            phase="PHASE3",
            sponsor="Bristol-Myers Squibb",
            overall_status="ACTIVE_NOT_RECRUITING",
        ),
    ]
    embedder = FakeEmbedder()
    store = ChromaStore(persist_dir=tmp_path, collection_name="test")
    store.add_chunks(chunks, embedder.embed_texts([c.content for c in chunks]))
    return Retriever(embedder, store)


@pytest.fixture
def llm():
    return EchoProvider()


# ---------------------------------------------------------------------------
# base_agent utility tests
# ---------------------------------------------------------------------------

def test_format_context_empty():
    result = format_context([])
    assert "No relevant" in result


def test_format_context_numbers_sources():
    chunk = TrialChunk(
        chunk_id="NCT001::overview::0", nct_id="NCT001",
        section_type="overview", content="Some trial content"
    )
    hits = [SearchResult(chunk=chunk, score=0.9)]
    result = format_context(hits)
    assert "[1]" in result
    assert "NCT001" in result
    assert "0.90" in result


def test_format_context_respects_max_chunks():
    chunks = [
        SearchResult(
            chunk=TrialChunk(
                chunk_id=f"NCT00{i}::overview::0",
                nct_id=f"NCT00{i}",
                section_type="overview",
                content=f"Content {i}"
            ),
            score=0.9,
        )
        for i in range(10)
    ]
    result = format_context(chunks, max_chunks=3)
    assert "[3]" in result
    assert "[4]" not in result


def test_extract_citations_finds_nct_ids():
    text = "Based on NCT05123456 and NCT04567890, the results show..."
    citations = extract_citations(text)
    assert "NCT05123456" in citations
    assert "NCT04567890" in citations


def test_extract_citations_deduplicates():
    text = "NCT05123456 shows X. NCT05123456 also shows Y."
    citations = extract_citations(text)
    assert citations.count("NCT05123456") == 1


def test_extract_citations_empty_text():
    assert extract_citations("No trials mentioned here.") == []


# ---------------------------------------------------------------------------
# Search agent node tests
# ---------------------------------------------------------------------------

def test_search_agent_returns_response(populated_retriever, llm):
    state: GraphState = {"query": "Find Phase 3 lung cancer trials"}
    result = search_agent_node(state, llm, populated_retriever)
    assert "agent_response" in result
    assert len(result["agent_response"]) > 0


def test_search_agent_returns_retrieved_chunks(populated_retriever, llm):
    state: GraphState = {"query": "pembrolizumab NSCLC trial"}
    result = search_agent_node(state, llm, populated_retriever)
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) > 0


def test_search_agent_returns_citations_list(populated_retriever, llm):
    state: GraphState = {"query": "eligibility for NCT05123456"}
    result = search_agent_node(state, llm, populated_retriever)
    assert "citations" in result
    assert isinstance(result["citations"], list)


def test_search_agent_handles_empty_index(tmp_path, llm):
    """Agent should handle empty vector store gracefully."""
    empty_store = ChromaStore(persist_dir=tmp_path, collection_name="empty")
    retriever = Retriever(FakeEmbedder(), empty_store)
    state: GraphState = {"query": "anything"}
    result = search_agent_node(state, llm, retriever)
    assert "agent_response" in result   # should still return something


# ---------------------------------------------------------------------------
# Full graph integration
# ---------------------------------------------------------------------------

def test_search_node_in_graph(populated_retriever, llm):
    """Verify the real search node integrates correctly in the graph."""
    from src.agents.graph import build_graph
    graph = build_graph(llm, populated_retriever)
    result = graph.invoke({"query": "Find Phase 3 lung cancer trials"})
    assert result["agent_route"] == "search"
    assert "final_answer" in result
    # Should NOT contain the stub marker anymore
    assert "SEARCH AGENT STUB" not in result["final_answer"]