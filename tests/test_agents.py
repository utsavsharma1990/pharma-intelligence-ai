"""Tests for comparative, safety, and synthesizer agents."""

import pytest
from pathlib import Path

from src.agents.comparative_agent import comparative_agent_node
from src.agents.safety_agent import safety_agent_node
from src.agents.synthesizer_agent import synthesizer_node
from src.agents.state import GraphState
from src.core.chroma_store import ChromaStore
from src.core.embeddings import EmbeddingProvider
from src.core.llm import EchoProvider
from src.core.retriever import Retriever
from src.ingestion.models import TrialChunk


class FakeEmbedder(EmbeddingProvider):
    DIM = 16
    def embed_texts(self, texts):
        return [[(sum(ord(c) for c in t) + i) % 100 * 0.01
                 for i in range(self.DIM)] for t in texts]
    @property
    def dimension(self): return self.DIM


@pytest.fixture
def populated_retriever(tmp_path):
    chunks = [
        TrialChunk(
            chunk_id="NCT05123456::adverse_events::0",
            nct_id="NCT05123456", section_type="adverse_events",
            content="NCT05123456 AEs: Pneumonitis 24/800 (3.0%), Colitis 16/800 (2.0%). Fatigue 320/800 (40%).",
            phase="PHASE3", sponsor="Merck", overall_status="RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT04567890::adverse_events::0",
            nct_id="NCT04567890", section_type="adverse_events",
            content="NCT04567890 AEs: Pneumonitis 18/582 (3.1%), Hepatitis 10/582 (1.7%). Fatigue 175/582 (30%).",
            phase="PHASE3", sponsor="Bristol-Myers Squibb", overall_status="ACTIVE_NOT_RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT05123456::overview::0",
            nct_id="NCT05123456", section_type="overview",
            content="NCT05123456: Pembrolizumab in NSCLC. Phase 3, RECRUITING, Merck.",
            phase="PHASE3", sponsor="Merck", overall_status="RECRUITING",
        ),
        TrialChunk(
            chunk_id="NCT04567890::overview::0",
            nct_id="NCT04567890", section_type="overview",
            content="NCT04567890: Nivolumab vs Docetaxel NSCLC. Phase 3. Bristol-Myers Squibb.",
            phase="PHASE3", sponsor="Bristol-Myers Squibb", overall_status="ACTIVE_NOT_RECRUITING",
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
# Safety agent
# ---------------------------------------------------------------------------

def test_safety_agent_returns_response(populated_retriever, llm):
    state: GraphState = {"query": "What adverse events were reported?"}
    result = safety_agent_node(state, llm, populated_retriever)
    assert "agent_response" in result
    assert len(result["agent_response"]) > 0


def test_safety_agent_returns_ae_chunks(populated_retriever, llm):
    state: GraphState = {"query": "adverse events in lung cancer trials"}
    result = safety_agent_node(state, llm, populated_retriever)
    chunks = result.get("retrieved_chunks", [])
    # All retrieved chunks should be AE section type
    for hit in chunks:
        assert hit.chunk.section_type == "adverse_events"


def test_safety_agent_handles_empty_index(tmp_path, llm):
    empty_store = ChromaStore(persist_dir=tmp_path, collection_name="empty")
    retriever = Retriever(FakeEmbedder(), empty_store)
    state: GraphState = {"query": "adverse events"}
    result = safety_agent_node(state, llm, retriever)
    assert "agent_response" in result  # graceful fallback


# ---------------------------------------------------------------------------
# Comparative agent
# ---------------------------------------------------------------------------

def test_comparative_agent_returns_response(populated_retriever, llm):
    state: GraphState = {
        "query": "Compare NCT05123456 and NCT04567890 safety profiles"
    }
    result = comparative_agent_node(state, llm, populated_retriever)
    assert "agent_response" in result


def test_comparative_agent_retrieves_both_trials(populated_retriever, llm):
    state: GraphState = {
        "query": "Compare NCT05123456 and NCT04567890"
    }
    result = comparative_agent_node(state, llm, populated_retriever)
    chunks = result.get("retrieved_chunks", [])
    nct_ids = {h.chunk.nct_id for h in chunks}
    assert "NCT05123456" in nct_ids
    assert "NCT04567890" in nct_ids


def test_comparative_agent_broad_search_fallback(populated_retriever, llm):
    """When no NCT IDs mentioned, should still return results."""
    state: GraphState = {"query": "Compare pembrolizumab vs nivolumab"}
    result = comparative_agent_node(state, llm, populated_retriever)
    assert "agent_response" in result
    assert len(result.get("retrieved_chunks", [])) > 0


# ---------------------------------------------------------------------------
# Synthesizer agent
# ---------------------------------------------------------------------------

def test_synthesizer_produces_final_answer(llm):
    state: GraphState = {
        "agent_response": "NCT05123456 showed OS benefit.",
        "citations": ["NCT05123456"],
        "agent_route": "search",
        "route_reason": "trial lookup",
    }
    result = synthesizer_node(state, llm)
    assert "final_answer" in result


def test_synthesizer_appends_footer(llm):
    state: GraphState = {
        "agent_response": "Some answer about NCT05123456.",
        "citations": ["NCT05123456"],
        "agent_route": "safety",
        "route_reason": "AE question",
    }
    result = synthesizer_node(state, llm)
    footer = result["final_answer"]
    assert "safety" in footer
    assert "NCT05123456" in footer


def test_synthesizer_handles_no_citations(llm):
    state: GraphState = {
        "agent_response": "General answer with no specific trial.",
        "citations": [],
        "agent_route": "search",
        "route_reason": "default",
    }
    result = synthesizer_node(state, llm)
    assert "final_answer" in result


# ---------------------------------------------------------------------------
# Full graph end-to-end with all real agents
# ---------------------------------------------------------------------------

def test_full_graph_safety_route(populated_retriever, llm):
    from src.agents.graph import build_graph
    graph = build_graph(llm, populated_retriever)
    result = graph.invoke({"query": "What adverse events were reported?"})
    assert result["agent_route"] == "safety"
    assert "final_answer" in result
    assert "SAFETY AGENT STUB" not in result["final_answer"]


def test_full_graph_comparative_route(populated_retriever, llm):
    from src.agents.graph import build_graph
    graph = build_graph(llm, populated_retriever)
    result = graph.invoke({
        "query": "Compare NCT05123456 vs NCT04567890"
    })
    assert result["agent_route"] == "comparative"
    assert "final_answer" in result


def test_full_graph_preserves_query(populated_retriever, llm):
    from src.agents.graph import build_graph
    graph = build_graph(llm, populated_retriever)
    q = "Test query for graph preservation"
    result = graph.invoke({"query": q})
    assert result["query"] == q