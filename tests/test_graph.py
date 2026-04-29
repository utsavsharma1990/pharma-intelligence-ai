"""
Tests for the supervisor + graph wiring.
Uses EchoProvider and FakeEmbedder so no real LLM/embedding calls.
"""

import pytest
from pathlib import Path

from src.agents.supervisor import _keyword_route, supervisor_node
from src.agents.state import GraphState
from src.agents.graph import build_graph, _route_after_supervisor
from src.core.llm import EchoProvider, LLMMessage
from src.core.chroma_store import ChromaStore
from src.core.embeddings import EmbeddingProvider
from src.core.retriever import Retriever


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
def retriever(tmp_path):
    store = ChromaStore(persist_dir=tmp_path, collection_name="test")
    return Retriever(FakeEmbedder(), store)


@pytest.fixture
def llm():
    return EchoProvider()


@pytest.fixture
def graph(llm, retriever):
    return build_graph(llm, retriever)


# ---------------------------------------------------------------------------
# Keyword router tests
# ---------------------------------------------------------------------------

def test_keyword_route_safety():
    route, reason = _keyword_route("What adverse events were reported?")
    assert route == "safety"


def test_keyword_route_comparative():
    route, reason = _keyword_route("Compare pembrolizumab vs nivolumab")
    assert route == "comparative"


def test_keyword_route_default_search():
    route, reason = _keyword_route("Find Phase 3 lung cancer trials")
    assert route == "search"


def test_keyword_route_toxicity():
    route, reason = _keyword_route("What are the toxicity concerns?")
    assert route == "safety"


# ---------------------------------------------------------------------------
# Supervisor node tests
# ---------------------------------------------------------------------------

def test_supervisor_returns_route(llm):
    state: GraphState = {"query": "What adverse events occurred?"}
    result = supervisor_node(state, llm)
    assert "agent_route" in result
    assert result["agent_route"] in ("search", "comparative", "safety")
    assert "route_reason" in result


def test_supervisor_routes_safety_query(llm):
    state: GraphState = {"query": "What were the serious adverse events?"}
    result = supervisor_node(state, llm)
    assert result["agent_route"] == "safety"


def test_supervisor_routes_compare_query(llm):
    state: GraphState = {"query": "Compare drug A vs drug B side effects"}
    result = supervisor_node(state, llm)
    assert result["agent_route"] in ("comparative", "safety")


# ---------------------------------------------------------------------------
# Conditional edge routing
# ---------------------------------------------------------------------------

def test_route_after_supervisor_safety():
    state: GraphState = {"agent_route": "safety"}
    assert _route_after_supervisor(state) == "safety"


def test_route_after_supervisor_comparative():
    state: GraphState = {"agent_route": "comparative"}
    assert _route_after_supervisor(state) == "comparative"


def test_route_after_supervisor_default_on_unknown():
    state: GraphState = {"agent_route": "nonexistent_agent"}
    assert _route_after_supervisor(state) == "search"


# ---------------------------------------------------------------------------
# Full graph execution tests
# ---------------------------------------------------------------------------

def test_graph_runs_safety_query(graph):
    result = graph.invoke({"query": "What adverse events were reported in the trial?"})
    assert "final_answer" in result
    assert result["agent_route"] == "safety"
    # Real agent now runs — just verify we got a non-empty answer
    assert len(result["final_answer"]) > 0


def test_graph_runs_search_query(graph):
    result = graph.invoke({"query": "Find Phase 3 lung cancer trials by Merck"})
    assert "final_answer" in result
    assert result["agent_route"] == "search"
    # Real agent now runs — just verify we got a non-empty answer
    assert len(result["final_answer"]) > 0


def test_graph_runs_comparative_query(graph):
    result = graph.invoke({"query": "Compare pembrolizumab vs nivolumab"})
    assert "final_answer" in result
    assert result["agent_route"] == "comparative"


def test_graph_final_answer_includes_route_info(graph):
    result = graph.invoke({"query": "What adverse events occurred?"})
    # Real synthesizer appends route info in footer
    assert "specialist" in result["final_answer"].lower() or \
           result["agent_route"] in result["final_answer"]


def test_graph_preserves_original_query(graph):
    query = "Unique query string XYZ123"
    result = graph.invoke({"query": query})
    assert result["query"] == query