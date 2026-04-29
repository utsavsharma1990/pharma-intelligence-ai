"""
FastAPI endpoint tests using TestClient (no real HTTP, no real agents).
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_graph, get_store, get_retriever


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset slowapi in-memory storage between tests to prevent count bleed."""
    from src.api.app import limiter
    # Clear the in-memory storage by resetting the underlying storage backend
    try:
        limiter._storage.reset()
    except Exception:
        pass  # some backends don't support reset — that's fine
    yield

@pytest.fixture
def mock_graph():
    """Fake graph that returns a canned response."""
    g = MagicMock()
    g.invoke.return_value = {
        "final_answer":     "NCT05123456 had pneumonitis (3%).",
        "agent_route":      "safety",
        "route_reason":     "AE question detected",
        "citations":        ["NCT05123456"],
        "retrieved_chunks": [],
    }
    return g


@pytest.fixture
def mock_store():
    s = MagicMock()
    s.count.return_value = 15
    return s


@pytest.fixture
def client(mock_graph, mock_store):
    """TestClient with mocked dependencies."""
    app = create_app()
    app.dependency_overrides[get_graph]    = lambda: mock_graph
    app.dependency_overrides[get_store]    = lambda: mock_store
    app.dependency_overrides[get_retriever] = lambda: MagicMock()
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health + Ready
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_returns_chunk_count(client):
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is True
    assert data["chunks_indexed"] == 15


def test_ready_not_ready_when_empty(mock_graph):
    empty_store = MagicMock()
    empty_store.count.return_value = 0
    app = create_app()
    app.dependency_overrides[get_graph]    = lambda: mock_graph
    app.dependency_overrides[get_store]    = lambda: empty_store
    app.dependency_overrides[get_retriever] = lambda: MagicMock()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/ready")
    assert r.json()["ready"] is False


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

def test_query_returns_answer(client):
    r = client.post("/query", json={"query": "What adverse events were reported?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "NCT05123456" in data["answer"]


def test_query_returns_agent_route(client):
    r = client.post("/query", json={"query": "What adverse events?"})
    data = r.json()
    assert data["agent_route"] == "safety"
    assert data["route_reason"] == "AE question detected"


def test_query_returns_citations(client):
    r = client.post("/query", json={"query": "What adverse events?"})
    data = r.json()
    assert "NCT05123456" in data["citations"]


def test_query_too_short_returns_422(client):
    r = client.post("/query", json={"query": "hi"})
    assert r.status_code == 422


def test_query_missing_body_returns_422(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_with_phase_filter(client):
    r = client.post("/query", json={
        "query": "Find lung cancer trials",
        "phase": "PHASE3",
        "sponsor": "Merck",
    })
    assert r.status_code == 200


def test_query_propagates_graph_error(mock_store):
    bad_graph = MagicMock()
    bad_graph.invoke.side_effect = RuntimeError("LLM timeout")

    app = create_app()
    app.dependency_overrides[get_graph]     = lambda: bad_graph
    app.dependency_overrides[get_store]     = lambda: mock_store
    app.dependency_overrides[get_retriever] = lambda: MagicMock()

    # Give this isolated app a fresh limiter with a very high limit
    # so previous tests' request counts don't bleed into this one.
    # We use app.state.limiter — slowapi reads this at request time.
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    fresh_limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["10000 per minute"],  # effectively unlimited
    )
    app.state.limiter = fresh_limiter

    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/query", json={"query": "Any question here"})
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Response headers
# ---------------------------------------------------------------------------

def test_response_has_request_id(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers


def test_response_has_timing_header(client):
    r = client.get("/health")
    assert "x-response-time" in r.headers