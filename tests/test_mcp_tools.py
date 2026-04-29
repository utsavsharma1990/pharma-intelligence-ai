"""
Tests for MCP tool implementations.
Uses a mock graph so no real LLM/embedding calls.
"""

from unittest.mock import MagicMock
import pytest

from src.mcp_server.tools import (
    search_trials,
    get_trial_details,
    compare_trials,
    get_safety_profile,
    find_eligible_trials,
)


@pytest.fixture
def mock_graph():
    """Graph that returns a canned final_answer."""
    g = MagicMock()
    g.invoke.return_value = {
        "final_answer": "Canned answer about NCT05123456.",
        "agent_route":  "search",
        "citations":    ["NCT05123456"],
    }
    return g


# ---------------------------------------------------------------------------
# search_trials
# ---------------------------------------------------------------------------

def test_search_trials_basic(mock_graph):
    result = search_trials(mock_graph, query="lung cancer trials")
    assert "NCT05123456" in result
    mock_graph.invoke.assert_called_once()


def test_search_trials_enriches_query_with_filters(mock_graph):
    search_trials(mock_graph, query="lung cancer", phase="PHASE3", sponsor="Merck")
    call_args = mock_graph.invoke.call_args[0][0]
    assert "PHASE3" in call_args["query"]
    assert "Merck" in call_args["query"]


def test_search_trials_no_filters(mock_graph):
    search_trials(mock_graph, query="any trial")
    call_args = mock_graph.invoke.call_args[0][0]
    assert call_args["query"] == "any trial"


# ---------------------------------------------------------------------------
# get_trial_details
# ---------------------------------------------------------------------------

def test_get_trial_details(mock_graph):
    result = get_trial_details(mock_graph, nct_id="NCT05123456")
    assert len(result) > 0
    call_args = mock_graph.invoke.call_args[0][0]
    assert "NCT05123456" in call_args["query"]


# ---------------------------------------------------------------------------
# compare_trials
# ---------------------------------------------------------------------------

def test_compare_trials_two_ids(mock_graph):
    result = compare_trials(mock_graph, nct_ids=["NCT05123456", "NCT04567890"])
    assert len(result) > 0
    call_args = mock_graph.invoke.call_args[0][0]
    assert "NCT05123456" in call_args["query"]
    assert "NCT04567890" in call_args["query"]


def test_compare_trials_with_aspect(mock_graph):
    compare_trials(mock_graph, nct_ids=["NCT001", "NCT002"], aspect="safety")
    call_args = mock_graph.invoke.call_args[0][0]
    assert "safety" in call_args["query"]


def test_compare_trials_too_few_ids(mock_graph):
    result = compare_trials(mock_graph, nct_ids=["NCT001"])
    assert "at least 2" in result
    mock_graph.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# get_safety_profile
# ---------------------------------------------------------------------------

def test_get_safety_profile_by_nct(mock_graph):
    result = get_safety_profile(mock_graph, nct_id="NCT05123456")
    assert len(result) > 0
    call_args = mock_graph.invoke.call_args[0][0]
    assert "NCT05123456" in call_args["query"]


def test_get_safety_profile_by_drug(mock_graph):
    result = get_safety_profile(mock_graph, drug_name="pembrolizumab")
    assert len(result) > 0
    call_args = mock_graph.invoke.call_args[0][0]
    assert "pembrolizumab" in call_args["query"]


def test_get_safety_profile_no_args(mock_graph):
    result = get_safety_profile(mock_graph)
    assert "Please provide" in result
    mock_graph.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# find_eligible_trials
# ---------------------------------------------------------------------------

def test_find_eligible_trials_full_profile(mock_graph):
    result = find_eligible_trials(
        mock_graph,
        age=65,
        condition="NSCLC",
        prior_therapies=["carboplatin"],
        ecog_status=1,
    )
    assert len(result) > 0
    call_args = mock_graph.invoke.call_args[0][0]
    query = call_args["query"]
    assert "NSCLC" in query
    assert "65" in query
    assert "carboplatin" in query
    assert "1" in query


def test_find_eligible_trials_minimal(mock_graph):
    result = find_eligible_trials(mock_graph, condition="CLL")
    assert len(result) > 0


def test_find_eligible_trials_graph_error(mock_graph):
    mock_graph.invoke.side_effect = RuntimeError("graph failed")
    result = find_eligible_trials(mock_graph, condition="NSCLC")
    assert "Error" in result   # graceful error string, not exception