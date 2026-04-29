"""Tests for the evaluation metrics."""

import pytest
from unittest.mock import MagicMock

from src.core.llm import EchoProvider, LLMMessage
from src.core.vector_store import SearchResult
from src.ingestion.models import TrialChunk
from src.evaluation.metrics import (
    compute_faithfulness,
    compute_answer_relevancy,
    compute_retrieval_precision,
    compute_latency_percentiles,
    _keyword_faithfulness,
)


def _make_hit(nct_id: str, content: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=TrialChunk(
            chunk_id=f"{nct_id}::overview::0",
            nct_id=nct_id,
            section_type="overview",
            content=content,
        ),
        score=score,
    )


@pytest.fixture
def llm():
    return EchoProvider()


# ---------------------------------------------------------------------------
# Retrieval precision
# ---------------------------------------------------------------------------

def test_precision_all_found():
    hits = [_make_hit("NCT001", "..."), _make_hit("NCT002", "...")]
    assert compute_retrieval_precision(hits, ["NCT001", "NCT002"]) == 1.0


def test_precision_partial():
    hits = [_make_hit("NCT001", "...")]
    assert compute_retrieval_precision(hits, ["NCT001", "NCT002"]) == 0.5


def test_precision_none_found():
    hits = [_make_hit("NCT003", "...")]
    assert compute_retrieval_precision(hits, ["NCT001", "NCT002"]) == 0.0


def test_precision_empty_expected():
    hits = [_make_hit("NCT001", "...")]
    assert compute_retrieval_precision(hits, []) == 1.0


def test_precision_empty_retrieved():
    assert compute_retrieval_precision([], ["NCT001"]) == 0.0


def test_precision_case_insensitive():
    hits = [_make_hit("nct001", "...")]
    assert compute_retrieval_precision(hits, ["NCT001"]) == 1.0


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------

def test_latency_basic():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    result = compute_latency_percentiles(latencies)
    assert result["p50"] == 300.0
    assert result["count"] == 5
    assert "p95" in result
    assert "p99" in result
    assert "mean" in result


def test_latency_empty():
    result = compute_latency_percentiles([])
    assert result["count"] == 0
    assert result["p50"] == 0.0


def test_latency_single():
    result = compute_latency_percentiles([150.0])
    assert result["p50"] == 150.0
    assert result["mean"] == 150.0


# ---------------------------------------------------------------------------
# Keyword faithfulness fallback
# ---------------------------------------------------------------------------

def test_keyword_faithfulness_high():
    chunk_content = "pembrolizumab showed pneumonitis colitis fatigue adverse events"
    hits = [_make_hit("NCT001", chunk_content)]
    answer = "pembrolizumab showed pneumonitis and colitis"
    score = _keyword_faithfulness(answer, hits)
    assert score > 0.5


def test_keyword_faithfulness_low():
    hits = [_make_hit("NCT001", "completely unrelated content here")]
    answer = "quantum mechanics photon wavelength electromagnetic"
    score = _keyword_faithfulness(answer, hits)
    assert score < 0.3


def test_keyword_faithfulness_empty_answer():
    hits = [_make_hit("NCT001", "some content")]
    assert _keyword_faithfulness("", hits) == 0.0


# ---------------------------------------------------------------------------
# LLM-based metrics with EchoProvider
# ---------------------------------------------------------------------------

def test_faithfulness_returns_float(llm):
    hits = [_make_hit("NCT001", "pembrolizumab trial results")]
    score = compute_faithfulness("query", "answer about pembrolizumab", hits, llm)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_faithfulness_empty_answer(llm):
    hits = [_make_hit("NCT001", "content")]
    assert compute_faithfulness("query", "", hits, llm) == 0.0


def test_faithfulness_no_chunks(llm):
    assert compute_faithfulness("query", "answer", [], llm) == 0.0


def test_relevancy_returns_float(llm):
    score = compute_answer_relevancy("What are the AEs?", "Pneumonitis was observed.", llm)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_relevancy_empty_answer(llm):
    assert compute_answer_relevancy("query", "", llm) == 0.0