"""
Evaluation metrics for the pharma-intelligence-ai system.

Four metrics:
  1. Faithfulness     — is the answer grounded in the retrieved context?
  2. Answer Relevancy — does it address the question?
  3. Retrieval Precision — were the right trials retrieved?
  4. Latency          — response time percentiles (p50, p95, p99)

Design: metrics are simple functions, not classes. They take plain Python
inputs and return plain floats. Easy to test, easy to extend.

Faithfulness and relevancy use an LLM judge (the same LLMProvider ABC
you already built) — this is the "LLM-as-judge" pattern. With EchoProvider,
scores will be deterministic. With a real LLM, they'll be meaningful.
"""

import statistics
from typing import Optional

from src.core.llm import LLMMessage, LLMProvider
from src.core.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness(
    query: str,
    answer: str,
    retrieved_chunks: list[SearchResult],
    llm: LLMProvider,
) -> float:
    """
    Score 0.0-1.0: is every claim in the answer supported by the context?

    Uses LLM-as-judge: asks the LLM to check if the answer is grounded
    in the retrieved chunks. Returns 1.0 if fully grounded, 0.0 if not,
    with intermediate values for partial grounding.
    """
    if not answer.strip():
        return 0.0
    if not retrieved_chunks:
        return 0.0

    context = "\n\n".join(
        hit.chunk.content[:400] for hit in retrieved_chunks[:5]
    )

    prompt = f"""Given this context from a clinical trials database:
{context}

And this answer to the query "{query}":
{answer[:600]}

Rate how well the answer is grounded in the provided context on a scale of 0-10.
10 = every claim is directly supported by the context.
0 = the answer contains claims not found in the context at all.
5 = roughly half the claims are supported.

Respond with ONLY a single integer from 0 to 10."""

    schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "Faithfulness score 0-10",
            }
        },
        "required": ["score"],
    }

    try:
        result = llm.complete_structured(
            messages=[LLMMessage(role="user", content=prompt)],
            schema=schema,
            max_tokens=64,
            temperature=0.0,
        )
        raw = result.get("score", 5)
        return min(max(int(raw), 0), 10) / 10.0
    except Exception:
        # Fallback: keyword overlap heuristic
        return _keyword_faithfulness(answer, retrieved_chunks)


def _keyword_faithfulness(
    answer: str,
    chunks: list[SearchResult],
) -> float:
    """
    Fallback faithfulness metric: what fraction of answer words appear in context?
    Crude but deterministic.
    """
    context_words = set()
    for hit in chunks:
        context_words.update(hit.chunk.content.lower().split())

    answer_words = [
        w.lower().strip(".,;:") for w in answer.split()
        if len(w) > 4  # skip short words
    ]
    if not answer_words:
        return 0.0

    overlap = sum(1 for w in answer_words if w in context_words)
    return round(overlap / len(answer_words), 3)


# ---------------------------------------------------------------------------
# Answer Relevancy
# ---------------------------------------------------------------------------

def compute_answer_relevancy(
    query: str,
    answer: str,
    llm: LLMProvider,
) -> float:
    """
    Score 0.0-1.0: does the answer actually address the question?

    High score = the answer is directly relevant to what was asked.
    Low score = the answer is off-topic, too generic, or doesn't address the query.
    """
    if not answer.strip():
        return 0.0

    schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "Relevancy score 0-10",
            }
        },
        "required": ["score"],
    }

    prompt = f"""Question: {query}

Answer: {answer[:600]}

Rate how well the answer addresses the question on a scale of 0-10.
10 = directly and completely answers the question.
0 = completely off-topic or doesn't address the question at all.

Respond with ONLY a single integer from 0 to 10."""

    try:
        result = llm.complete_structured(
            messages=[LLMMessage(role="user", content=prompt)],
            schema=schema,
            max_tokens=64,
            temperature=0.0,
        )
        raw = result.get("score", 5)
        return min(max(int(raw), 0), 10) / 10.0
    except Exception:
        return 0.5  # neutral fallback


# ---------------------------------------------------------------------------
# Retrieval Precision
# ---------------------------------------------------------------------------

def compute_retrieval_precision(
    retrieved_chunks: list[SearchResult],
    expected_nct_ids: list[str],
) -> float:
    """
    Score 0.0-1.0: what fraction of expected NCT IDs appear in retrieved chunks?

    This is a recall metric (not traditional precision) — we care about
    whether the right trials were retrieved, not whether extra irrelevant
    ones slipped in.

    Example:
        expected = ["NCT001", "NCT002"]
        retrieved contains NCT001 but not NCT002 → score = 0.5
    """
    if not expected_nct_ids:
        return 1.0  # nothing expected = trivially satisfied
    if not retrieved_chunks:
        return 0.0

    retrieved_ids = {hit.chunk.nct_id.upper() for hit in retrieved_chunks}
    expected_ids  = {nct.upper() for nct in expected_nct_ids}

    found = len(expected_ids & retrieved_ids)
    return round(found / len(expected_ids), 3)


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------

def compute_latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """
    Compute p50, p95, p99 from a list of latency measurements (in ms).

    Why percentiles over averages?
    A single slow outlier (30s timeout) skews the mean dramatically.
    p95 tells you "95% of users got a response in under X ms" — a much
    more meaningful user experience metric.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        idx = int(p / 100 * n)
        return round(sorted_latencies[min(idx, n - 1)], 1)

    return {
        "p50":   percentile(50),
        "p95":   percentile(95),
        "p99":   percentile(99),
        "mean":  round(statistics.mean(sorted_latencies), 1),
        "count": n,
    }