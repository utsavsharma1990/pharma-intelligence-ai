"""
Comparative Analysis Agent — compares trials, drugs, sponsors, or indications.

Handles queries like:
  - "Compare the safety profiles of pembrolizumab and nivolumab in NSCLC"
  - "Which trial has better enrollment — NCT05123456 or NCT04567890?"
  - "Compare Merck vs Bristol-Myers Squibb NSCLC trials"

Strategy:
  - Detect multiple NCT IDs in the query → run targeted searches for each
  - If no NCT IDs, do a broad search and let the LLM compare what it finds
  - System prompt explicitly instructs structured side-by-side comparison

Why parallel searches for multiple NCT IDs?
  A single semantic search may not return ALL sections for both trials if
  they're semantically distant. Targeted per-NCT retrieval guarantees we
  get context for each trial being compared.
"""

import logging
import re
from src.agents.base_agent import build_state_update, format_context
from src.agents.state import GraphState
from src.core.llm import LLMMessage, LLMProvider
from src.core.retriever import Retriever, RetrievalQuery
from src.core.vector_store import SearchResult

logger = logging.getLogger(__name__)

NCT_PATTERN = re.compile(r"\bNCT\d{3,10}\b", re.IGNORECASE)

COMPARATIVE_SYSTEM_PROMPT = """You are a clinical trials research analyst specializing in
comparative analysis. You compare clinical trials, drugs, and sponsors using ONLY the
provided context from ClinicalTrials.gov.

When comparing trials, structure your response as:
1. **Overview** — brief description of each trial being compared
2. **Key Differences** — phase, enrollment, sponsor, status
3. **Endpoint Comparison** — primary and secondary outcomes
4. **Safety Comparison** — adverse event profiles if available
5. **Summary** — which trial addresses what differently and why it matters

Always cite NCT IDs explicitly. If data is missing for one trial, say so.

Context from ClinicalTrials.gov:
{context}"""


def comparative_agent_node(
    state: GraphState,
    llm: LLMProvider,
    retriever: Retriever,
) -> dict:
    query = state.get("query", "")
    logger.info(f"[ComparativeAgent] query: {query[:80]!r}")

    # Detect NCT IDs mentioned in the query
    nct_ids = [m.group(0).upper() for m in NCT_PATTERN.finditer(query)]

    all_hits: list[SearchResult] = []

    if len(nct_ids) >= 2:
        # Targeted retrieval: fetch chunks for each specific trial
        # This guarantees coverage even if trials are semantically distant
        for nct_id in nct_ids[:4]:  # cap at 4 trials for context size
            hits = retriever.get_trial_chunks(nct_id, top_k=6)
            all_hits.extend(hits)
            logger.info(f"[ComparativeAgent] {nct_id}: {len(hits)} chunks")
    else:
        # No explicit NCT IDs — broad semantic search, let LLM compare
        all_hits = retriever.search(RetrievalQuery(text=query, top_k=10))
        logger.info(f"[ComparativeAgent] broad search: {len(all_hits)} chunks")

    context = format_context(all_hits, max_chunks=10)
    system_prompt = COMPARATIVE_SYSTEM_PROMPT.format(context=context)

    response = llm.complete(
        messages=[LLMMessage(role="user", content=query)],
        system=system_prompt,
        max_tokens=1500,   # comparisons are verbose — allow more tokens
        temperature=0.1,
    )

    logger.info(f"[ComparativeAgent] answer: {len(response)} chars")
    return build_state_update(response, all_hits)


def make_comparative_node(llm: LLMProvider, retriever: Retriever):
    def _node(state: GraphState) -> dict:
        return comparative_agent_node(state, llm, retriever)
    _node.__name__ = "comparative"
    return _node