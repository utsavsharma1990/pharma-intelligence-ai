"""
Trial Search Agent — handles general trial lookup queries.

This agent answers questions like:
  - "Find Phase 3 lung cancer trials sponsored by Merck"
  - "What are the eligibility criteria for NCT05123456?"
  - "What interventions are being tested in CLL trials?"
  - "Show me recruiting trials with pembrolizumab"

Strategy:
  1. Auto-detect if the query mentions a specific NCT ID (use targeted retrieval)
  2. Otherwise: broad semantic search, optionally filtered by phase/status
  3. Format context + call LLM with clinical research system prompt
  4. Return answer + citations + retrieved chunks

Why no section_type filter here?
  The Trial Search agent answers GENERAL questions about trials. We don't
  know upfront whether the user wants eligibility, endpoints, or overview
  info — so we do a broad search and let the LLM synthesize from whatever
  sections are most relevant. The Safety agent (Activity 15) is more targeted.
"""

import logging
from typing import Optional

from src.agents.base_agent import build_state_update, format_context
from src.agents.state import GraphState
from src.core.llm import LLMMessage, LLMProvider
from src.core.retriever import Retriever, RetrievalQuery

logger = logging.getLogger(__name__)

SEARCH_AGENT_SYSTEM_PROMPT = """You are a clinical trials research specialist with deep expertise
in oncology and pharmaceutical research. You answer questions about clinical trials using
ONLY the provided context from the ClinicalTrials.gov database.

Guidelines:
- Always cite the specific NCT ID when referencing a trial (e.g., "In trial NCT05123456...")
- If the context doesn't contain enough information, say so clearly
- For eligibility questions, list inclusion and exclusion criteria separately
- For endpoint questions, distinguish primary from secondary outcomes
- Be precise with medical terminology but explain acronyms on first use
- Do NOT make up information not present in the context

Context from ClinicalTrials.gov:
{context}"""


def search_agent_node(
    state: GraphState,
    llm: LLMProvider,
    retriever: Retriever,
) -> dict:
    """
    LangGraph node: retrieves relevant trial chunks and generates an answer.
    Returns a partial state update with agent_response, retrieved_chunks, citations.
    """
    query = state.get("query", "")
    logger.info(f"[SearchAgent] query: {query[:80]!r}")

    # --- Step 1: Retrieve relevant chunks ---
    hits = retriever.search(RetrievalQuery(
        text=query,
        top_k=8,
        # No section_type filter — broad search to let LLM synthesize
    ))

    logger.info(f"[SearchAgent] retrieved {len(hits)} chunks")

    # --- Step 2: Format context for the LLM ---
    context = format_context(hits, max_chunks=6)
    system_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(context=context)

    # --- Step 3: Generate answer ---
    response = llm.complete(
        messages=[LLMMessage(role="user", content=query)],
        system=system_prompt,
        max_tokens=1024,
        temperature=0.1,   # slight creativity OK for synthesis, but stay grounded
    )

    logger.info(f"[SearchAgent] answer length: {len(response)} chars")
    return build_state_update(response, hits)


def make_search_node(llm: LLMProvider, retriever: Retriever):
    """Factory: bind dependencies, return a LangGraph-compatible node callable."""
    def _node(state: GraphState) -> dict:
        return search_agent_node(state, llm, retriever)
    _node.__name__ = "search"
    return _node