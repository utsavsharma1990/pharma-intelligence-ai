"""
Shared utilities for all specialist agents.

Every specialist agent does the same boring things:
  - Format retrieved chunks into a context string for the LLM prompt
  - Extract NCT ID citations from the LLM's response
  - Build a standard partial state update dict

Centralizing this here means each specialist only writes the parts that
are actually different: the retrieval strategy and the system prompt.
"""

import re
from src.core.vector_store import SearchResult

# Reuse the same NCT pattern as the retriever
NCT_CITATION_PATTERN = re.compile(r"\bNCT\d{3,10}\b", re.IGNORECASE)


def format_context(hits: list[SearchResult], max_chunks: int = 8) -> str:
    """
    Format retrieved chunks into a numbered context string for LLM prompts.

    We include:
      - Source citation (NCT ID + section type)
      - Similarity score (so the LLM can weight more-relevant chunks higher)
      - Full chunk content

    Why numbered? The LLM can say "Based on source [3]..." in its answer,
    giving us a hook for citation extraction later.

    Args:
        hits:       SearchResult list from the retriever (ordered by score)
        max_chunks: cap context size to avoid token limit issues
    """
    if not hits:
        return "No relevant clinical trial data found in the index."

    lines = []
    for i, hit in enumerate(hits[:max_chunks], 1):
        c = hit.chunk
        lines.append(
            f"[{i}] Source: {c.nct_id} / {c.section_type} "
            f"(relevance: {hit.score:.2f})"
        )
        lines.append(c.content)
        lines.append("")   # blank line between chunks

    return "\n".join(lines)


def extract_citations(text: str) -> list[str]:
    """
    Extract all NCT IDs mentioned in the LLM's response.
    Returns a deduplicated list preserving first-mention order.
    """
    seen = set()
    citations = []
    for match in NCT_CITATION_PATTERN.finditer(text):
        nct = match.group(0).upper()
        if nct not in seen:
            seen.add(nct)
            citations.append(nct)
    return citations


def build_state_update(
    agent_response: str,
    hits: list[SearchResult],
) -> dict:
    """
    Build the standard partial state update returned by every specialist.
    """
    return {
        "agent_response":   agent_response,
        "retrieved_chunks": hits,
        "citations":        extract_citations(agent_response),
    }