"""
Safety & Adverse Events Agent — analyzes AE profiles and safety data.

Handles queries like:
  - "What adverse events were reported in NCT05123456?"
  - "What are the serious AEs for pembrolizumab trials?"
  - "What immune-related toxicities were observed?"

Strategy:
  - ALWAYS filter to section_type='adverse_events' — high precision
  - Optionally filter to specific NCT ID if mentioned
  - System prompt focuses on clinical significance of AEs

Why strict section filtering here but not in the Search agent?
  Safety queries have a very narrow information need. We know EXACTLY
  which chunk type has the answer (adverse_events). Broad search would
  dilute the context with eligibility/endpoint chunks that don't help.
  Precision > recall for safety queries.
"""

import logging
from src.agents.base_agent import build_state_update, format_context
from src.agents.state import GraphState
from src.core.llm import LLMMessage, LLMProvider
from src.core.retriever import Retriever

logger = logging.getLogger(__name__)

SAFETY_SYSTEM_PROMPT = """You are a clinical pharmacovigilance specialist analyzing
adverse event data from clinical trials. Use ONLY the provided context.

When analyzing adverse events:
1. **Serious Adverse Events (SAEs)** — list with incidence rates (n/N, %)
2. **Common Adverse Events** — list most frequent (>5% incidence)
3. **Organ System Summary** — group by affected system (respiratory, GI, etc.)
4. **Clinical Significance** — note any Grade 3/4 events or treatment discontinuations
5. **Comparison note** — if multiple trials, compare AE profiles

Important:
- Always report incidence as both count and percentage: "24/800 (3.0%)"
- Distinguish serious vs common adverse events clearly
- If data is incomplete, say so explicitly — never estimate AE rates
- Cite the specific NCT ID for each AE data point

Context from ClinicalTrials.gov (adverse events section):
{context}"""


def safety_agent_node(
    state: GraphState,
    llm: LLMProvider,
    retriever: Retriever,
) -> dict:
    query = state.get("query", "")
    logger.info(f"[SafetyAgent] query: {query[:80]!r}")

    # ALWAYS filter to adverse_events chunks — this is the key differentiator
    # vs the Search agent which does a broad search
    hits = retriever.search_safety(
        text=query,
        top_k=8,
        # nct_id auto-extracted from query text by the retriever
    )

    logger.info(f"[SafetyAgent] retrieved {len(hits)} AE chunks")

    # If no AE chunks found, try a broader search as fallback
    if not hits:
        logger.warning("[SafetyAgent] No AE chunks found, falling back to broad search")
        from src.core.retriever import RetrievalQuery
        hits = retriever.search(RetrievalQuery(text=query, top_k=5))

    context = format_context(hits, max_chunks=8)
    system_prompt = SAFETY_SYSTEM_PROMPT.format(context=context)

    response = llm.complete(
        messages=[LLMMessage(role="user", content=query)],
        system=system_prompt,
        max_tokens=1024,
        temperature=0.0,   # safety data should be reproduced precisely, not creatively
    )

    logger.info(f"[SafetyAgent] answer: {len(response)} chars")
    return build_state_update(response, hits)


def make_safety_node(llm: LLMProvider, retriever: Retriever):
    def _node(state: GraphState) -> dict:
        return safety_agent_node(state, llm, retriever)
    _node.__name__ = "safety"
    return _node