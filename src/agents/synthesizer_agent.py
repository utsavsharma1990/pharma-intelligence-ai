"""
Synthesizer Agent — final polish and citation formatting.

The synthesizer is the LAST node before END. It receives:
  - state['agent_response']: the raw answer from the specialist agent
  - state['citations']: NCT IDs cited in the answer
  - state['agent_route']: which specialist produced the answer
  - state['retrieved_chunks']: all retrieved context (for citation verification)

Its job:
  1. Polish the answer (fix formatting, ensure citations are properly linked)
  2. Add a source summary footer (which trials were cited)
  3. Optionally do a faithfulness check (does the answer match the context?)

Design decision: the Synthesizer does NOT re-retrieve or re-generate.
It only polishes what the specialist already produced. This keeps it fast
and prevents the synthesizer from introducing hallucinations by "improving"
factual content.

With EchoProvider: just returns the agent_response with a formatted footer.
With real LLM: can do light rewriting for clarity and citation formatting.
"""

import logging
from src.agents.state import GraphState
from src.core.llm import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are a clinical research editor. You receive a draft
answer from a specialist agent and must:

1. Ensure all NCT IDs are properly cited (format: NCT########)
2. Fix any formatting issues (headers, bullet points, spacing)
3. Add a brief "Sources" section at the end listing the trials referenced
4. Keep all factual content exactly as written — do NOT change numbers,
   percentages, or medical conclusions
5. Do NOT add information not in the draft

Return the polished answer followed by a Sources section.

Draft answer to polish:
{draft}

Trials cited: {citations}"""


def synthesizer_node(state: GraphState, llm: LLMProvider) -> dict:
    """
    Polish the specialist agent's response and format citations.
    Returns the final_answer that the API layer returns to the user.
    """
    draft     = state.get("agent_response", "No response generated.")
    citations = state.get("citations", [])
    route     = state.get("agent_route", "unknown")
    reason    = state.get("route_reason", "")

    logger.info(
        f"[Synthesizer] polishing {len(draft)} char response "
        f"from {route!r} agent, {len(citations)} citations"
    )

    # With a real LLM: light rewrite for polish
    # With EchoProvider: the prompt goes in but comes out as canned text,
    # so we just format it ourselves below as a fallback
    citations_str = ", ".join(citations) if citations else "none cited"

    polished = llm.complete(
        messages=[LLMMessage(
            role="user",
            content=f"Please polish this clinical trial research answer:\n\n{draft}"
        )],
        system=SYNTHESIZER_SYSTEM_PROMPT.format(
            draft=draft,
            citations=citations_str,
        ),
        max_tokens=1500,
        temperature=0.0,
    )

    # Build the footer — always appended regardless of LLM output
    footer_parts = [
        f"\n\n---",
        f"**Specialist:** {route} agent | **Routing reason:** {reason}",
    ]
    if citations:
        footer_parts.append(f"**Sources:** {', '.join(citations)}")

    final_answer = polished + "\n".join(footer_parts)

    logger.info(f"[Synthesizer] final answer: {len(final_answer)} chars")
    return {"final_answer": final_answer}


def make_synthesizer_node(llm: LLMProvider):
    def _node(state: GraphState) -> dict:
        return synthesizer_node(state, llm)
    _node.__name__ = "synthesizer"
    return _node