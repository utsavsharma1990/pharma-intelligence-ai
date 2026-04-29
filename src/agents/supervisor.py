"""
Supervisor agent — the router at the center of the multi-agent graph.

Responsibilities:
  1. Read the user's query
  2. Decide which specialist agent should handle it
  3. Return a routing decision (agent name + reason)

The Supervisor does NOT retrieve context or generate answers — it ONLY routes.
This separation keeps each agent's responsibility clear and makes the system
auditable: you can always see WHY a query was routed where it was.

Routing logic (in priority order):
  1. Explicit NCT ID mention -> Trial Search agent (get specific trial data)
  2. Safety/AE keywords      -> Safety agent
  3. Comparison keywords     -> Comparative Analysis agent
  4. Default                 -> Trial Search agent (general lookup)

With a real LLM, the supervisor uses structured output to get a JSON routing
decision. With EchoProvider, the keyword matching in EchoProvider.complete_structured
handles it deterministically.
"""

import logging
from typing import Literal

from src.agents.state import GraphState
from src.core.llm import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

# The valid routing targets — used in the schema enum
VALID_ROUTES = Literal["search", "comparative", "safety"]

# System prompt for the supervisor — gets the LLM to reason about routing
SUPERVISOR_SYSTEM_PROMPT = """You are a clinical trials research supervisor.
Your job is to read the user's question and decide which specialist agent
should answer it. You must choose ONE of these agents:

- "search"      : Find specific trials, get trial details, answer eligibility questions.
                  Use when the user asks about a specific NCT ID, wants to find trials
                  matching criteria, or asks about interventions/endpoints.
- "comparative" : Compare multiple trials, drugs, or sponsors side-by-side.
                  Use when the user asks to compare, contrast, or rank things.
- "safety"      : Analyze adverse events and safety profiles.
                  Use when the user asks about side effects, AEs, toxicity, or safety.

Respond with your routing decision as structured JSON. Be concise in your reason."""


def supervisor_node(state: GraphState, llm: LLMProvider) -> dict:
    """
    LangGraph node: reads the query, returns routing decision.

    Returns a partial state update with `agent_route` and `route_reason`.
    LangGraph merges this into the full GraphState automatically.
    """
    query = state.get("query", "")
    logger.info(f"Supervisor routing query: {query[:80]!r}")

    # Schema for the routing decision — the LLM must return this exact shape
    routing_schema = {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": ["search", "comparative", "safety"],
                "description": "Which specialist agent to call",
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining why you chose this agent",
            },
        },
        "required": ["agent", "reason"],
    }

    try:
        decision = llm.complete_structured(
            messages=[LLMMessage(role="user", content=query)],
            system=SUPERVISOR_SYSTEM_PROMPT,
            schema=routing_schema,
            max_tokens=256,
            temperature=0.0,  # routing decisions should be deterministic
        )
        agent_route  = decision.get("agent", "search")
        route_reason = decision.get("reason", "default routing")

    except Exception as e:
        # If the LLM fails, fall back to keyword-based routing
        logger.warning(f"Supervisor LLM failed, using keyword fallback: {e}")
        agent_route, route_reason = _keyword_route(query)

    logger.info(f"Supervisor routed to: {agent_route!r} ({route_reason})")
    return {
        "agent_route":  agent_route,
        "route_reason": route_reason,
        "metadata":     {"supervised": True},
    }


def _keyword_route(query: str) -> tuple[str, str]:
    """
    Fallback keyword-based router — used when the LLM is unavailable.
    Same logic as EchoProvider but explicit and testable.
    """
    q = query.lower()

    if any(k in q for k in ("adverse", "side effect", "toxicity", "safety", "ae ")):
        return "safety", "keyword match: safety/AE terms detected"

    if any(k in q for k in ("compare", "versus", " vs ", "difference between")):
        return "comparative", "keyword match: comparison terms detected"

    return "search", "default: general trial search"


def make_supervisor_node(llm: LLMProvider):
    """
    Factory that binds the LLM to the supervisor node function.
    Returns a callable that LangGraph can use as a node.

    Why a factory instead of a class?
    LangGraph nodes are just callables: fn(state) -> partial_state.
    A factory that captures `llm` in closure is simpler than a class
    and easier to test (just call the returned function directly).
    """
    def _node(state: GraphState) -> dict:
        return supervisor_node(state, llm)
    _node.__name__ = "supervisor"
    return _node