"""
Shared state for the pharma-intelligence multi-agent graph.

Design decision: TypedDict over dataclass or Pydantic model.
LangGraph requires TypedDict for its state annotations — it uses the
type hints to determine how state keys are merged across nodes.

The `Annotated[list, operator.add]` pattern is LangGraph's way of saying
"when two nodes both write to this key, ADD their values instead of
overwriting". This is how we accumulate retrieved chunks across multiple
agent calls without losing earlier results.

Every node receives the FULL state and returns a PARTIAL state
(only the keys it changes). LangGraph merges the partial update
into the full state before calling the next node.
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from src.core.vector_store import SearchResult


class GraphState(TypedDict, total=False):
    """
    The shared state dict that flows between all agents in the graph.

    Keys:
        query           : the original user question (never modified)
        agent_route     : which specialist the supervisor chose
        route_reason    : supervisor's reasoning (useful for debugging)
        retrieved_chunks: accumulated retrieval results (additive across agents)
        agent_response  : the specialist agent's raw answer
        final_answer    : the synthesizer's polished final output
        citations       : list of NCT IDs cited in the answer
        error           : set if any node fails — triggers graceful error path
        metadata        : arbitrary dict for agents to pass context to each other
    """
    query:            str
    agent_route:      str    # "search" | "comparative" | "safety" | "synthesizer"
    route_reason:     str
    # Annotated with operator.add = LangGraph accumulates across node writes
    retrieved_chunks: Annotated[list[SearchResult], operator.add]
    agent_response:   str
    final_answer:     str
    citations:        Annotated[list[str], operator.add]
    error:            Optional[str]
    metadata:         dict