"""
LangGraph StateGraph — full multi-agent graph with real specialist nodes.
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.comparative_agent import make_comparative_node
from src.agents.safety_agent import make_safety_node
from src.agents.search_agent import make_search_node
from src.agents.state import GraphState
from src.agents.supervisor import make_supervisor_node
from src.agents.synthesizer_agent import make_synthesizer_node
from src.core.llm import LLMProvider
from src.core.retriever import Retriever

logger = logging.getLogger(__name__)


def _route_after_supervisor(
    state: GraphState,
) -> Literal["search", "comparative", "safety"]:
    route = state.get("agent_route", "search")
    if route not in ("search", "comparative", "safety"):
        logger.warning(f"Unknown route {route!r}, defaulting to 'search'")
        return "search"
    return route  # type: ignore[return-value]


def build_graph(llm: LLMProvider, retriever: Retriever):
    """
    Assemble and compile the full multi-agent StateGraph.
    All stub nodes replaced with real specialist implementations.
    """
    graph = StateGraph(GraphState)

    # --- Nodes ---
    graph.add_node("supervisor",  make_supervisor_node(llm))
    graph.add_node("search",      make_search_node(llm, retriever))
    graph.add_node("comparative", make_comparative_node(llm, retriever))
    graph.add_node("safety",      make_safety_node(llm, retriever))
    graph.add_node("synthesizer", make_synthesizer_node(llm))

    # --- Edges ---
    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "search":      "search",
            "comparative": "comparative",
            "safety":      "safety",
        },
    )

    graph.add_edge("search",      "synthesizer")
    graph.add_edge("comparative", "synthesizer")
    graph.add_edge("safety",      "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()