"""
MCP tool implementations — thin wrappers over the agent graph.

Each tool builds a query string and invokes the agent graph,
returning the final_answer as a string. The MCP server just
needs strings back (or dicts that get JSON-serialized).

Design: tools are intentionally thin. All the real logic lives
in the agents. Tools just handle:
  - Translating MCP arguments into a query string
  - Calling the graph
  - Extracting the right output field
  - Formatting errors as strings (MCP clients display these to users)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _invoke_graph(graph, query: str) -> str:
    """Run the agent graph and return the final answer."""
    try:
        result = graph.invoke({"query": query})
        return result.get("final_answer", "No answer generated.")
    except Exception as e:
        logger.error(f"Graph invocation error: {e}", exc_info=True)
        return f"Error processing query: {str(e)}"


def search_trials(
    graph,
    query: str,
    phase: Optional[str] = None,
    status: Optional[str] = None,
    sponsor: Optional[str] = None,
) -> str:
    """
    Natural language search across all indexed clinical trials.

    Builds a structured query string that includes any filters,
    then lets the agent graph handle retrieval and synthesis.
    """
    # Enrich the query with filter context so the supervisor/agents
    # can use it as hints even though they don't read structured params directly
    parts = [query]
    if phase:    parts.append(f"phase: {phase}")
    if status:   parts.append(f"status: {status}")
    if sponsor:  parts.append(f"sponsor: {sponsor}")

    enriched = " | ".join(parts)
    logger.info(f"[MCP:search_trials] query={enriched[:80]!r}")
    return _invoke_graph(graph, enriched)


def get_trial_details(graph, nct_id: str) -> str:
    """
    Fetch full details for a specific trial by NCT ID.
    The retriever auto-extracts NCT IDs from the query text.
    """
    query = (
        f"Give me complete details about trial {nct_id} including "
        f"eligibility criteria, interventions, endpoints, and current status."
    )
    logger.info(f"[MCP:get_trial_details] nct_id={nct_id}")
    return _invoke_graph(graph, query)


def compare_trials(graph, nct_ids: list[str], aspect: Optional[str] = None) -> str:
    """
    Compare 2+ trials side by side.
    `aspect` focuses the comparison: 'safety', 'endpoints', 'eligibility', etc.
    """
    if len(nct_ids) < 2:
        return "Please provide at least 2 NCT IDs to compare."

    ids_str = " and ".join(nct_ids)
    aspect_str = f" focusing on {aspect}" if aspect else ""
    query = f"Compare trials {ids_str}{aspect_str} side by side."

    logger.info(f"[MCP:compare_trials] ids={nct_ids} aspect={aspect!r}")
    return _invoke_graph(graph, query)


def get_safety_profile(
    graph,
    nct_id: Optional[str] = None,
    drug_name: Optional[str] = None,
) -> str:
    """
    Extract adverse events and safety data.
    Accepts either a specific NCT ID or a drug name.
    """
    if nct_id:
        query = f"What are the adverse events and safety profile for trial {nct_id}?"
    elif drug_name:
        query = f"What adverse events and safety data exist for {drug_name} trials?"
    else:
        return "Please provide either an NCT ID or a drug name."

    logger.info(f"[MCP:get_safety_profile] nct_id={nct_id} drug={drug_name!r}")
    return _invoke_graph(graph, query)


def find_eligible_trials(
    graph,
    age: Optional[int] = None,
    condition: Optional[str] = None,
    prior_therapies: Optional[list[str]] = None,
    ecog_status: Optional[int] = None,
) -> str:
    """
    Find trials a patient may be eligible for based on their profile.
    Builds a natural language patient profile and searches for matching trials.
    """
    profile_parts = ["Find clinical trials for a patient with the following profile:"]

    if condition:       profile_parts.append(f"Condition: {condition}")
    if age:             profile_parts.append(f"Age: {age} years old")
    if ecog_status is not None:
        profile_parts.append(f"ECOG performance status: {ecog_status}")
    if prior_therapies:
        therapies = ", ".join(prior_therapies)
        profile_parts.append(f"Prior therapies: {therapies}")

    profile_parts.append(
        "What trials are they likely eligible for? "
        "Check inclusion and exclusion criteria carefully."
    )

    query = " ".join(profile_parts)
    logger.info(f"[MCP:find_eligible_trials] condition={condition!r} age={age}")
    return _invoke_graph(graph, query)