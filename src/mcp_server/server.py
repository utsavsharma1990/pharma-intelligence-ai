"""
MCP (Model Context Protocol) server for pharma-intelligence-ai.

Exposes 5 tools that any MCP-compatible client (Claude Desktop, etc.)
can call to query the clinical trials database.

How MCP works:
  1. Claude Desktop reads your MCP config (claude_desktop_config.json)
  2. It starts this server as a subprocess
  3. The server announces its tools via the `list_tools` handler
  4. When Claude calls a tool, `call_tool` is invoked
  5. The result is returned as a list of TextContent objects

To use with Claude Desktop, add to ~/AppData/Roaming/Claude/claude_desktop_config.json:
{
  "mcpServers": {
    "pharma-intelligence": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "C:/path/to/pharma-intelligence-ai"
    }
  }
}

Why MCP over just the REST API?
  - Claude Desktop can call tools mid-conversation without user copy-pasting URLs
  - Tools appear natively in the Claude UI alongside other connected tools
  - No need to run a separate web server process for local dev
  - Standardized protocol = works with any future MCP-compatible client
"""

import asyncio
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from src.agents.graph import build_graph
from src.core.chroma_store import ChromaStore
from src.core.config import get_settings
from src.core.embeddings import HuggingFaceEmbeddings
from src.core.llm import get_llm_provider
from src.core.retriever import Retriever
from src.mcp_server import tools as tool_impl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the agent graph (shared across all tool calls in this process)
# ---------------------------------------------------------------------------

def _build_agent_graph():
    settings  = get_settings()
    llm       = get_llm_provider()
    embedder  = HuggingFaceEmbeddings(model_name=settings.vs.embedding_model)
    store     = ChromaStore(
        persist_dir=Path(settings.vs.chroma_persist_dir),
        collection_name=settings.vs.collection_name,
    )
    retriever = Retriever(embedder, store)
    return build_graph(llm, retriever)


# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------

def create_mcp_server() -> Server:
    server = Server("pharma-intelligence-ai")
    graph  = _build_agent_graph()

    # ------------------------------------------------------------------
    # Tool definitions — what the MCP client sees
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_trials",
                description=(
                    "Search for clinical trials using natural language. "
                    "Returns a synthesized answer about matching trials."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query":   {"type": "string",
                                    "description": "Natural language search query"},
                        "phase":   {"type": "string",
                                    "description": "Trial phase filter: PHASE1, PHASE2, PHASE3, PHASE4",
                                    "enum": ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]},
                        "status":  {"type": "string",
                                    "description": "Trial status filter",
                                    "enum": ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"]},
                        "sponsor": {"type": "string",
                                    "description": "Sponsor name filter"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_trial_details",
                description=(
                    "Get complete details for a specific clinical trial by NCT ID. "
                    "Returns eligibility criteria, interventions, endpoints, and status."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nct_id": {"type": "string",
                                   "description": "NCT identifier (e.g. NCT05123456)"},
                    },
                    "required": ["nct_id"],
                },
            ),
            types.Tool(
                name="compare_trials",
                description=(
                    "Compare two or more clinical trials side by side. "
                    "Can focus on safety, endpoints, eligibility, or overall design."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nct_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "description": "List of NCT IDs to compare (minimum 2)",
                        },
                        "aspect": {
                            "type": "string",
                            "description": "What to focus the comparison on",
                            "enum": ["safety", "endpoints", "eligibility", "design", "overall"],
                        },
                    },
                    "required": ["nct_ids"],
                },
            ),
            types.Tool(
                name="get_safety_profile",
                description=(
                    "Extract adverse events and safety data for a specific trial "
                    "or drug. Returns serious and common AEs with incidence rates."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "nct_id":    {"type": "string",
                                      "description": "NCT ID of the trial"},
                        "drug_name": {"type": "string",
                                      "description": "Drug name to search across trials"},
                    },
                },
            ),
            types.Tool(
                name="find_eligible_trials",
                description=(
                    "Find clinical trials a patient may be eligible for "
                    "based on their profile (age, condition, prior therapies, ECOG)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "Patient's primary diagnosis (e.g. 'NSCLC', 'CLL')",
                        },
                        "age": {
                            "type": "integer",
                            "description": "Patient age in years",
                        },
                        "ecog_status": {
                            "type": "integer",
                            "description": "ECOG performance status (0-4)",
                            "minimum": 0,
                            "maximum": 4,
                        },
                        "prior_therapies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of prior treatments received",
                        },
                    },
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool call handler
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict,
    ) -> list[types.TextContent]:
        """Route incoming tool calls to the right implementation."""
        logger.info(f"[MCP] tool={name!r} args={list(arguments.keys())}")

        try:
            if name == "search_trials":
                result = tool_impl.search_trials(
                    graph,
                    query=arguments["query"],
                    phase=arguments.get("phase"),
                    status=arguments.get("status"),
                    sponsor=arguments.get("sponsor"),
                )

            elif name == "get_trial_details":
                result = tool_impl.get_trial_details(
                    graph,
                    nct_id=arguments["nct_id"],
                )

            elif name == "compare_trials":
                result = tool_impl.compare_trials(
                    graph,
                    nct_ids=arguments["nct_ids"],
                    aspect=arguments.get("aspect"),
                )

            elif name == "get_safety_profile":
                result = tool_impl.get_safety_profile(
                    graph,
                    nct_id=arguments.get("nct_id"),
                    drug_name=arguments.get("drug_name"),
                )

            elif name == "find_eligible_trials":
                result = tool_impl.find_eligible_trials(
                    graph,
                    age=arguments.get("age"),
                    condition=arguments.get("condition"),
                    prior_therapies=arguments.get("prior_therapies"),
                    ecog_status=arguments.get("ecog_status"),
                )

            else:
                result = f"Unknown tool: {name!r}"

        except Exception as e:
            logger.error(f"Tool {name!r} failed: {e}", exc_info=True)
            result = f"Tool error: {str(e)}"

        return [types.TextContent(type="text", text=result)]

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting pharma-intelligence MCP server...")
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())