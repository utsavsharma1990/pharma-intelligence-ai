"""
Pydantic schemas for all FastAPI request and response bodies.

Design decision: separate schemas from the domain models (ParsedTrial,
TrialChunk, etc.). API schemas are shaped for the consumer (what makes
a good API response), domain models are shaped for the business logic.
These are different concerns that happen to overlap — keeping them
separate means you can evolve the API without breaking internals.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    A natural-language question about clinical trials.

    Filters are optional. If provided, they're passed directly to the
    retriever as metadata constraints on top of semantic search.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about clinical trials",
        examples=["What adverse events were reported in pembrolizumab trials?"],
    )
    phase: Optional[str] = Field(
        None,
        description="Filter by trial phase: PHASE1, PHASE2, PHASE3, PHASE4",
        examples=["PHASE3"],
    )
    sponsor: Optional[str] = Field(
        None,
        description="Filter by sponsor name (partial match supported)",
        examples=["Merck"],
    )
    overall_status: Optional[str] = Field(
        None,
        description="Filter by status: RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED",
        examples=["RECRUITING"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve",
    )


class ReingestRequest(BaseModel):
    """Parameters for the /admin/reingest endpoint."""
    condition: Optional[str] = Field(
        None,
        description="Condition to search (e.g. 'lung cancer')",
    )
    phase: Optional[str] = None
    sponsor: Optional[str] = None
    max_studies: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Maximum number of studies to ingest",
    )
    reset: bool = Field(
        default=False,
        description="If true, wipe existing index before ingesting",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ChunkCitation(BaseModel):
    """A single retrieved chunk cited in the answer."""
    nct_id: str
    section_type: str
    score: float
    content_preview: str   # first 200 chars of the chunk content


class QueryResponse(BaseModel):
    """The full response to a /query request."""
    answer: str = Field(..., description="The agent's synthesized answer")
    agent_route: str = Field(..., description="Which specialist handled the query")
    route_reason: str = Field(..., description="Why the supervisor chose that agent")
    citations: list[str] = Field(
        default_factory=list,
        description="NCT IDs mentioned in the answer",
    )
    chunks_retrieved: int = Field(
        ...,
        description="Number of chunks retrieved from the vector store",
    )
    top_chunks: list[ChunkCitation] = Field(
        default_factory=list,
        description="Top 3 retrieved chunks for transparency",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    ready: bool
    chunks_indexed: int
    message: str


class ReingestResponse(BaseModel):
    success: bool
    trials_processed: int
    chunks_indexed: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None