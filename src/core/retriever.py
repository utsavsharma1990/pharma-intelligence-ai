"""
High-level retrieval layer that combines:
  - semantic search (via embeddings + vector store)
  - metadata filtering (phase, sponsor, section_type, nct_id)
  - light query parsing (auto-extract NCT IDs from natural language)

This is the surface every agent will use. Agents shouldn't know about
ChromaDB or sentence-transformers — they just call retriever.search(...).

Design decisions:
  - Build filters from a structured `RetrievalQuery` instead of raw kwargs.
    Lets us validate, log, and extend without breaking caller signatures.
  - Auto-detect NCT IDs in the natural language query. If the user says
    "What are eligibility criteria for NCT05123456?", we both search
    semantically AND filter to that exact NCT ID — best of both worlds.
  - Section routing is OPTIONAL. The agents will set it explicitly
    (Safety agent → adverse_events), but the API also exposes a free-form
    search for the Trial Search agent.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.core.embeddings import EmbeddingProvider
from src.core.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


# Match NCT IDs anywhere in a query: "NCT" + 8 digits.
# Real-world queries paste IDs in many forms; case-insensitive is safer.
# Real NCT IDs are 8 digits, but we accept 3-10 to support test fixtures
# and any future format changes. The leading \bNCT and digit-only body
# still keep this very specific — no real word will match.
NCT_PATTERN = re.compile(r"\bNCT\d{3,10}\b", re.IGNORECASE)


@dataclass
class RetrievalQuery:
    """
    Structured query object — replaces a wall of optional kwargs.

    Why a dataclass instead of just kwargs?
      - Self-documenting: you see the contract without reading the impl
      - Validatable: can add __post_init__ checks centrally
      - Loggable: __repr__ gives a readable trace for debugging
    """
    text: str
    nct_id: Optional[str] = None
    phase: Optional[str] = None
    sponsor: Optional[str] = None
    overall_status: Optional[str] = None
    section_type: Optional[str] = None
    top_k: int = 5

    # Internal: filters get auto-populated from `text` if not set explicitly
    extracted_nct_ids: list[str] = field(default_factory=list)


class Retriever:
    """
    The retrieval surface. Combines semantic search + metadata filtering.

    Usage:
        retriever = Retriever(embedder, vector_store)
        hits = retriever.search(RetrievalQuery(
            text="lung problems in immunotherapy trials",
            section_type="adverse_events",
            phase="PHASE3",
            top_k=5,
        ))
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore):
        self.embedder = embedder
        self.store = store

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def search(self, query: RetrievalQuery) -> list[SearchResult]:
        """
        Run a hybrid search: semantic similarity over the text, with metadata
        filters applied as exact-match constraints.
        """
        # 1. Auto-extract NCT IDs from the query text (if any)
        self._auto_extract_nct(query)

        # 2. Build the filter dict for the vector store
        filters = self._build_filters(query)

        # 3. Embed the query text
        query_vec = self.embedder.embed_text(query.text)

        # 4. Run the actual vector store search
        hits = self.store.search(
            query_embedding=query_vec,
            top_k=query.top_k,
            filters=filters,
        )

        logger.info(
            f"Retrieval: text={query.text[:60]!r} filters={filters} "
            f"\u2192 {len(hits)} hits"
        )
        return hits

    # ------------------------------------------------------------------
    # Convenience wrappers — what agents will call most often
    # ------------------------------------------------------------------

    def search_text(self, text: str, top_k: int = 5) -> list[SearchResult]:
        """Plain semantic search, no filters. The Trial Search agent uses this."""
        return self.search(RetrievalQuery(text=text, top_k=top_k))

    def search_safety(
        self,
        text: str,
        nct_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Filter to AE chunks only. The Safety agent uses this."""
        return self.search(RetrievalQuery(
            text=text,
            section_type="adverse_events",
            nct_id=nct_id,
            top_k=top_k,
        ))

    def search_eligibility(
        self,
        text: str,
        nct_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Filter to eligibility chunks only."""
        return self.search(RetrievalQuery(
            text=text,
            section_type="eligibility",
            nct_id=nct_id,
            top_k=top_k,
        ))

    def get_trial_chunks(self, nct_id: str, top_k: int = 20) -> list[SearchResult]:
        """
        Get all chunks for a specific trial. Used when the agent needs the
        whole trial as context (e.g., 'tell me everything about NCT05123456').
        Top-k is set high to capture all sections; the embedding similarity
        is irrelevant here \u2014 the filter does all the work.
        """
        return self.search(RetrievalQuery(
            text=nct_id,        # text doesn't matter much; filter dominates
            nct_id=nct_id,
            top_k=top_k,
        ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_extract_nct(query: RetrievalQuery) -> None:
        """
        Find NCT IDs mentioned in the query text. If found and the user
        didn't explicitly set nct_id, use the first one as a hard filter.

        Why first one? In multi-NCT queries ("compare NCT001 vs NCT002"),
        the comparative agent should issue separate searches for each.
        That's a Phase 4 concern; here we err toward the conservative default.
        """
        ids = [m.group(0).upper() for m in NCT_PATTERN.finditer(query.text)]
        query.extracted_nct_ids = ids
        if ids and query.nct_id is None:
            query.nct_id = ids[0]

    @staticmethod
    def _build_filters(query: RetrievalQuery) -> Optional[dict]:
        """Pack non-None filter fields into a dict the vector store accepts."""
        filters: dict = {}
        if query.nct_id:         filters["nct_id"]         = query.nct_id
        if query.phase:          filters["phase"]          = query.phase
        if query.sponsor:        filters["sponsor"]        = query.sponsor
        if query.overall_status: filters["overall_status"] = query.overall_status
        if query.section_type:   filters["section_type"]   = query.section_type
        return filters or None