"""
Vector store abstraction layer.

Design decision: Define an ABC (abstract base class) so we can swap
backends (ChromaDB, Pinecone, Weaviate, pgvector) without changing
any code outside this layer.

The contract:
  - add_chunks(chunks, embeddings): index pre-embedded chunks
  - search(query_embedding, top_k, filters): nearest-neighbor search
  - delete_by_nct_id(nct_id): bulk delete (for re-ingestion)
  - count(): total chunks indexed

Why pre-computed embeddings (not raw text)?
  Separation of concerns. The vector store should be agnostic to which
  embedding model produced the vectors. Activity 9 builds the embedding
  pipeline that wraps this layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.ingestion.models import TrialChunk


@dataclass
class SearchResult:
    """A single retrieval hit with its similarity score."""
    chunk: TrialChunk
    score: float       # cosine similarity, 0..1 (higher = more similar)


class VectorStore(ABC):
    """
    Abstract interface every vector store backend must implement.

    Implementations: ChromaStore (local dev), PineconeStore (prod, future).
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[TrialChunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Index a batch of chunks with their pre-computed embeddings.
        chunks[i] must correspond to embeddings[i] — same order, same length.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most similar chunks to the query embedding.

        Args:
            filters: optional metadata constraints, e.g.
                     {"phase": "PHASE3", "section_type": "adverse_events"}
                     Implementation must enforce these as exact-match filters.
        """
        ...

    @abstractmethod
    def delete_by_nct_id(self, nct_id: str) -> int:
        """
        Delete all chunks belonging to a given trial.
        Returns the number of chunks deleted.
        Used during re-ingestion: delete-then-insert avoids stale data.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks indexed."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Wipe everything. Use carefully — primarily for tests."""
        ...