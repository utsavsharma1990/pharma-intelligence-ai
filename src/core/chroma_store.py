"""
ChromaDB implementation of the VectorStore interface.

Why ChromaDB for local development?
  - Zero infrastructure: persists to a local directory, no Docker needed
  - SQLite-backed: trivial to inspect, debug, version
  - Native metadata filtering: `where={"phase": "PHASE3"}` works out of the box
  - Free, MIT-licensed, well-maintained

Why not it for production?
  - Single-node only (no horizontal scaling)
  - Limited filter expressiveness vs Pinecone
  - The ABC layer means swapping is a one-config-change away

Important Chroma quirks we handle:
  - Metadata values must be str/int/float/bool (no None, no nested dicts).
    Our TrialChunk.to_metadata_dict() already strips Nones.
  - Distances are returned as L2 by default; we convert to cosine similarity.
  - Collections are persisted via `persist_directory`; if you delete the
    collection, you must re-instantiate.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.vector_store import SearchResult, VectorStore
from src.ingestion.models import TrialChunk

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """
    ChromaDB-backed implementation of VectorStore.

    Usage:
        store = ChromaStore(persist_dir=Path("./chroma_db"), collection="clinical_trials")
        store.add_chunks(chunks, embeddings)
        hits = store.search(query_embedding, top_k=5, filters={"phase": "PHASE3"})
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # PersistentClient automatically saves to disk; no .persist() call needed.
        # anonymized_telemetry=False because we don't want surprise outbound traffic
        # in a corporate environment.
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # 'cosine' so we can return cosine similarity directly without
            # converting from euclidean distance.
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[TrialChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        # Chroma requires three parallel arrays: ids, embeddings, metadatas, documents.
        # Using upsert (not add) so re-ingesting the same chunk_id overwrites
        # instead of crashing on duplicate-id error.
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[c.to_metadata_dict() for c in chunks],
        )
        logger.info(f"Indexed {len(chunks)} chunks into '{self.collection_name}'")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        # Chroma's query API: where={"key": "value"} for single-field filters,
        # where={"$and": [{...}, {...}]} for multiple. We auto-build the right form.
        where = self._build_where(filters)

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns lists-of-lists (one per query). We sent one query → take [0].
        ids       = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        hits: list[SearchResult] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            chunk = self._reconstruct_chunk(chunk_id, doc, meta)
            # Cosine distance → similarity: similarity = 1 - distance
            score = max(0.0, 1.0 - float(dist))
            hits.append(SearchResult(chunk=chunk, score=score))

        return hits

    def delete_by_nct_id(self, nct_id: str) -> int:
        """
        Delete all chunks for a given trial.
        Used during re-ingestion to avoid duplicate/stale data.
        """
        # First, find what we'd delete (so we can return a count)
        existing = self._collection.get(where={"nct_id": nct_id}, include=[])
        n = len(existing["ids"])

        if n > 0:
            self._collection.delete(where={"nct_id": nct_id})
        return n

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Tests + dev only."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # already gone
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(filters: Optional[dict]) -> Optional[dict]:
        """
        Convert our flat {key: value} filter dict to Chroma's where syntax.
        Chroma quirk: a single-key filter is `{"k": "v"}`, but multiple keys
        require `{"$and": [{"k1": "v1"}, {"k2": "v2"}]}`. We handle both.
        """
        if not filters:
            return None
        if len(filters) == 1:
            return dict(filters)
        return {"$and": [{k: v} for k, v in filters.items()]}

    @staticmethod
    def _reconstruct_chunk(
        chunk_id: str,
        document: str,
        metadata: dict,
    ) -> TrialChunk:
        """
        Rebuild a TrialChunk from what Chroma stored.
        We stored content as `document`, and everything else as metadata.
        """
        return TrialChunk(
            chunk_id=chunk_id,
            content=document,
            nct_id=metadata.get("nct_id", ""),
            section_type=metadata.get("section_type", ""),
            phase=metadata.get("phase"),
            overall_status=metadata.get("overall_status"),
            sponsor=metadata.get("sponsor"),
            sponsor_class=metadata.get("sponsor_class"),
            enrollment_count=metadata.get("enrollment_count"),
            brief_title=metadata.get("brief_title"),
        )