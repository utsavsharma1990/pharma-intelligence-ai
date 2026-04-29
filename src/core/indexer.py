"""
The indexing pipeline — the conductor for the whole ingestion flow.

Glues together every component you've built so far:

    fetch (TrialFetcher)
       ↓
    parse (parse_trial)
       ↓
    chunk (chunk_trial)
       ↓
    embed (EmbeddingProvider.embed_texts)
       ↓
    index (VectorStore.add_chunks)

Design decisions:
  - Batch embeddings: embedding 100 chunks in one call is 10-50x faster than
    100 single calls (the model amortizes overhead across the batch)
  - Delete-then-insert per trial: ensures re-ingesting an updated trial
    cleanly replaces stale chunks (no duplicates, no orphans)
  - Idempotent: running ingest twice in a row produces the exact same DB state
  - Progress reporting: real ingestion runs take minutes, users want feedback
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.core.embeddings import EmbeddingProvider
from src.core.vector_store import VectorStore
from src.ingestion.chunker import chunk_trial
from src.ingestion.fetcher import TrialFetcher
from src.ingestion.parser import parse_trial

logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Summary of an indexing run — useful for logs, dashboards, and tests."""
    trials_processed: int = 0
    chunks_indexed: int = 0
    chunks_deleted: int = 0    # from the delete-before-insert step

    def __str__(self) -> str:
        return (
            f"trials={self.trials_processed} "
            f"chunks_indexed={self.chunks_indexed} "
            f"chunks_replaced={self.chunks_deleted}"
        )


class TrialIndexer:
    """
    Orchestrates the full ingestion pipeline.

    Usage:
        indexer = TrialIndexer(
            fetcher=TrialFetcher(use_mock=True),
            embedder=HuggingFaceEmbeddings(),
            store=ChromaStore(...),
        )
        stats = indexer.index(condition="lung cancer", phase="PHASE3")
    """

    def __init__(
        self,
        fetcher: TrialFetcher,
        embedder: EmbeddingProvider,
        store: VectorStore,
        embed_batch_size: int = 64,
    ):
        self.fetcher = fetcher
        self.embedder = embedder
        self.store = store
        self.embed_batch_size = embed_batch_size

    def index(
        self,
        condition: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        sponsor: Optional[str] = None,
        max_studies: Optional[int] = None,
    ) -> IndexingStats:
        """Run the full pipeline. Returns stats about what was indexed."""
        stats = IndexingStats()

        # We accumulate chunks across trials, then embed in big batches.
        # Embedding 64 chunks in one model call >> 64 single calls.
        pending_chunks = []
        BATCH = self.embed_batch_size

        for raw_study in self.fetcher.fetch_all(
            condition=condition,
            phase=phase,
            status=status,
            sponsor=sponsor,
            max_studies=max_studies,
        ):
            trial = parse_trial(raw_study)
            if not trial.nct_id:
                logger.warning("Skipping trial with no NCT ID")
                continue

            # Delete-before-insert pattern: if this trial was previously indexed,
            # we wipe its old chunks first. Keeps the index clean on re-runs.
            deleted = self.store.delete_by_nct_id(trial.nct_id)
            stats.chunks_deleted += deleted

            chunks = chunk_trial(trial)
            pending_chunks.extend(chunks)
            stats.trials_processed += 1

            # Flush in batches — keeps memory bounded for large ingestions
            while len(pending_chunks) >= BATCH:
                batch = pending_chunks[:BATCH]
                pending_chunks = pending_chunks[BATCH:]
                self._index_batch(batch)
                stats.chunks_indexed += len(batch)

        # Flush any remaining chunks
        if pending_chunks:
            self._index_batch(pending_chunks)
            stats.chunks_indexed += len(pending_chunks)

        logger.info(f"Indexing complete: {stats}")
        return stats

    def _index_batch(self, chunks: list) -> None:
        """Embed and store a batch of chunks."""
        if not chunks:
            return
        texts = [c.content for c in chunks]
        vectors = self.embedder.embed_texts(texts)
        self.store.add_chunks(chunks, vectors)
        logger.info(f"Indexed batch: {len(chunks)} chunks")