"""
Run the full indexing pipeline.

Usage:
    # Use mock data (default — corporate networks)
    python ingest.py

    # Use real ClinicalTrials.gov API
    python ingest.py --real

    # Limit to N studies (good for testing)
    python ingest.py --max 5

    # Specific condition / phase / sponsor
    python ingest.py --condition "lung cancer" --phase PHASE3
"""

import argparse
import logging
from pathlib import Path

from src.core.chroma_store import ChromaStore
from src.core.config import get_settings
from src.core.embeddings import HuggingFaceEmbeddings
from src.core.indexer import TrialIndexer
from src.ingestion.fetcher import TrialFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest clinical trials data")
    parser.add_argument("--real", action="store_true",
                        help="Use real CT.gov API (default: mock data)")
    parser.add_argument("--max",  type=int, default=None,
                        help="Maximum number of trials to ingest")
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--phase",     type=str, default=None,
                        help="e.g. PHASE3")
    parser.add_argument("--status",    type=str, default=None,
                        help="e.g. RECRUITING")
    parser.add_argument("--sponsor",   type=str, default=None)
    parser.add_argument("--reset", action="store_true",
                        help="Wipe existing index before ingesting")
    args = parser.parse_args()

    settings = get_settings()

    # Construct each layer explicitly — no hidden globals
    fetcher  = TrialFetcher(use_mock=not args.real)
    embedder = HuggingFaceEmbeddings(model_name=settings.vs.embedding_model)
    store    = ChromaStore(
        persist_dir=Path(settings.vs.chroma_persist_dir),
        collection_name=settings.vs.collection_name,
    )

    if args.reset:
        print("⚠️  Resetting existing index…")
        store.reset()

    indexer = TrialIndexer(fetcher=fetcher, embedder=embedder, store=store)

    print("🔄 Starting ingestion…")
    stats = indexer.index(
        condition=args.condition,
        phase=args.phase,
        status=args.status,
        sponsor=args.sponsor,
        max_studies=args.max,
    )

    print(f"\n✅ Done — {stats}")
    print(f"   Total chunks in store: {store.count()}")


if __name__ == "__main__":
    main()