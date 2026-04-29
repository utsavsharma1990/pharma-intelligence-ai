"""
FastAPI dependency injection for shared resources.

Design decision: use FastAPI's dependency injection (Depends) instead of
module-level globals. Benefits:
  - Easy to override in tests (just swap the dependency)
  - Lazy initialization (resources created on first request, not import)
  - Explicit about what each endpoint needs

All three resources (graph, retriever, store) are singletons — created
once at startup and reused across requests. This is important because:
  - Loading the embedding model takes ~2s
  - ChromaDB connections have overhead
  - LLM SDK clients maintain connection pools
"""

from functools import lru_cache
from pathlib import Path

from src.agents.graph import build_graph
from src.core.chroma_store import ChromaStore
from src.core.config import get_settings
from src.core.embeddings import HuggingFaceEmbeddings
from src.core.llm import get_llm_provider
from src.core.retriever import Retriever


@lru_cache(maxsize=1)
def get_store() -> ChromaStore:
    settings = get_settings()
    return ChromaStore(
        persist_dir=Path(settings.vs.chroma_persist_dir),
        collection_name=settings.vs.collection_name,
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    settings = get_settings()
    embedder = HuggingFaceEmbeddings(model_name=settings.vs.embedding_model)
    return Retriever(embedder, get_store())


@lru_cache(maxsize=1)
def get_graph():
    """Build and return the compiled LangGraph agent graph."""
    llm = get_llm_provider()
    retriever = get_retriever()
    return build_graph(llm, retriever)