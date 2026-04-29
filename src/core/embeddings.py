"""
Embedding provider abstraction.

Same Strategy pattern as the vector store: define an ABC, implement once,
switch backends via config. Today: HuggingFace local. Tomorrow: OpenAI,
Cohere, Voyage AI — all without touching the indexer or the agents.

Why local embeddings as the default?
  - Zero API cost (a real concern at scale: 100K chunks = $$ on hosted APIs)
  - No network dependency (works in offline / air-gapped environments)
  - Deterministic output for a given model version (reproducible tests)
  - Privacy: clinical trial data never leaves your machine

Why an ABC instead of just calling sentence-transformers everywhere?
  - Easy to mock in tests
  - Easy to swap to hosted APIs when scale demands
  - Forces a clear contract: one method, embed_texts(list[str]) → list[list[float]]
"""

from abc import ABC, abstractmethod
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """
    Abstract interface for any embedding model.

    Implementations: HuggingFaceEmbeddings (default), and (future) OpenAIEmbeddings,
    CohereEmbeddings, VoyageEmbeddings, etc.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.
        Returns a list of vectors; len(result) == len(texts).
        Order is preserved: result[i] is the embedding of texts[i].
        """
        ...

    def embed_text(self, text: str) -> list[float]:
        """Convenience: embed a single text. Default implementation calls embed_texts."""
        return self.embed_texts([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality (e.g. 384 for MiniLM, 1536 for OpenAI ada-002)."""
        ...


class HuggingFaceEmbeddings(EmbeddingProvider):
    """
    Local HuggingFace embeddings via sentence-transformers.

    The model is loaded lazily on first use — saves startup time when the
    rest of the app (FastAPI server, MCP server) doesn't need embeddings yet.

    Models auto-download to ~/.cache/huggingface/hub/ on first use.
    Subsequent runs use the cached model — no network needed.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None       # lazy-loaded
        self._dimension = None   # discovered after first load

    def _load_model(self):
        """Lazy load: only import + download when actually needed."""
        if self._model is None:
            # Import here (not at top of file) so just importing this module
            # doesn't trigger torch loading. Keeps `from src.core.embeddings`
            # snappy even if you never call embed_texts.
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded — {self._dimension} dims")
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        # convert_to_numpy=False returns a list of tensors; .tolist() yields
        # Python lists which is what every vector store expects.
        # normalize_embeddings=True makes cosine similarity numerically equivalent
        # to dot product, which is what Chroma's hnsw:cosine assumes.
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension


# ---------------------------------------------------------------------------
# Factory: returns a singleton based on config.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """
    Default factory. Reads model name from settings.
    Cached so we don't reload the model on every call.
    """
    from src.core.config import get_settings
    settings = get_settings().vs
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)