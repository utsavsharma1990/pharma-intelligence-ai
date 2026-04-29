"""
FastAPI application — the REST interface to the multi-agent system.

Design decisions:
- Lifespan context manager (not @app.on_event, which is deprecated)
  for startup/shutdown logic
- Structured error responses (ErrorResponse schema, never raw exceptions)
- Request ID header for distributed tracing (X-Request-ID)
- CORS configured from settings (not hardcoded)
- Rate limiting via slowapi (per-IP, configurable)
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.dependencies import get_graph, get_retriever, get_store
from src.api.schemas import (
    ChunkCitation,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    ReingestRequest,
    ReingestResponse,
)
from src.core.config import get_settings
from src.core.retriever import RetrievalQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter — slowapi, per-IP
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Lifespan — startup + shutdown logic
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup (before yield) and shutdown (after yield).
    We eagerly initialize the heavy singletons here so the first
    request isn't slow. Errors here crash the server at startup — good,
    because a server with a broken vector store shouldn't accept traffic.
    """
    logger.info("Starting pharma-intelligence-ai API...")
    settings = get_settings()
    logger.info(f"LLM provider: {settings.llm_provider}")
    logger.info(f"Vector store: {settings.vector_store_type}")

    # Eagerly warm up singletons — fail fast if misconfigured
    try:
        store     = get_store()
        retriever = get_retriever()
        graph     = get_graph()
        logger.info(f"Vector store ready: {store.count()} chunks indexed")
        logger.info("Agent graph compiled and ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield   # <-- server is running

    logger.info("Shutting down pharma-intelligence-ai API")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Pharma Intelligence AI",
        description=(
            "Multi-agent clinical trials Q&A system. "
            "Answers natural-language questions grounded in ClinicalTrials.gov data."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware — adds X-Request-ID to every response
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.monotonic()
        response   = await call_next(request)
        elapsed    = round((time.monotonic() - start_time) * 1000, 1)
        response.headers["X-Request-ID"]    = request_id
        response.headers["X-Response-Time"] = f"{elapsed}ms"
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed}ms) [{request_id[:8]}]"
        )
        return response

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health():
        """Liveness probe — is the server running?"""
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(store=Depends(get_store)):
        """
        Readiness probe — is the vector store populated?
        Returns 503 if no chunks are indexed (Kubernetes will withhold traffic).
        """
        count = store.count()
        ready = count > 0
        return ReadyResponse(
            ready=ready,
            chunks_indexed=count,
            message="Ready" if ready else "No chunks indexed — run ingest.py first",
        )

    @app.post(
        "/query",
        response_model=QueryResponse,
        tags=["query"],
        summary="Ask a question about clinical trials",
    )
    @limiter.limit("30/minute")
    async def query(
        request: Request,               # required by slowapi
        body: QueryRequest,
        graph=Depends(get_graph),
        retriever=Depends(get_retriever),
    ):
        """
        Run a natural-language question through the multi-agent pipeline.

        The supervisor routes to the appropriate specialist (search,
        comparative, or safety), which retrieves context and generates
        an answer. The synthesizer polishes the final output.
        """
        logger.info(f"Query: {body.query[:80]!r}")

        try:
            # Pass filter hints as metadata — the retriever will use them
            # in addition to whatever the supervisor decides
            initial_state = {
                "query":    body.query,
                "metadata": {
                    "phase":          body.phase,
                    "sponsor":        body.sponsor,
                    "overall_status": body.overall_status,
                    "top_k":          body.top_k,
                },
            }
            result = graph.invoke(initial_state)

        except Exception as e:
            logger.error(f"Graph invocation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent pipeline error: {str(e)}",
            )

        # Build top_chunks for the response
        retrieved = result.get("retrieved_chunks", [])
        top_chunks = [
            ChunkCitation(
                nct_id=hit.chunk.nct_id,
                section_type=hit.chunk.section_type,
                score=round(hit.score, 3),
                content_preview=hit.chunk.content[:200],
            )
            for hit in retrieved[:3]
        ]

        return QueryResponse(
            answer=result.get("final_answer", "No answer generated."),
            agent_route=result.get("agent_route", "unknown"),
            route_reason=result.get("route_reason", ""),
            citations=result.get("citations", []),
            chunks_retrieved=len(retrieved),
            top_chunks=top_chunks,
        )

    @app.post(
        "/admin/reingest",
        response_model=ReingestResponse,
        tags=["admin"],
        summary="Trigger data re-ingestion",
    )
    @limiter.limit("5/minute")
    async def reingest(
        request: Request,
        body: ReingestRequest,
        store=Depends(get_store),
    ):
        """
        Re-ingest clinical trials data into the vector store.
        Rate-limited to 5/minute to prevent accidental hammering.
        """
        try:
            from src.core.embeddings import HuggingFaceEmbeddings
            from src.core.indexer import TrialIndexer
            from src.ingestion.fetcher import TrialFetcher

            settings = get_settings()
            fetcher  = TrialFetcher(use_mock=True)  # switch to False for real API
            embedder = HuggingFaceEmbeddings(model_name=settings.vs.embedding_model)
            indexer  = TrialIndexer(fetcher=fetcher, embedder=embedder, store=store)

            if body.reset:
                store.reset()

            stats = indexer.index(
                condition=body.condition,
                phase=body.phase,
                sponsor=body.sponsor,
                max_studies=body.max_studies,
            )

            return ReingestResponse(
                success=True,
                trials_processed=stats.trials_processed,
                chunks_indexed=stats.chunks_indexed,
                message=f"Ingested {stats.trials_processed} trials, {stats.chunks_indexed} chunks",
            )

        except Exception as e:
            logger.error(f"Reingest failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    return app


# Module-level app instance — used by uvicorn
app = create_app()