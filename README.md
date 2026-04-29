# 🧬 Pharma Intelligence AI

> A production-grade multi-agent AI system that pulls real clinical trials data from ClinicalTrials.gov and answers complex pharma research questions — grounded in real data, never hallucinated.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does (in plain English)

Ask questions like a clinical researcher, get answers grounded in real trial data:

- *"Show me all Phase III lung cancer trials sponsored by Merck that are recruiting"*
- *"Compare the safety profiles of pembrolizumab and nivolumab in NSCLC trials"*
- *"What are the eligibility criteria for NCT05123456?"*
- *"Find all trials testing BTK inhibitors in CLL"*
- *"What adverse events were reported in the ibrutinib trial?"*

A supervisor LLM routes each question to the right specialist agent, which retrieves grounded context from a vector database of real ClinicalTrials.gov data, and a synthesizer polishes the final cited answer.

---

## Architecture

```
  ClinicalTrials.gov v2 API
           │
           ▼
  ┌──────────────────┐
  │  Ingestion       │  ◄── retry + exponential backoff + rate limiting
  │  (httpx + tenacity)     + persistent disk cache
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Parser          │  raw JSON → ParsedTrial domain objects
  └────────┬─────────┘  (anti-corruption layer pattern)
           │
           ▼
  ┌──────────────────┐
  │  Domain-aware    │  preserves Eligibility / Endpoints / AE sections
  │  Chunker         │  5 sections per trial, rich metadata per chunk
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Embeddings      │  sentence-transformers/all-MiniLM-L6-v2 (local, free)
  │  + ChromaDB      │  hybrid search: semantic + metadata filters
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────┐
  │  Multi-Agent System (LangGraph)                  │
  │                                                  │
  │   Supervisor ──► routes to specialist            │
  │      │                                           │
  │      ├── Trial Search Agent                      │
  │      ├── Comparative Analysis Agent              │
  │      ├── Safety & AE Agent                       │
  │      └── Synthesizer Agent (final polish)        │
  └────────────────────┬─────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
        FastAPI                MCP Server
     (REST /query)         (Claude Desktop,
                            any MCP client)
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/pharma-intelligence-ai.git
cd pharma-intelligence-ai
python -m venv .venv && source .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY (or set LLM_PROVIDER=echo for offline mode)

# 3. Ingest clinical trials data
python ingest.py --reset

# 4. Start the API
uvicorn src.api.app:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

---

## API Endpoints

### `POST /query` — Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What adverse events were reported in pembrolizumab trials?",
    "phase": "PHASE3"
  }'
```

**Response:**
```json
{
  "answer": "In trial NCT05123456 (Pembrolizumab + Chemotherapy), serious adverse events included...",
  "agent_route": "safety",
  "route_reason": "Query contains adverse event keywords",
  "citations": ["NCT05123456"],
  "chunks_retrieved": 5,
  "top_chunks": [...]
}
```

### `GET /health` — Liveness probe

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}
```

### `GET /ready` — Readiness probe

```bash
curl http://localhost:8000/ready
# {"ready": true, "chunks_indexed": 47, "message": "Ready"}
```

### `POST /admin/reingest` — Trigger re-ingestion

```bash
curl -X POST http://localhost:8000/admin/reingest \
  -H "Content-Type: application/json" \
  -d '{"condition": "lung cancer", "phase": "PHASE3", "reset": true}'
```

---

## MCP Server (Claude Desktop)

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac):

```json
{
  "mcpServers": {
    "pharma-intelligence": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/pharma-intelligence-ai",
      "env": {
        "PYTHONPATH": "/path/to/pharma-intelligence-ai"
      }
    }
  }
}
```

Restart Claude Desktop. You'll see 5 tools available:
- `search_trials` — natural language search
- `get_trial_details` — full details by NCT ID
- `compare_trials` — side-by-side comparison
- `get_safety_profile` — adverse event analysis
- `find_eligible_trials` — patient-criteria matching

---

## Configuration

All configuration via environment variables (copy `.env.example` → `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `echo` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Claude model to use |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `CT_API_BASE_URL` | `https://clinicaltrials.gov/api/v2` | ClinicalTrials.gov API |
| `CT_RATE_LIMIT_PER_SEC` | `1.0` | API politeness limit |
| `CT_PAGE_SIZE` | `100` | Results per page (max 1000) |
| `VECTOR_STORE` | `chroma` | `chroma` \| `pinecone` |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB local storage path |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `API_PORT` | `8000` | FastAPI server port |
| `API_RATE_LIMIT` | `60/minute` | Per-IP rate limit |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_FORMAT` | `console` | `console` (dev) \| `json` (prod) |

**Offline / no API credits?** Set `LLM_PROVIDER=echo` to use the deterministic EchoProvider. All agent wiring works — responses are canned but the full retrieval pipeline runs.

---

## Running Evaluations

```bash
python evaluate.py
# or with custom output
python evaluate.py --output data/eval/my_report.json
```

Output includes: agent routing accuracy, faithfulness, answer relevancy, retrieval precision, and p50/p95/p99 latency across 10 golden test cases.

---

## Project Structure

```
pharma-intelligence-ai/
├── src/
│   ├── core/              # Config, LLM providers, vector store, embeddings, retriever, indexer
│   ├── ingestion/         # CT.gov HTTP client, fetcher, parser, chunker, models
│   ├── agents/            # LangGraph graph, supervisor, search, comparative, safety, synthesizer
│   ├── mcp_server/        # MCP protocol server + tool implementations
│   ├── api/               # FastAPI app, schemas, dependencies
│   └── evaluation/        # Metrics, evaluator
├── tests/                 # Unit + integration tests (mock-based, no network)
├── data/
│   ├── raw/               # Raw API responses (cached)
│   ├── processed/         # Processed documents
│   └── eval/              # Golden test set + eval reports
├── scripts/               # One-off utilities (mock data generator, golden set generator)
├── notebooks/             # Exploration scripts
├── ingest.py              # Data ingestion CLI
├── evaluate.py            # Evaluation runner
├── Dockerfile             # Multi-stage build (builder + runtime)
├── docker-compose.yml     # Local stack
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack — and Why

| Tool | Role | Why this one |
|---|---|---|
| **Python 3.12** | Runtime | Modern type hints, async maturity |
| **pydantic v2 + pydantic-settings** | Validation + config | Reads `.env`, validates types, fails fast on missing values |
| **httpx + tenacity** | HTTP + retries | Async-capable client; declarative retry with exponential backoff + jitter |
| **LangGraph** | Multi-agent orchestration | State machine model fits supervisor pattern; built-in checkpointing |
| **LangChain Core** | LLM abstractions | Provider-agnostic (Anthropic + OpenAI + Echo from one interface) |
| **ChromaDB** | Vector store (local) | Zero-infrastructure dev; swap to Pinecone via config change |
| **sentence-transformers** | Local embeddings | `all-MiniLM-L6-v2` = 384-dim, fast, no API cost, strong on biomedical text |
| **FastAPI + slowapi** | REST API + rate limiting | Auto OpenAPI docs, native async, per-IP throttling in 2 lines |
| **structlog** | Structured logging | JSON logs in prod, console logs in dev — same code |
| **MCP SDK** | Claude Desktop integration | Native tool access from any MCP-compatible client |
| **pytest + respx** | Tests + HTTP mocks | Full test suite, no network required, no API credits burned |

---

## Key Design Decisions

**Anti-corruption layer.** Raw ClinicalTrials.gov JSON gets parsed into `ParsedTrial` domain objects at ingestion time. If CT.gov v2 → v3, we change one parser file, not 30 call sites.

**Domain-aware chunking.** Generic chunkers (split every 500 chars) destroy clinical trial structure. We split by section (Eligibility, Endpoints, Interventions, AEs) so each chunk has semantic coherence and section-type metadata for filtering.

**Hybrid search.** Semantic similarity alone isn't enough — "Show me Phase III Merck trials" needs hard metadata filters. Every chunk carries NCT ID, phase, sponsor, status, section type for exact-match filtering on top of embedding similarity.

**Supervisor pattern.** One router, multiple specialists, nothing calls anything else directly. Every routing decision is logged with a reason. Easy to audit, easy to extend.

**Mock-mode toggle.** Corporate networks often block Python TLS fingerprints. The HTTP client has a `use_mock=True` flag that reads from local JSON — same code path as real API, zero network dependency.

**EchoProvider.** A deterministic fake LLM for offline dev and CI. All agent wiring, retrieval, and formatting logic runs without spending API credits. Swap to real provider with one `.env` change.

**ABC for everything swappable.** `VectorStore`, `EmbeddingProvider`, and `LLMProvider` are all ABCs. Switching ChromaDB → Pinecone, local embeddings → OpenAI, or Claude → GPT-4 is one class + one config change each.

---

## Roadmap

- [ ] Pinecone adapter for production-scale vector store
- [ ] Async FastAPI endpoints (currently sync graph invocation)
- [ ] Redis-backed rate limiting (survives restarts, works across instances)
- [ ] Biomedical embedding model (PubMedBERT for improved clinical text retrieval)
- [ ] Real CT.gov API ingestion (currently using mock data on restricted networks)
- [ ] Streaming responses via Server-Sent Events
- [ ] Authentication layer (API key middleware)
- [ ] Prometheus metrics endpoint for observability

---

## What I Learned Building This

This project taught me that production AI systems are 20% LLM calls and 80% boring engineering done right. The interesting problems weren't the agents — they were:

- **pydantic-settings v2 footguns**: nested `BaseSettings` tries to JSON-parse env vars. Flat class + dataclass accessors solved it.
- **Corporate TLS fingerprinting**: Python's httpx gets blocked by enterprise firewalls that let browsers through. Mock mode as a first-class feature is the right answer.
- **Chunking strategy matters more than model choice**: Domain-aware section chunking improved retrieval quality more than any embedding model change.
- **Test stubs have expiry dates**: Assertions on stub markers break the moment you replace the stub. Assert on behavior, not implementation artifacts.
- **Rate limiters are stateful**: In-process rate limiters bleed state across tests. Always reset between test cases — and use Redis in production.

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built as a learning project to explore production-grade RAG + multi-agent systems on real clinical trials data.*
