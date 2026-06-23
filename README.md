# AssistGen

An intelligent customer-service agent built with LangGraph (FastAPI backend, Vue 3 frontend). Each query is classified and routed to a dedicated handler: plain chat, clarification, knowledge-graph retrieval, image understanding, or document Q&A.

## Architecture

```
User input
    ↓
analyze_and_route_query   ← classifies intent via structured (Pydantic) output
    ↓
┌──────────────────────────────────────────────────────────────┐
│ general-query    → respond_to_general_query   plain LLM chat   │
│ additional-query → get_additional_info        ask for details  │
│ graphrag-query   → create_research_plan       GraphRAG / Neo4j │
│ image-query      → create_image_query         vision model     │
│ file-query       → create_file_query          document RAG     │
└──────────────────────────────────────────────────────────────┘
```

- **Intent router** — the LLM returns a typed `Router` and the graph dispatches to the matching branch.
- **Confidence-aware routing** — the router reports its confidence; a low-confidence knowledge-base query falls back to asking the user for details instead of guessing.
- **Guardrails** — the clarification and GraphRAG branches scope-check the question and politely decline out-of-business requests.
- **Document RAG (`file-query`)** — an uploaded PDF is chunked, embedded with sentence-transformers and indexed in FAISS; answers are grounded in the retrieved passages.
- **Streaming** — responses stream back over SSE for real-time rendering.

## Tech stack

| Layer | Technology |
|------|------|
| Agent framework | LangGraph |
| LLM | DeepSeek / Ollama (switchable) |
| Backend | FastAPI + SQLAlchemy |
| Knowledge graph | Neo4j + GraphRAG |
| Retrieval | sentence-transformers + FAISS |
| Cache | Redis (semantic cache) |
| Database | MySQL |
| Auth | JWT |
| Frontend | Vue 3 + TypeScript |

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```env
CHAT_SERVICE=deepseek            # or ollama
DEEPSEEK_API_KEY=your-api-key
DB_HOST=localhost
DB_PASSWORD=your-password
REDIS_HOST=localhost
```

### 3. Initialize the database

```bash
python scripts/init_db.py
```

### 4. Run

```bash
python run.py
```

Then open:
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Unit and node tests fake the LLM and need no external services; they run in CI on every push. An on-demand router-classification eval lives under `evals/`.

## Project structure

```
├── main.py                  # FastAPI entry point
├── app/
│   ├── api/                 # routes: auth, agent endpoint
│   ├── core/                # config, database, JWT, logging
│   ├── lg_agent/            # the LangGraph agent
│   │   ├── builder.py       # graph + node definitions
│   │   ├── state.py         # state types
│   │   └── kg_sub_graph/    # Neo4j / GraphRAG sub-graph
│   ├── services/            # LLM factory, embeddings, Redis cache, search
│   └── models/              # SQLAlchemy models
├── tests/                   # unit + node tests
├── evals/                   # on-demand evaluation scripts
└── scripts/init_db.py       # database initialization
```

## Roadmap

- Output validation / hallucination gate for knowledge-base answers
- Persistent checkpointer (SQLite/Postgres) so conversations survive restarts
- Unified cross-cutting middleware (guardrails, caching, tracing) across all branches

## License

MIT
