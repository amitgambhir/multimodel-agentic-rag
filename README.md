# Multi-LLM Multimodal Agentic RAG

A grounded RAG demo with a **provider switcher** (Claude or Gemini), persistent
**Chroma** vector store, **OpenAI `text-embedding-3`** for text and a
**vision-caption-then-embed** path for images, a **streaming** answer pipeline
with prompt caching, and a 3D embedding-space visualizer.

The agent runs up to **N hops** of retrieve → reason → optional refine → answer.
Citations are derived from the same retrieval that the model sees, so the panel
on the right is always faithful to the evidence.

## Features

- **Multi-LLM**: Claude (default) and Gemini, switchable from the header
- **Multimodal ingest**: text, URL, PDF, image (audio/video deferred)
- **Persistence**: Chroma store on disk so the index survives restarts
- **Streaming answers** via SSE; prompt caching for Claude
- **Configurable agent**: `top_k` and `max_hops` (1–3) from the header
- **Frontend**: React 19 + Vite + Tailwind v4 + Three.js 3D PCA view with
  hover tooltips, modality colors, query trail, drag-drop ingest, dark UI,
  per-answer cost/latency badges

## Layout

```
multimodel-agentic-rag/
├── backend/
│   ├── server.py              FastAPI app
│   ├── agent/loop.py          Agent loop (1–2+ hops)
│   ├── llm/{claude,gemini}.py Provider-neutral streaming
│   ├── rag_store.py           Chroma-backed multimodal store
│   ├── embeddings.py          OpenAI text embeddings + Claude/Gemini image captioning (local-hash fallback)
│   ├── ingest.py              URL/PDF/image helpers
│   └── config.py              Env-driven config
├── frontend/
│   ├── src/App.tsx            Layout + provider switcher
│   └── src/components/        SourceManager, AskPanel, EmbeddingView, TraceDrawer
└── data/chroma/               Persistent vector store (gitignored)
```

## Setup

### 1. Configure keys

Copy `.env.example` to `.env` and set at least one LLM key:

```bash
cp .env.example .env
# edit ANTHROPIC_API_KEY and/or GOOGLE_API_KEY and OPENAI_API_KEY
```

Without `OPENAI_API_KEY`, embeddings fall back to a deterministic local hash so
the UI still works for testing the plumbing. Image ingestion captions the
image with Claude (or Gemini) vision and embeds the caption — so you need at
least one LLM key for image sources to be meaningful.

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py        # http://localhost:8897
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev             # http://localhost:5173
```

Vite is configured with a `/api` proxy to the backend, so no CORS dance needed
in dev.

## Try it

1. Open http://localhost:5173.
2. Add some sources (paste text, fetch a URL, drop a PDF or image).
3. Ask a question. Watch the answer stream and citations appear.
4. Toggle the provider in the header to compare Claude vs Gemini on the same
   evidence.
5. Inspect the 3D view — cited points glow and a line connects them to the
   query point.

## Configuration

| Var | Default | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Enables Claude |
| `GOOGLE_API_KEY` | — | Enables Gemini |
| `OPENAI_API_KEY` | — | Enables real text embeddings |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Override per `.env` |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Or `text-embedding-3-large` (3072 dims) |
| `MAX_AGENT_HOPS` | `2` | UI lets you go 1–3 |
| `CHROMA_DIR` | `./data/chroma` | Persistence path |
| `PORT` | `8897` | Backend port |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated |
| `ALLOW_PRIVATE_URLS` | `false` | Allow localhost/private IP ingestion |

## API

Backend base: `http://localhost:8897` (proxied as `/api/*` from the frontend).

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Status, configured providers (`claude`, `gemini`), embedding provider (`openai` or `local-hash`), embedding dim, source count |
| `GET` | `/space` | Snapshot: sources, 3D PCA points, recent events, dim, embedding provider |
| `POST` | `/sources/text` | Add a text source. Body: `{ "title": str, "text": str }` |
| `POST` | `/sources/url` | Fetch and index a public URL. Body: `{ "url": str, "title"?: str }`. Localhost / private IPs blocked unless `ALLOW_PRIVATE_URLS=true` |
| `POST` | `/sources/file` | Multipart upload of a **PDF** or **image** (≤ 50 MB). Form fields: `file`, `title?`, `notes?`. Images are captioned via Claude/Gemini vision and the caption is embedded |
| `DELETE` | `/sources/{id}` | Remove a source and all its chunks |
| `POST` | `/ask` | **SSE stream**. Body: `{ "question": str, "provider": "claude" \| "gemini", "top_k": 1–12, "max_hops": 1–3 }` |

### `/ask` event stream

Server-Sent Events (`text/event-stream`); each line is `data: <json>`. Event
types in order:

| `type` | Fields | Meaning |
| --- | --- | --- |
| `trace` | `step`, `detail` | Agent step (e.g. `retrieve`, `think:1`) |
| `answer_delta` | `delta` | Token chunk of the streamed answer |
| `tool` | `name`, `input`, `result_summary` | A tool call the agent ran (currently only `refine_search` on hop ≥ 2) |
| `usage` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens` | Provider-reported token counts (cache fields are Claude-only) |
| `done` | `answer`, `matches`, `trace`, `usage`, `space` | Terminal event with the final answer, citations, full trace, and a fresh `space` snapshot with the query projected into the same PCA basis |
| `error` | `message` | Stream-time failure |

### Quick examples

```bash
# Add a text source
curl -X POST http://localhost:8897/sources/text \
  -H 'Content-Type: application/json' \
  -d '{"title":"PCA primer","text":"PCA finds orthogonal directions of maximum variance…"}'

# Fetch a URL
curl -X POST http://localhost:8897/sources/url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/article"}'

# Upload a PDF or image
curl -X POST http://localhost:8897/sources/file \
  -F 'file=@paper.pdf' -F 'title=Paper'

# Ask (streaming)
curl -N -X POST http://localhost:8897/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does PCA optimize?","provider":"claude","top_k":4,"max_hops":2}'
```

## Notes

- Audio/video ingestion is intentionally out of MVP scope. Add later via
  Whisper/AssemblyAI to transcribe, then route through the text path.
- Prompt caching is enabled on the Claude system prompt; repeated questions
  over the same source set should show non-zero `cache_read_tokens` after the
  first call (visible in the Ask panel's badge and the trace drawer).
- For production: add auth, background ingestion, evals, and consider a
  managed vector DB. The store is persistent locally but single-process.
