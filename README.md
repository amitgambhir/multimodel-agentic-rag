# Multi-LLM Multimodal Agentic RAG

A grounded, citation-faithful **Retrieval-Augmented Generation (RAG)** demo with a runtime **provider switcher** between Claude and Gemini, persistent **Chroma** vector store, **OpenAI `text-embedding-3`** for text and a **vision-caption-then-embed** path for images, a **streaming** SSE answer pipeline with prompt caching, and a **3D PCA embedding-space visualizer**.

Up to **N hops** of `retrieve → reason → optional refine_search → answer`. The citation panel is always derived from the same retrieval the model saw — never a separate search.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Component Breakdown](#component-breakdown)
   - [Frontend](#frontend-react--vite--threejs)
   - [Backend API](#backend-api-fastapi)
   - [Ingestion Pipeline](#ingestion-pipeline)
   - [Embeddings (text + image)](#embeddings-text--image)
   - [Vector Store](#vector-store-chroma)
   - [Agent Loop](#agent-loop)
   - [LLM Providers](#llm-providers-claude--gemini)
   - [3D Embedding View](#3d-embedding-view)
4. [Key Design Decisions](#key-design-decisions)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Quick Start](#quick-start)
8. [Extending the System](#extending-the-system)
9. [Project Structure](#project-structure)
10. [License](#license)

---

## Features

| Capability | Detail |
| --- | --- |
| **Multi-LLM** | Runtime switch between Claude and Gemini for answer generation; greys out unconfigured providers |
| **Multimodal ingest** | Text paste, URL fetch, PDF upload, image upload (drag-and-drop) |
| **Streaming answers** | SSE event stream with token-by-token rendering, abort/stop support |
| **Prompt caching** | Claude system prompt marked as ephemeral cache breakpoint; cache reads surfaced in the UI |
| **Faithful citations** | Citation panel reflects the same chunks the model received — including any `refine_search` hop |
| **Configurable agent** | 1–3 hops, `top_k` 1–12, all from the header |
| **3D embedding view** | Hand-rolled PCA + Three.js scene with hover tooltips, modality colors, and lines from query to cited points |
| **Persistent index** | Chroma store on disk; survives restarts |
| **SSRF-hardened URL ingest** | Manual redirect handling re-validates every hop against private/loopback/link-local IPs |

---

## Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                  Browser (React 19 + Vite + Tailwind v4)             │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────────┐  ┌────────────────┐  │
│  │  Source Manager  │  │      Ask Panel       │  │   3D View      │  │
│  │  - Text / URL    │  │  - Streaming answer  │  │  - Three.js    │  │
│  │    / PDF / Image │  │  - Citations panel   │  │  - PCA points  │  │
│  │  - Drag & drop   │  │  - Provider switch   │  │  - Query trail │  │
│  │  - Modality chips│  │  - top_k / hops      │  │  - Hover info  │  │
│  └────────┬─────────┘  └──────────┬───────────┘  └────────┬───────┘  │
└───────────┼───────────────────────┼───────────────────────┼──────────┘
            │ /api proxy            │ SSE                   │
            ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                               │
│                                                                      │
│  GET  /health     /space                                             │
│  POST /sources/{text,url,file}    DELETE /sources/{id}               │
│  POST /ask  ── Server-Sent Events ──▶                                │
│                                                                      │
│  ┌────────────────────┐    ┌────────────────────────────────────┐    │
│  │ Ingestion Pipeline │    │           Agent Loop               │    │
│  │                    │    │                                    │    │
│  │  Text  ─┐          │    │  retrieve(top_k)                   │    │
│  │  URL   ─┼─▶ chunk  │    │   │                                │    │
│  │  PDF   ─┘   text   │    │   ▼                                │    │
│  │           │        │    │  format evidence in user msg       │    │
│  │           ▼        │    │   │                                │    │
│  │   embed_documents  │    │   ▼                                │    │
│  │           │        │    │  provider.stream(messages, tools)  │    │
│  │  Image  ──┘        │    │   ├─ text  → SSE answer_delta      │    │
│  │   │                │    │   └─ tool  → refine_search ── hop2 │    │
│  │   ▼                │    │              │                     │    │
│  │  caption (Claude/  │    │              ▼                     │    │
│  │   Gemini vision)   │    │       merge matches → final answer │    │
│  │   │                │    │                                    │    │
│  │   ▼                │    └────────────────────────────────────┘    │
│  │  embed caption     │                                              │
│  └────────────────────┘                                              │
└────────────┬───────────────────────────────────────┬─────────────────┘
             │                                       │
   ┌─────────▼─────────┐                  ┌──────────▼──────────┐
   │    Chroma         │                  │  Provider SDKs      │
   │  (persistent,     │                  │   Anthropic         │
   │   project root)   │                  │   Google GenAI      │
   │                   │                  │   OpenAI (embeds)   │
   │  Collection:      │                  └─────────────────────┘
   │  chunks (cosine)  │
   └───────────────────┘
```

---

## Component Breakdown

### Frontend (React + Vite + Three.js)

**Stack:** React 19, Vite 6, Tailwind CSS v4 (via `@tailwindcss/vite` plugin), Three.js, Lucide icons.

A three-pane layout:

- **Left — Source Manager.** Tabbed text / URL / file input with a drag-and-drop zone, modality filter chips, and per-source cards that highlight when cited. Image cards display the thumbnail; PDF/URL/text cards show a snippet preview.
- **Center — Ask Panel.** Question input with `⌘/Ctrl+Enter` shortcut, token-by-token streamed answer with a blinking cursor, hover-highlight citation cards, and a footer badge with provider, top_k, hops, latency, and (Claude only) cache-hit tokens.
- **Right — 3D Embedding View + Trace Drawer.** Hand-rolled Three.js scene (no React-Three-Fiber). Each source is one point colored by modality; the query is rendered as a white dot with a ring; lines connect cited points to the query so the spatial story is legible. The trace drawer shows the agent's step timeline with a usage footer.

The Vite dev server proxies `/api/*` to the backend, so the frontend never needs to know the backend URL in development.

### Backend API (FastAPI)

**Stack:** FastAPI, uvicorn, Pydantic v2, httpx, BeautifulSoup, pypdf, Pillow.

Two surface areas: source management and the agentic `/ask` stream. All blocking work (`RAG_STORE` calls, PDF/image extraction) is wrapped in `run_in_threadpool` so the event loop stays free for concurrent SSE streams. The store uses an `RLock` and is not async-safe on its own.

`/ask` returns a `text/event-stream` SSE response. Each line is `data: <json>` carrying one of: `trace`, `answer_delta`, `reset_answer`, `tool`, `usage`, `done`, `error`. The frontend parses this directly in `lib/api.ts`.

### Ingestion Pipeline

```text
Add source
   │
   ├─ Text  ─────────────────────────┐
   ├─ URL   ─▶ httpx (manual         │
   │            redirect handling,   │
   │            every hop validated  │
   │            against private IPs) │
   │            │                    │
   │            ▼                    │
   │          BeautifulSoup strip ───┤
   │                                 │
   ├─ PDF   ─▶ pypdf extract text ───┤
   │                                 │
   └─ Image ─▶ Pillow normalize      │
              │                      │
              ▼                      │
            Claude Haiku (or Gemini) │
            single-sentence caption  │
              │                      │
              └────── caption ───────┤
                                     ▼
                          chunk text (700 / 80 overlap)
                                     │
                                     ▼
                          OpenAI text-embedding-3
                                     │
                                     ▼
                          Chroma (persistent, cosine HNSW)
```

URL ingest disables `httpx` auto-redirects and walks each hop manually so the validator runs on every URL, not just the initial one — closing the SSRF gap where a public URL 302s to `127.0.0.1` or cloud-metadata IPs.

### Embeddings (text + image)

`backend/embeddings.py` is the only embedding adapter.

- **Text / URL / PDF:** OpenAI `text-embedding-3-small` by default (1536 dims), swappable to `-large` (3072 dims) via `OPENAI_EMBED_MODEL`. A first successful call locks the dim for the lifetime of the process so a transient OpenAI failure can't corrupt the index with mismatched vectors.
- **Image:** Captioned by Claude Haiku (one short sentence focused on subject, setting, and notable details). Falls back to Gemini if no Anthropic key. The caption is then embedded as text — so retrieval quality on images depends on caption quality, but no separate multimodal embedding model is required.
- **No-key fallback:** A deterministic SHA256-derived local hash so the UI plumbing works without keys. The dim is locked from the existing collection on boot, so the fallback always produces same-dim vectors.

### Vector Store (Chroma)

`backend/rag_store.py` wraps Chroma's `PersistentClient`. The collection uses `hnsw:space=cosine`. Sources are virtual — they exist as metadata on chunks (`source_id`, `title`, `modality`, `chunk_index`, `preview`) — and are reconstructed in memory by `_restore_sources()` at boot.

**Important:** Chroma is sensitive to dim mismatches on a single collection. Switching `OPENAI_EMBED_MODEL` between `-small` (1536) and `-large` (3072) requires wiping `data/chroma/` first; the embedding adapter will refuse to write a wrong-dim vector once the dim is locked.

### Agent Loop

`backend/agent/loop.py` `run_agent` is the heart of the system. The flow:

```text
1. retrieve(top_k)               # one search, before any LLM call
   └─ matches  →  formatted as numbered evidence in user message
2. provider.stream(msgs, tools=[refine_search])
   ├─ text deltas → SSE answer_delta
   ├─ usage       → SSE usage
   └─ tool_call(refine_search) → emit reset_answer (drop pre-tool tokens)
3. if tool was called and hops < max:
       run refine_search via RAG_STORE.search   (in threadpool)
       merge new matches into the citation set, dedupe by (source_id, chunk_index)
       append tool_result block, loop back to (2)
4. otherwise: emit `done` with the final answer + matches + trace + usage
   plus a fresh /space snapshot with the query projected into the same PCA basis
```

The single-source-of-truth contract: **the citation panel is built from the same `matches` array the model sees as evidence.** The agent never silently reruns retrieval; the only second retrieval is the explicit `refine_search` tool, and its results are surfaced to the UI as a `tool` event.

### LLM Providers (Claude + Gemini)

`backend/llm/base.py` defines a small streaming protocol — every provider yields the same `ProviderEvent` shape:

```text
{type: "text",      delta: str}
{type: "tool_call", id, name, input: dict}
{type: "usage",     input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}
{type: "stop",      reason: str}
```

`llm/claude.py` translates Anthropic SDK events into this shape and marks the system prompt with `cache_control: ephemeral` so repeated questions over the same source set hit the cache (visible in the UI as `cache_read_tokens`).

`llm/gemini.py` does the equivalent for `google-genai`. Two notable translations live here: an `id → name` lookup so `tool_result` blocks (which carry only `tool_use_id` per Anthropic schema) can be matched back to function names, and a JSON-Schema-to-Gemini-Type uppercase normalizer so tool schemas defined in Anthropic-native format work for Gemini too.

### 3D Embedding View

`frontend/src/components/EmbeddingView.tsx`. PCA is computed server-side in pure Python (`_power_iteration` in `rag_store.py`) so the math has no numpy dependency. `snapshot(query_vector=...)` projects sources and the query into the **same basis** for that request — the query dot is comparable to the source points, not in a separate space.

The Three.js scene manages its own state in a ref; the React component only re-renders point geometry when the `snapshot` or `citedIds` set changes. Drag to orbit, scroll to zoom, hover for tooltips. Cited points glow brighter and draw a translucent line back to the query dot.

---

## Key Design Decisions

### 1. Single retrieval, faithful citations

`/ask` runs retrieval once before the model is called and passes those exact matches as evidence. The model can request a second search only via the explicit `refine_search` tool; results are merged into the citation set and surfaced as a `tool` event. This guarantees the right-hand citation panel is what the model actually saw.

### 2. Provider-neutral streaming

Both Claude and Gemini providers yield the same `ProviderEvent` shape, so `agent/loop.py` is provider-agnostic. Adding a third provider is a single new file in `llm/`; nothing else changes.

### 3. Caption-then-embed for images

OpenAI text embeddings are text-only. Rather than introducing a second embedding model just for images, the system captions images with Claude (or Gemini) vision and embeds the caption as text. The trade-off: image retrieval quality depends on caption quality, but the system stays on a single embedding model with consistent dims.

### 4. Dim locking

A first successful OpenAI embed locks the dim for the process. If OpenAI later fails, the local-hash fallback still produces same-dim vectors — Chroma never sees mismatched dims and the index can't be silently corrupted. A model swap (`-small` → `-large`) returning a different dim raises a clear error instead of writing.

### 5. Prompt caching on the system block

Claude's system prompt is marked as a cache breakpoint. After the first query in a session, repeated questions show non-zero `cache_read_tokens` in the UI badge and trace drawer, cutting cost on the system-prompt portion of the context.

### 6. Pre-tool text bleed handled explicitly

If a model emits text before calling `refine_search` (e.g. "Let me search for that…"), that text would otherwise contaminate the final answer. The agent emits a `reset_answer` SSE event so the frontend clears the streamed buffer; the post-tool turn produces the actual answer.

### 7. SSRF defence on every redirect hop

URL ingestion disables `httpx` auto-redirect and walks each hop manually, re-running the private-IP validator at each step. The pre-existing single-shot validator was bypassable by a 302 to `127.0.0.1`.

### 8. Project-root anchored persistence

Relative `CHROMA_DIR` values are resolved against the project root, not the CWD. Running `python server.py` from `backend/` puts the store at `<repo>/data/chroma`, never `backend/data/chroma`.

### 9. Hand-rolled PCA over numpy

The PCA projection is power-iteration in pure Python (~50 lines). It's slower than numpy at scale but keeps the dependency footprint small and lets the projection be embedded in the JSON snapshot directly.

---

## API Reference

Backend base: `http://localhost:8897` (proxied as `/api/*` from the frontend).

### `GET /health`
Server status, configured providers, embedding provider, embedding dim, source count.

```json
{
  "status": "ok",
  "providers": ["claude", "gemini"],
  "max_hops_default": 2,
  "sources_total": 3,
  "embedding_provider": "openai",
  "dimensions": 1536
}
```

### `GET /space`
Snapshot of sources, 3D PCA points, recent events, dim, embedding provider. Used by the frontend on mount to populate the source list and the 3D view.

### `POST /sources/text`
Add a text source.

**Request:** `{ "title": str, "text": str }`
**Response:** `{ "source": {...}, "space": {...} }`

### `POST /sources/url`
Fetch and index a public URL. Localhost / private / link-local IPs blocked unless `ALLOW_PRIVATE_URLS=true`.

**Request:** `{ "url": str, "title"?: str }`

### `POST /sources/file`
Multipart upload. Supported: **PDF** and **image** (≤ 50 MB). Images are captioned via Claude/Gemini vision and the caption is embedded.

**Form fields:** `file` (required), `title?`, `notes?`

### `DELETE /sources/{id}`
Remove a source and all its chunks.

### `POST /ask`
**SSE stream.** Each line is `data: <json>`. Body:

```json
{
  "question": "What does PCA optimize?",
  "provider": "claude",
  "top_k": 6,
  "max_hops": 2
}
```

Event types in order:

| `type` | Fields | Meaning |
| --- | --- | --- |
| `trace` | `step`, `detail` | Agent step (`retrieve`, `think:1`, `refine_search`, `answer`) |
| `answer_delta` | `delta` | Token chunk of the streamed answer |
| `reset_answer` | — | Discard streamed tokens so far (model emitted text before calling a tool) |
| `tool` | `name`, `input`, `result_summary` | Tool call the agent ran (currently only `refine_search`) |
| `usage` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens` | Provider-reported token counts (cache fields are Claude-only) |
| `done` | `answer`, `matches`, `trace`, `usage`, `space` | Terminal event with the final answer, citations, full trace, and a fresh `space` snapshot |
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

---

## Configuration

All settings live in `.env` at the project root (copy from `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Enables Claude as an answer provider and image captioning |
| `GOOGLE_API_KEY` | — | Enables Gemini as an answer provider and image captioning fallback |
| `OPENAI_API_KEY` | — | Enables real text embeddings (without it, falls back to local hash) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model used for answers |
| `CLAUDE_FAST_MODEL` | `claude-haiku-4-5-20251001` | Used for image captioning |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini model used for answers and image-caption fallback |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Or `text-embedding-3-large` (3072 dims) |
| `MAX_AGENT_HOPS` | `2` | Default agent hop limit; UI lets you go 1–3 |
| `PORT` | `8897` | Backend port |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated CORS allowlist |
| `ALLOW_PRIVATE_URLS` | `false` | Allow localhost / private / link-local IP ingestion (dev only) |
| `CHROMA_DIR` | `./data/chroma` | Persistence path (relative paths anchor to project root) |

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.9 works but is past EOL for some sub-deps)
- Node 18+
- An Anthropic API key (Claude) and/or a Google AI Studio key (Gemini) — at least one
- An OpenAI API key for real embeddings (optional, but strongly recommended)

### Steps

```bash
# 1. Configure keys
cp .env.example .env
# edit .env and fill in ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY

# 2. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py                  # → http://localhost:8897

# 3. Frontend (in a second terminal)
cd frontend
npm install
npm run dev                       # → http://localhost:5173
```

Open <http://localhost:5173> in your browser.

| Service | URL |
| --- | --- |
| Frontend (React) | <http://localhost:5173> |
| Backend API | <http://localhost:8897> |
| OpenAPI docs (Swagger) | <http://localhost:8897/docs> |

If the backend is on a non-default port, point the frontend at it with `VITE_API_URL=http://localhost:<port> npm run dev`.

### Verifying the install

The header in the UI shows `online · openai · 1536 dims · agent · 2 hops` when everything is wired up. Provider buttons grey out for any LLM without a configured key.

---

## Extending the System

### Add a third LLM provider

In `backend/llm/`, add a new file (e.g. `openai.py`) implementing the `LLMProvider` protocol from `base.py`. The streaming method must yield the same `ProviderEvent` shape as the others. Then register it in `llm/__init__.py`:

```python
elif name == "openai":
    return OpenAIProvider()
```

The frontend `ProviderPicker` already filters by what `/health` returns, so adding the provider name to `available_providers()` in `config.py` is enough to make it show up.

### Add a new ingestion modality

Add a loader in `backend/ingest.py` (e.g. `extract_docx_text`) and dispatch on it from the `/sources/file` handler in `server.py`. If the modality is non-text (e.g. audio), follow the image pattern: convert to a textual representation (transcription / caption) and route through `embed_documents`.

### Add a new agent tool

Define the tool in `backend/agent/loop.py` alongside `_refine_tool_def()`, append it to the `tools` list, and add a branch to the `for tc in pending_tool_calls` block to handle the call. Tool input/output is JSON-serialisable; the loop already takes care of merging it back into the citation set or surfacing it as a `tool` SSE event.

### Swap the vector store

All Chroma interaction is in `backend/rag_store.py` behind a small interface (`add_text_source`, `add_image_source`, `search`, `snapshot`, `remove_source`). Replacing Chroma with Qdrant / Weaviate is a single-file change.

### Use larger embeddings

Set `OPENAI_EMBED_MODEL=text-embedding-3-large` in `.env`. **Wipe `data/chroma/` first** — the embedding dim differs (1536 vs 3072) and the system will refuse to mix them.

---

## Project Structure

```text
multimodal-agentic-rag/
│
├── .env.example                   # Template for all config vars
├── README.md
├── CLAUDE.md                      # Guidance for Claude Code in this repo
│
├── backend/
│   ├── server.py                  # FastAPI app, CORS, SSE /ask stream
│   ├── config.py                  # All env vars, project-root path anchoring
│   ├── app_state.py               # Singleton RAG_STORE
│   ├── embeddings.py              # OpenAI text + caption-then-embed image path
│   ├── ingest.py                  # URL (SSRF-hardened), PDF, image helpers
│   ├── rag_store.py               # Chroma wrapper, dim locking, PCA projection
│   ├── requirements.txt
│   │
│   ├── agent/
│   │   └── loop.py                # 1–3 hop streaming agent, refine_search tool
│   │
│   └── llm/
│       ├── base.py                # ProviderEvent shape + LLMProvider protocol
│       ├── claude.py              # Anthropic SDK adapter, prompt caching
│       └── gemini.py              # google-genai adapter, schema normalizer
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts             # /api proxy to backend
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                # Three-pane layout, provider switcher
│       ├── styles.css             # Tailwind v4 + design tokens (@theme)
│       ├── lib/
│       │   ├── api.ts             # fetch wrappers, SSE reader
│       │   └── types.ts           # Shared types (Source, Match, StreamEvent, …)
│       └── components/
│           ├── SourceManager.tsx  # Tabs, drag-drop, modality filter, source cards
│           ├── AskPanel.tsx       # Streaming answer + citations + cost badge
│           ├── EmbeddingView.tsx  # Three.js PCA scene
│           └── TraceDrawer.tsx    # Agent step timeline
│
└── data/
    └── chroma/                    # Persistent vector store (gitignored)
```

---

## License

This project is licensed under the [MIT License](LICENSE).
