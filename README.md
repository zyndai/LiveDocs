# ragpipline — Haystack-based docs RAG

Multi-turn doc Q&A built on Haystack. This folder is self-contained — treat
it as if it were its own GitHub repo. All commands below assume you've opened
this folder directly in VS Code (so the integrated terminal's working
directory is the ragpipline root).

## What's here

| File | Role |
|---|---|
| `config.py` | All knobs: models, ports, paths, Qdrant URL |
| `chunker.py` | Heading-aware markdown chunker (two-pass) |
| `fetch_docs.py` | Clones / pulls the docs repo into `tmp/docs/` |
| `index.py` | **Build the index** — runs once when docs change |
| `pipeline.py` | **Query path** — embed → hybrid search → rerank → LLM |
| `app.py` | FastAPI HTTP wrapper around `pipeline.py` |
| `.env` | `GOOGLE_API_KEY`, `GITHUB_TOKEN` (gitignored, copy from `.env.example`) |
| `requirements.txt` | Python deps |

## Prerequisites

- **Python 3.12+**
- **Docker** (for Qdrant)
- A **Google AI Studio API key** — get one at https://aistudio.google.com/apikey
- A **GitHub PAT** if your docs repo is private (see below)

## First-time setup (do once)

```powershell
# 1. create + activate the venv (lives inside this folder)
python -m venv .venv
.\.venv\Scripts\activate

# 2. install deps (~5 minutes first time -- pulls torch, transformers, etc.)
pip install -r requirements.txt

# 3. set up secrets
copy .env.example .env
notepad .env        # paste GOOGLE_API_KEY (and GITHUB_TOKEN if the docs repo is private)
```

### Private docs repo

`DOCS_REPO_URL` in `config.py` points at the docs repo. If that repo is
private, you need `GITHUB_TOKEN` in `.env`.

## Running the stack (three terminals)

All three terminals start at this folder (the ragpipline root). No `cd ..`
anywhere.

### Terminal 1 — Qdrant (vector database)

```powershell
docker run -p 6333:6333 -p 6334:6334 `
    -v ${PWD}/tmp/qdrant_storage:/qdrant/storage `
    qdrant/qdrant
```

Wait for `Qdrant HTTP listening on 6333`. Leave it running.

Persistent data lives in `tmp/qdrant_storage/` inside this folder. Survives
container restarts. Already in `.gitignore` so it won't be committed.

Dashboard: http://localhost:6333/dashboard

### Terminal 2 — Build the index (run only when docs change)

```powershell
.\.venv\Scripts\activate
python index.py
```

What this does:
1. Clones the docs repo into `tmp/docs/` (first run) or pulls latest
2. Chunks all `.md` files (heading-aware, two-pass)
3. Embeds each chunk with **bge-m3** (dense, 1024-dim) **AND** SPLADE (sparse)
4. Writes everything to the `zynd_docs_haystack` Qdrant collection
5. Prints `Haystack reports N documents written.` and exits

First run downloads model weights (~3.5 GB total) into your HuggingFace cache
(`~/.cache/huggingface/`). Subsequent runs reuse them.

**Time estimate:** 30–45 min on CPU first time. 10–15 min thereafter (no
re-download). Under a minute on a GPU.

You only re-run this when docs change.

### Terminal 3 — Start the API server

```powershell
.\.venv\Scripts\activate
uvicorn app:app --port 8002
```

Wait for `Ready.` then `Application startup complete.` (model warm-up takes
about 30 seconds).

Backend is now live at **http://localhost:8002**.

## Verify it works

### Health check

```powershell
curl.exe http://localhost:8002/health
# -> {"status":"ok","impl":"haystack"}
```

### Single-shot question (no history)

Open http://localhost:8002/docs in your browser → click `/ask` →
"Try it out" → paste:

```json
{ "question": "how do I deploy an agent on zynd?" }
```

Or from PowerShell:

```powershell
$body = @{ question = "how do I deploy an agent on zynd?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8002/ask `
    -ContentType "application/json" -Body $body
```

Response:

```json
{
  "answer": "You can deploy in two ways...",
  "sources": [{"source": "deployer/deploy.md", ...}, ...],
  "retrieval_query": "how do I deploy an agent on zynd?"
}
```

### Multi-turn (with history)

Send the prior conversation as `history`:

```json
{
  "question": "what files should I include in the zip?",
  "history": [
    { "role": "user", "content": "how do I deploy an agent on zynd?" },
    { "role": "assistant", "content": "<the previous answer text>" }
  ]
}
```

Watch `retrieval_query` in the response — when history is non-empty, a
separate Gemini Flash call rewrites the question into a standalone form
before retrieval. So `"what files should I include in the zip?"` becomes
something like `"What files should I include when deploying an agent to
deployer.zynd.ai?"`, which retrieves much better chunks than the bare
follow-up would.

## Configuration knobs (`config.py`)

| Variable | Default | Effect |
|---|---|---|
| `DOCS_REPO_URL` | (set in config.py) | The docs repo to clone |
| `DENSE_EMBEDDING_MODEL` | `BAAI/bge-m3` | Must match between index + query |
| `SPARSE_EMBEDDING_MODEL` | `prithivida/Splade_PP_en_v1` | Same — must match |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder for final ordering |
| `LLM_MODEL` | `gemma-4-26b-a4b-it` | Switch to `"gemini-2.5-flash"` for faster/cheaper |
| `REWRITER_MODEL` | `gemini-2.5-flash` | Follow-up question rewriter |
| `RETRIEVE_TOP_K` | `10` | How many chunks the retriever returns |
| `RERANK_TOP_K` | `5` | How many chunks the LLM sees |
| `QDRANT_COLLECTION` | `zynd_docs_haystack` | Collection name in Qdrant |
| `API_PORT` | `8002` | Note: actually passed via uvicorn `--port` |

## Updating docs after they change

```powershell
.\.venv\Scripts\activate
python index.py       # re-fetch + re-chunk + re-embed + re-write Qdrant
```

The API server stays up — no restart needed. It reads Qdrant on every query,
so the new index takes effect immediately.

## Troubleshooting

### `ModuleNotFoundError: No module named '...'`
The venv isn't activated, or the package installed into a different Python.
Bulletproof install:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `GOOGLE_API_KEY not set`
Missing `.env`. Copy from `.env.example` and add your key.

### `Authentication failed` cloning docs
`GITHUB_TOKEN` is missing, expired, or doesn't have access to the docs repo.
Check token scope at https://github.com/settings/tokens.

### `Connection refused` to Qdrant
Qdrant container isn't running. Check Terminal 1.

### `500 INTERNAL` from the LLM
Gemma 4 endpoint occasionally flakes. Edit `config.py` →
`LLM_MODEL = "gemini-2.5-flash"`, restart uvicorn.

### Server hangs on first `/ask`
Likely still loading models. Wait for `Ready.` in the uvicorn log.

### Slow queries (5–10s on CPU)
That's normal on CPU. See "Performance" below.

## Performance

| Stage | Time on CPU | Time on GPU |
|---|---|---|
| Question embed (bge-m3 + SPLADE) | ~1 s | <0.1 s |
| Qdrant hybrid search | <100 ms | <100 ms |
| Rerank (bge-reranker-large, 10 candidates) | 2–4 s | 0.2 s |
| LLM call (Gemma 4, network) | 2–5 s | 2–5 s (network-bound) |
| **Total** | **~6–10 s** | **~3 s** |

The LLM call is network-bound to Google's servers — your hardware doesn't
change it. Only self-hosting the LLM (e.g. via Ollama) eliminates that ~3s
floor.

Biggest CPU-side win: `LLM_MODEL = "gemini-2.5-flash"` in `config.py`
(~2–3s saved per query, comparable answer quality for grounded Q&A).

## What runs where

```
   ┌─────────────────────┐
   │  Your frontend      │     VitePress site (separately hosted)
   └──────────┬──────────┘
              │ POST /ask  {question, history}
              ▼
   ┌─────────────────────┐
   │     app.py          │     FastAPI on localhost:8002
   │  (this folder)      │     Holds models in RAM, ~4 GB
   └──────┬──────────┬───┘
          │          │ HTTPS
          │          └─────────► Google AI Studio (Gemma 4 / Gemini Flash)
          ▼
   ┌─────────────────────┐
   │      Qdrant         │     Docker on localhost:6333
   │   N vectors         │     Persistent volume in tmp/qdrant_storage/
   └─────────────────────┘
```
