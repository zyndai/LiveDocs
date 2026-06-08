# Codebase RAG — vector + graph

Chat with your codebase. Drop repos into `code/`, run one build command, then
ask questions in natural language. Answers are grounded in your actual code,
cite `repo/file.ext:start-end`, and use a call graph to explain how pieces
connect. Heavy work happens once at build time; runtime queries are fast.

## How it works

```
BUILD (slow, once)                          RUNTIME (fast, per query)
──────────────────                          ─────────────────────────
walk code/  ──► tree-sitter chunk ──┐       question (+ history)
(py/js/ts/go)   (1 chunk/symbol)    │            │
                                    ├─► embed  decompose into sub-queries
                 extract edges ─────┤   bge-m3  (splits "and"/multi-intent)
                 (calls/imports)    │   +SPLADE      │
                      │             └─► Qdrant   hybrid search each ──► pool
                 build graph                         │
                      │                          rerank pool (cross-encoder)
                 serialize ──► tmp/code_graph.pkl    │
                                    ▲           graph expand: 1-hop callers/
                              loaded into RAM    callees (O(1) RAM lookup)
                              at server start         │
                                                 LLM explain + cite file:line
```

| File | Role |
|---|---|
| `config.py` | All knobs (paths, models, ports, code extensions, graph) |
| `code_walker.py` | Walk `code/`, filter by extension, skip junk dirs |
| `code_chunker.py` | tree-sitter → one chunk per function/class/method + call/import edges |
| `code_graph.py` | Build + serialize graph (build); O(1) adjacency lookups (runtime) |
| `index.py` | Embed (bge-m3 dense + SPLADE sparse) → Qdrant |
| `build.py` | **One command**: walk → chunk → graph → embed → index |
| `code_pipeline.py` | Query: decompose → retrieve → rerank → graph-expand → LLM |
| `app.py` | FastAPI HTTP wrapper (`POST /ask`) |

## Prerequisites

- **Python 3.12+**
- **Docker** (for Qdrant)
- A **Google AI Studio API key** — https://aistudio.google.com/apikey

## Setup (once)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt      # ~5 min first time (pulls torch, transformers)
cp .env.example .env                 # then add your GOOGLE_API_KEY
```

## Add your code

Drop any number of repos/folders into `code/`:

```
code/
  my-backend/
  my-frontend/
  some-library/
```

Indexed file types: `.py .js .jsx .ts .tsx .go`. Junk dirs
(`node_modules`, `.venv`, `dist`, `build`, `target`, `vendor`, `.git`, …) are
skipped automatically. Adjust `CODE_LANG_BY_EXT` / `CODE_IGNORE_DIRS` in
`config.py`.

## Run (three terminals)

### Terminal 1 — Qdrant (vector DB)
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v ${PWD}/tmp/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
Dashboard: http://localhost:6333/dashboard

### Terminal 2 — Build the knowledge base (re-run when code changes)
```bash
source .venv/bin/activate
python build.py
```
Walks `code/`, chunks with tree-sitter, builds the graph
(`tmp/code_graph.pkl`), embeds, and writes Qdrant. First run downloads model
weights (~3.5 GB) into `~/.cache/huggingface/`. Slow on CPU first time; fast
after.

### Terminal 3 — Start the API server
```bash
source .venv/bin/activate
uvicorn app:app --port 8002
```
Wait for `Ready.` (loads models + graph into RAM). API at http://localhost:8002.

## Use it

```bash
curl http://localhost:8002/health
# -> {"status":"ok","impl":"code-vector-graph"}
```

Interactive docs: http://localhost:8002/docs → `/ask`.

```json
{ "question": "how does auth work and how do I add a new route?" }
```

Response:
```json
{
  "answer": "... cites my-backend/auth.py:8-15 ...",
  "sources":     [{ "repo": "...", "file": "...", "symbol": "...", "start_line": 8, "end_line": 15, "score": 0.91 }],
  "related":     [{ "...": "call-graph neighbors pulled in as context" }],
  "sub_queries": ["how does auth work?", "how do I add a new route?"]
}
```

Multi-turn: pass prior turns as `history` (same format as the docs pipeline).
Follow-ups are rewritten to standalone form before retrieval.

## Key config knobs (`config.py`)

| Variable | Default | Effect |
|---|---|---|
| `CODE_DIR` | `./code` | Where your repos live |
| `CODE_LANG_BY_EXT` | py/js/ts/go | Extensions → tree-sitter grammar |
| `CODE_MAX_TOKENS` | `1000` | Max tokens per symbol chunk (oversized split) |
| `GRAPH_EXPAND_HOPS` | `1` | Call-graph hops pulled in as context |
| `GRAPH_MAX_NEIGHBORS` | `6` | Cap neighbors per hit |
| `MAX_SUBQUERIES` | `4` | Cap on compound-question fan-out |
| `RETRIEVE_TOP_K` | `10` | Chunks retrieved per sub-query |
| `RERANK_TOP_K` | `5` | Chunks the LLM sees |
| `LLM_MODEL` | `gemma-4-26b-a4b-it` | Switch to `gemini-2.5-flash` for faster/cheaper |
| `CODE_QDRANT_COLLECTION` | `codebase_rag` | Qdrant collection name |

## Updating after code changes

```bash
source .venv/bin/activate
python build.py        # re-walk + re-chunk + rebuild graph + re-embed
```
Restart the server so it reloads the new graph into RAM.

## Troubleshooting

- **`CODE_DIR not found`** — create `code/` and drop repos in it.
- **`No chunks produced`** — `code/` has no `.py/.js/.ts/.go` files (or all in ignored dirs).
- **`GOOGLE_API_KEY not set`** — copy `.env.example` → `.env`, add your key.
- **`Connection refused` to Qdrant** — Qdrant container isn't running (Terminal 1).
- **No graph expansion** — `tmp/code_graph.pkl` missing; run `build.py`.
- **`500 INTERNAL` from LLM** — set `LLM_MODEL = "gemini-2.5-flash"`, restart server.
