<div align="center">

# 🧙 LiveDocs

**The AI that reads your code so your docs don't have to be perfect.**

Drop a chat box on your docs site that answers from your _actual codebase_ plus
your structured docs — so users get the latest, deepest answer without waiting
for the next documentation update. Grounded in real source, cites
`repo/file.ext:line` with direct GitHub links, follows the call graph, and
streams token-by-token over SSE.

</div>

---

## The problem

Nobody reads the docs. And the docs are never finished anyway.

Writing and refining SDK + system documentation is slow, and there's _always_
something missing — an edge case, a new flag, a function that changed last
sprint. Users hit that gap and file a ticket, or give up. You spend the next
hackathon patching docs instead of building.

The asymmetry: **your docs lag, but your code is always current.** The answer
already exists — it's sitting in the source, just not written down yet.

## The idea

Put a chatbot on your docs page that knows **both**:

- **Your codebase** — the latest, deepest source of truth. Updated the moment
  you push. No waiting for someone to write it up.
- **Your structured docs** — the curated narrative: getting-started paths,
  concepts, the "intended" way to use things.

Together they cover each other's weaknesses:

- **Docs incomplete?** The AI fills the gap from the code itself.
- **Docs wrong or stale?** The code grounds the answer, so the AI self-corrects
  against what the system _actually_ does — and you can see the mismatch.
- **Complex / open-ended question?** Users brainstorm and reason with an
  assistant that has read the whole system, not just one page.

Users stop waiting for doc updates. You stop firefighting doc gaps.

## How LiveDocs delivers it

Most "chat with your code" tools embed files and hope cosine similarity finds
the right chunk. LiveDocs adds what makes the answers trustworthy:

- **Code + docs in one answer.** Two corpora retrieved together — blends "how
  it's documented" with "how it's actually built."
- **Call-graph expansion.** When a code chunk is retrieved, LiveDocs pulls in
  its 1-hop callers/callees from a precomputed graph, so the model sees the
  function _and_ everything around it. Built once, O(1) at query time.
- **Cited, verifiable answers.** Every claim links to `repo/file.ext:line` on
  GitHub — users (and you) can check the source, which is what makes
  self-correction against stale docs safe.

Plus the table stakes: hybrid dense + sparse retrieval, compound-question
decomposition, conversation history, and streaming answers.

## How it works

```
BUILD (slow, once)                          RUNTIME (fast, per query)
──────────────────                          ─────────────────────────
walk code/  ──► tree-sitter chunk ──┐       question (+ history)
(py/js/ts/go)   (1 chunk/symbol)    │            │
                                    │       rewrite to standalone (if history)
                extract call/import │            │
                edges ──► graph ────┤       decompose compound question
                     │              │            │
                serialize           ├──► embed:  hybrid search each sub-query
                tmp/code_graph.pkl  │    Gemini  over BOTH collections
                                    │    (dense) │
parse docs/ ──► section chunk ──────┘    + BM25  merge + dedupe ──► top-k
(markdown)                               (sparse)     │
                                              ▲   code hits: 1-hop graph expand
                                         Qdrant    doc hits: ±1 section expand
                                       (2 colls)        │
                                                   LLM explains + cites
                                                   (streamed via SSE)
```

| Component | Build time | Runtime |
|---|---|---|
| **Code** | tree-sitter → 1 chunk per function/class/method, extract call+import edges | 1-hop caller/callee expansion |
| **Docs** | markdown → section chunks | ±1 adjacent-section expansion |
| **Dense** | Gemini embeddings (`gemini-embedding-2`, 768-dim) | same |
| **Sparse** | BM25 (fastembed, CPU, ~1ms) — keyword/identifier recall | same |
| **LLM** | — | Gemini 2.5 Pro (swappable: OpenAI / Anthropic) |
| **Store** | Qdrant, two collections (`codebase_rag`, `docs_rag`) | hybrid retrieval |

No GPU, no local model downloads — dense embeddings + LLM are API calls, sparse
is a pure CPU tokenizer.

## Project layout

| File | Role |
|---|---|
| `config.py` | All knobs — paths, models, ports, extensions, graph, GitHub URL map |
| `code_walker.py` | Walk `code/`, filter by extension, skip junk dirs |
| `code_chunker.py` | tree-sitter → one chunk per symbol + call/import edges |
| `code_graph.py` | Build + serialize the call graph (build); O(1) adjacency (runtime) |
| `chunker.py` | Markdown section parsing + chunking |
| `gemini_embedder.py` | Dense embeddings via the Gemini SDK directly |
| `index.py` | Embed (Gemini dense + BM25 sparse) → Qdrant, with batch checkpointing |
| `build.py` | **One command**: walk → chunk → graph → embed → index |
| `code_pipeline.py` | Query: rewrite → decompose → retrieve → graph-expand → LLM |
| `llm.py` | Provider factory (Google / OpenAI / Anthropic) |
| `app.py` | FastAPI streaming SSE server (`POST /ask/stream`) |
| `app_utils.py` | Source serialization + GitHub URL generation |

## Prerequisites

- **Python 3.12+**
- **Docker** (for Qdrant)
- A **Google AI Studio API key** — free at https://aistudio.google.com/apikey

## Quick start

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # add your GOOGLE_API_KEY

# 2. Add content
#    drop repos into code/   and   markdown into docs/

# 3. Start Qdrant
docker compose up -d qdrant

# 4. Build the index (re-run when code/docs change)
python build.py

# 5. Serve
uvicorn app:app --port 8002
```

Wait for `Ready.`, then the API is at http://localhost:8002.

### Adding content

```
code/                 docs/
  my-backend/           guide/intro.md
  my-frontend/          api/reference.md
  some-library/         ...
```

Indexed code types: `.py .js .jsx .ts .tsx .go`. Junk dirs (`node_modules`,
`.venv`, `dist`, `build`, `target`, `vendor`, `.git`, …) are skipped. The first
path component under `code/` becomes the **repo** name. Adjust
`CODE_LANG_BY_EXT` / `CODE_IGNORE_DIRS` in `config.py`.

### Build options

```bash
python build.py            # build both code + docs
python build.py --code     # code only
python build.py --docs     # docs only (wipes + rebuilds docs_rag)
python build.py --resume   # resume from last checkpoint after a crash
```

## Using the API

`POST /ask/stream` returns a Server-Sent Events stream:

```
event: token   data: "partial answer text"
event: token   data: "..."
event: meta    data: {"sources":[...], "related":[...], "sub_queries":[...]}
event: done    data: ""
```

```bash
curl -N -X POST http://localhost:8002/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "how does auth work and how do I add a route?"}'
```

Each source carries `repo`, `file`, `symbol`, `start_line`/`end_line`, a
relevance `score`, and a `github_url` (when the repo is mapped in
`REPO_GITHUB_URLS`). Multi-turn: pass prior turns as `history` — follow-ups are
rewritten to standalone form before retrieval.

**Full frontend integration guide** (React hook + vanilla JS + every event
shape): see [`INTEGRATION.md`](./INTEGRATION.md).

## Configuration

Key knobs in `config.py`:

| Variable | Default | Effect |
|---|---|---|
| `CODE_DIR` / `DOCS_DIR` | `./code` / `./docs` | Where source + docs live |
| `CODE_LANG_BY_EXT` | py/js/ts/go | Extensions → tree-sitter grammar |
| `GRAPH_EXPAND_HOPS` | `1` | Call-graph hops pulled as context |
| `GRAPH_MAX_NEIGHBORS` | `6` | Cap neighbors per hit |
| `MAX_SUBQUERIES` | `4` | Cap on compound-question fan-out |
| `RETRIEVE_TOP_K` / `RERANK_TOP_K` | `10` / `5` | Retrieved per query / sent to LLM |
| `LLM_PROVIDER` / `LLM_MODEL` | `google` / `gemini-2.5-pro` | Swap provider + model |
| `LLM_MAX_OUTPUT_TOKENS` | `8192` | Output budget (thinking models need headroom) |
| `LLM_THINKING_BUDGET` | `1024` | Cap reasoning tokens (Gemini 2.5 thinking) |
| `REPO_GITHUB_URLS` | `{}` | Map repo name → GitHub base URL for citation links |

### GitHub citation links

```python
# config.py
REPO_GITHUB_URLS = {
    "my-backend": "https://github.com/your-org/my-backend",
}
REPO_GITHUB_BRANCH = "main"   # or per-repo: {"my-backend": "develop"}
```

Citations become `https://github.com/your-org/my-backend/blob/main/path.py#L8-L15`.
Computed at query time — no reindex needed when you change the map.

### Switching LLM provider

```python
# config.py
LLM_PROVIDER = "anthropic"          # "google" | "openai" | "anthropic"
LLM_MODEL    = "claude-sonnet-4-6"
```
Set the matching key (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) in `.env`.
Embeddings always use Gemini, so `GOOGLE_API_KEY` stays required.

## Deployment

Full stack (Qdrant + API) via Docker Compose:

```bash
docker compose --profile build run --rm build   # one-shot index build
docker compose up -d qdrant app                  # serve (localhost-bound)
```

Behind a domain with auto-TLS via Caddy — **`flush_interval -1` is required**
or SSE buffers and the answer only appears once it's fully done:

```caddyfile
docs.example.com {
    encode gzip
    handle /ask/stream* {
        reverse_proxy 127.0.0.1:8002 {
            flush_interval -1
            transport http { read_timeout 0 }
        }
    }
    handle {
        reverse_proxy 127.0.0.1:8002
    }
}
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GOOGLE_API_KEY not set` | Copy `.env.example` → `.env`, add your key |
| `Connection refused` to Qdrant | `docker compose up -d qdrant` |
| `No chunks produced` | `code/` has no indexable files (or all in ignored dirs) |
| No graph expansion | `tmp/code_graph.pkl` missing — run `build.py` |
| Empty answer, only Sources shown | `LLM_MAX_OUTPUT_TOKENS` too low for a thinking model — raise it (default is now 8192) |
| Answer appears only when complete (no streaming) | Proxy buffering — set Caddy `flush_interval -1` / nginx `proxy_buffering off` |
| Hitting Gemini free-tier rate limits on build | Increase `_BATCH_SLEEP` in `index.py` |

## License

MIT — see [`LICENSE`](./LICENSE).

## Contributing

Issues and PRs welcome. LiveDocs is provider-agnostic by design — new
embedding/LLM backends and language grammars are especially appreciated.
