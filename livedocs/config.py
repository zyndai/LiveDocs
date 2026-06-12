"""Central config + knobs for LiveDocs. Single source of truth for all paths."""
from pathlib import Path

# --- Paths ---
# config.py lives at livedocs/config.py; project root is its grandparent.
# code/, docs/, tmp/ all live at the project root, not inside the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = PROJECT_ROOT
# Drop .md files here to index prose docs. No git fetch required.
DOCS_DIR = PROJECT_ROOT / "docs"

# --- Docs source (optional remote fetch via fetch_docs.py) ---
DOCS_REPO_URL = "https://github.com/zyndai/docs"
DOCS_BRANCH = "main"

# --- Chunking ---
MAX_TOKENS = 800
MIN_TOKENS = 150
OVERLAP_TOKENS = 120
MERGE_STOP_TOKENS = 400
TOKENIZER_ENCODING = "cl100k_base"
IGNORE_DIRS = {".vitepress", "node_modules", ".git", "dist", ".cache"}


# =====================================================================
# CODEBASE RAG (vector + graph). Self-contained from the markdown config
# above; the code path uses only the keys in this section plus the shared
# embedding / reranker / LLM settings below.
# =====================================================================

# --- Code source ---
# Drop any number of repos/folders into CODE_DIR. The walker recurses into all
# of them. First path component under CODE_DIR is treated as the "repo" name.
CODE_DIR = HERE / "code"

# File extensions to index, mapped to the tree-sitter grammar name used by
# tree-sitter-language-pack. Anything not listed is skipped.
CODE_LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
}
CODE_EXTENSIONS = set(CODE_LANG_BY_EXT)

# Directories never walked (build artifacts, vendored deps, VCS, caches).
CODE_IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "vendor", "dist", "build", "out", "target",
    ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".cache", ".next", ".nuxt", "coverage", ".idea", ".vscode",
}

# Skip files larger than this (generated bundles, minified, lockfiles).
CODE_MAX_FILE_BYTES = 1_000_000

# Code-chunk sizing (token budget per symbol chunk; oversized funcs are split).
CODE_MAX_TOKENS = 1000
CODE_OVERLAP_TOKENS = 120

# --- Graph ---
# Serialized at build time, loaded into RAM at server start. Runtime lookups
# are O(1) adjacency reads -- no traversal cost.
GRAPH_PATH = HERE / "tmp" / "code_graph.pkl"
# How many graph hops of neighbors to pull in as extra LLM context per hit.
GRAPH_EXPAND_HOPS = 1
# Cap neighbors pulled per retrieved chunk so context stays bounded.
GRAPH_MAX_NEIGHBORS = 6

# --- Qdrant collections ---
CODE_QDRANT_COLLECTION = "codebase_rag"
DOCS_QDRANT_COLLECTION = "docs_rag"

# --- GitHub URL mapping ---
# Maps repo folder name (first component of code/) to its GitHub base URL.
# Used to generate clickable links: {base}/blob/{branch}/{file_path}#L{start}-L{end}
# Leave empty to disable GitHub links (falls back to local path citations).
REPO_GITHUB_URLS: dict[str, str] = {
    "AgentDNS": "https://github.com/zyndai/AgentDNS",
}
# Default branch for all repos. Override per-repo: {"AgentDNS": "main", "old-repo": "master"}
REPO_GITHUB_BRANCH: str | dict[str, str] = "main"

# --- Query decomposition ---
# Max sub-queries a compound question is split into (caps fan-out latency).
MAX_SUBQUERIES = 4


# --- Embedding ---
# Dense: Gemini Embedding 2 — latest available model, outputDimensionality=768.
# text-embedding-004 is NOT available on AI Studio free keys; use gemini-embedding-2.
DENSE_EMBEDDING_MODEL = "gemini-embedding-2"
DENSE_EMBEDDING_DIM = 768

# Sparse: BM25 via fastembed — pure tokenizer, no neural model, runs in ~1ms on CPU.
# Provides keyword/exact-match recall for code identifiers. No API cost.
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# Reranker: fastembed cross-encoder (local, CPU, no API cost). Deep-scores
# query<->chunk pairs after hybrid retrieval, before docs-first slotting.
# Fast alternative: "Xenova/ms-marco-MiniLM-L-6-v2" (~10x faster, weaker on code).
# Set to "" to disable and fall back to raw Qdrant fusion scores.
# Provider: "cloudflare" | "local" (fastembed) | "cohere" | "jina"
RERANKER_PROVIDER = "cloudflare"
RERANKER_MODEL = "@cf/baai/bge-reranker-base"
# Cap candidates fed to the cross-encoder (cost is linear in pool size).
RERANK_CANDIDATES = 40
# Confidence gate: if the best sigmoid(logit) score after reranking is below
# this, skip generation and return an insufficient-evidence answer. 0 disables.
# BGE reranker logits are very negative for technical content — sigmoid scores
# for relevant pairs land around 0.01-0.05; off-topic is ~0.0001. 0.005 is a
# safe default: blocks truly unrelated queries, passes everything in-domain.
MIN_CONFIDENCE = 0.005

# --- LLM ---
# Provider: "google" | "openai" | "anthropic". Swap = change LLM_PROVIDER + LLM_MODEL + env var.
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-2.5-pro"
LLM_TEMPERATURE = 0.2
# gemini-2.5-pro is a THINKING model: thinking tokens count against this cap.
# 1024 was too low -- thinking ate the budget and the visible answer got
# truncated (finishReason=MAX_TOKENS), leaving only the Sources line.
LLM_MAX_OUTPUT_TOKENS = 8192
# Cap internal reasoning so it can't consume the whole output budget.
# gemini-2.5-pro minimum thinking_budget is 128; -1 = dynamic (model decides).
LLM_THINKING_BUDGET = 1024

# Cheap+fast model for rewrite/decompose calls (same provider as LLM_PROVIDER).
REWRITER_MODEL = "gemini-2.5-flash"

# --- Qdrant ---
import os as _os
QDRANT_URL = _os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = _os.environ.get("QDRANT_API_KEY")  # None = no auth (local Docker)

# --- Retrieval ---
RETRIEVE_TOP_K = 10
RERANK_TOP_K = 6
# Rerank slots reserved for documentation passages. Docs are authoritative for
# product behavior/workflows, so they get first claim on the context budget;
# code fills the remaining slots. 0 = code-first (docs only as backfill).
DOCS_TOP_K = 3

# --- Question log ---
# Every question asked via /ask or /ask/stream is recorded here (SQLite).
# Shown in the dashboard "Questions" tab.
QUESTIONS_DB_PATH = HERE / "tmp" / "questions.db"

# --- API ---
API_PORT = 8002    # main app is on 8001
