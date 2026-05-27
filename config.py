"""Config for the Haystack implementation. Self-contained -- does NOT import
from the root config.py. ragpipline/ is intended to stand alone."""
from pathlib import Path

# --- Paths ---
HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "tmp" / "docs"   # local clone, fully inside ragpipline/.

# --- Docs source (for ragpipline/fetch_docs.py) ---
DOCS_REPO_URL = "https://github.com/zyndai/docs"
DOCS_BRANCH = "main"

# --- Chunking ---
MAX_TOKENS = 800
MIN_TOKENS = 150
OVERLAP_TOKENS = 120
MERGE_STOP_TOKENS = 400
TOKENIZER_ENCODING = "cl100k_base"
IGNORE_DIRS = {".vitepress", "node_modules", ".git", "dist", ".cache"}


# --- Embedding ---
# Dense: same bge-m3 as the main pipeline (used in dense-only mode via sentence-transformers).
DENSE_EMBEDDING_MODEL = "BAAI/bge-m3"
DENSE_EMBEDDING_DIM = 1024

# Sparse: SPLADE via fastembed -- the standard Haystack hybrid sparse choice.
# Replaces bge-m3's joint sparse output (which Haystack doesn't support out of the box).
SPARSE_EMBEDDING_MODEL = "prithivida/Splade_PP_en_v1"

# --- Reranker ---
RERANKER_MODEL = "BAAI/bge-reranker-large"

# --- LLM ---
# Same as main pipeline. Switch to "gemini-2.5-flash" if Gemma 4 misbehaves.
LLM_MODEL = "gemma-4-26b-a4b-it"
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 1024

# Cheap+fast model used only to rewrite follow-up questions into standalone form
# before retrieval. Doesn't need to be smart, just obedient.
REWRITER_MODEL = "gemini-2.5-flash"

# --- Qdrant ---
QDRANT_URL = "http://localhost:6333"
# Separate collection so we don't disturb the main pipeline's 'zynd_docs'.
QDRANT_COLLECTION = "zynd_docs_haystack"

# --- Retrieval ---
RETRIEVE_TOP_K = 10
RERANK_TOP_K = 5

# --- API ---
API_PORT = 8002    # main app is on 8001
