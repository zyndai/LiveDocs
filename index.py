"""Haystack docs indexing pipeline.

  parse + chunk docs/ folder (chunker.py)
       v
  build Haystack Documents (content + metadata including seq for neighbor expansion)
       v
  Pipeline:  dense_embedder -> sparse_embedder -> writer
       v
  Qdrant collection 'docs_rag'

Drop .md files in docs/ then run:
    python index.py

To fetch from a remote git repo first, run fetch_docs.py manually.
"""
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from chunker import parse_docs_tree, chunk_sections
from config import (
    DOCS_DIR,
    DENSE_EMBEDDING_MODEL, DENSE_EMBEDDING_DIM, SPARSE_EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_API_KEY, DOCS_QDRANT_COLLECTION,
)

from haystack import Pipeline, Document
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from gemini_embedder import GeminiDocumentEmbedder


def build_documents(chunks):
    """Convert markdown chunk dicts into Haystack Documents.

    Assigns a stable per-source `seq` (document order within each source file)
    so the query path can fetch adjacent sections as related context.
    """
    from collections import defaultdict
    seq_counter = defaultdict(int)
    docs = []
    for c in chunks:
        source = c["source"]
        seq = seq_counter[source]
        seq_counter[source] += 1
        docs.append(Document(
            content=c["content"],
            meta={
                "source": source,
                "product_area": c["product_area"],
                "heading": c["heading"],
                "subheading": c["subheading"],
                "chunk_index": c["chunk_index"],
                "chunk_total": c["chunk_total"],
                "token_count": c["token_count"],
                "seq": seq,
            },
        ))
    return docs


def build_code_documents(chunks):
    """Convert code chunk dicts (code_chunker) into Haystack Documents.

    The graph node id is stored in meta as 'node_id' so the query path can map
    a retrieved chunk straight to its graph node for neighbor expansion.
    """
    from code_graph import qualified_id
    docs = []
    for c in chunks:
        docs.append(Document(
            content=c["content"],
            meta={
                "node_id": qualified_id(c["repo"], c["file"], c["symbol"]),
                "repo": c["repo"],
                "file": c["file"],
                "lang": c["lang"],
                "symbol": c["symbol"],
                "symbol_type": c["symbol_type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "calls": c.get("calls", []),
                "chunk_index": c.get("chunk_index", 0),
                "chunk_total": c.get("chunk_total", 1),
                "token_count": c.get("token_count", 0),
            },
        ))
    return docs


_CHECKPOINT_DIR = Path(__file__).resolve().parent / "tmp" / "checkpoints"
_WRITE_BATCH = 10    # docs per embed+write cycle
_BATCH_SLEEP = 10    # seconds between batches — keeps Gemini free tier under 30K TPM


def _checkpoint_path(collection):
    return _CHECKPOINT_DIR / f"{collection}_progress.json"


def index_documents(documents, collection, recreate=True, resume=False):
    """Embed+write+checkpoint per batch. Crash-safe: resume from last completed batch.

    resume=True: skip already-written batches, continue from checkpoint.
    Each batch: dense embed -> sparse embed -> write to Qdrant -> save checkpoint.
    """
    import json
    from haystack.utils import Secret

    checkpoint = _checkpoint_path(collection)
    start_batch = 0

    if resume and checkpoint.exists():
        data = json.loads(checkpoint.read_text())
        start_batch = data.get("done", 0)
        print(f"\nResuming {collection} from batch {start_batch}/{data.get('total')}")
        recreate = False  # collection already exists, don't wipe it

    batches = [documents[i:i + _WRITE_BATCH] for i in range(0, len(documents), _WRITE_BATCH)]
    total_batches = len(batches)

    print(f"\n--- {collection}: {len(documents)} docs in {total_batches} batches ---")
    print(f"--- Dense: Gemini {DENSE_EMBEDDING_MODEL} | Sparse: BM25 fastembed ---")

    dense_embedder = GeminiDocumentEmbedder(model=DENSE_EMBEDDING_MODEL)
    dense_embedder.warm_up()
    sparse_embedder = FastembedSparseDocumentEmbedder(model=SPARSE_EMBEDDING_MODEL)
    sparse_embedder.warm_up()

    _api_key = Secret.from_token(QDRANT_API_KEY) if QDRANT_API_KEY else None
    print(f"--- Qdrant at {QDRANT_URL} ---")

    doc_store = QdrantDocumentStore(
        url=QDRANT_URL,
        index=collection,
        embedding_dim=DENSE_EMBEDDING_DIM,
        recreate_index=(recreate and start_batch == 0),
        use_sparse_embeddings=True,
        api_key=_api_key,
    )
    writer = DocumentWriter(document_store=doc_store)

    written_total = 0
    for i, batch in enumerate(batches):
        if i < start_batch:
            print(f"  Batch {i+1}/{total_batches} — skipped (already written)")
            continue

        print(f"\n  Batch {i+1}/{total_batches} ({len(batch)} docs) — embedding...")
        embedded = dense_embedder.run(documents=batch)["documents"]
        embedded = sparse_embedder.run(documents=embedded)["documents"]

        # After first write collection exists; never recreate again mid-run
        if i > 0 or not (recreate and start_batch == 0):
            doc_store.recreate_index = False

        result = writer.run(documents=embedded)
        n = result.get("documents_written", len(embedded))
        written_total += n

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"done": i + 1, "total": total_batches}))
        print(f"  Written {n} docs. Progress: {i+1}/{total_batches} checkpointed.")

        if i + 1 < total_batches:
            time.sleep(_BATCH_SLEEP)

    checkpoint.unlink(missing_ok=True)
    print(f"\nDone. {written_total} total documents written to '{collection}'.")
    return written_total


def main():
    if not DOCS_DIR.exists():
        raise SystemExit(
            f"DOCS_DIR not found: {DOCS_DIR}\n"
            "Create the docs/ folder, drop .md files in it, then re-run.\n"
            "To fetch from a remote repo first: python fetch_docs.py"
        )

    print("\n--- Chunking docs/ ---")
    sections = parse_docs_tree(DOCS_DIR)
    chunks = chunk_sections(sections)
    print(f"{len(sections)} sections -> {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No chunks. Check that docs/ contains .md files.")

    documents = build_documents(chunks)
    index_documents(documents, DOCS_QDRANT_COLLECTION, recreate=True)

    print(f"\nVerify: http://localhost:6333/dashboard  (collection: {DOCS_QDRANT_COLLECTION})")



if __name__ == "__main__":
    main()
