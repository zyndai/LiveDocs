"""Haystack indexing pipeline — embed + write to Qdrant.

Reads provider/model/dim/qdrant config from get_settings().
Supports google, openai, and local (fastembed) dense embedders.
Sparse is always BM25 via fastembed (CPU, no API cost).
"""
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from livedocs.ingest.chunker import parse_docs_tree, chunk_sections
from livedocs.config import (
    DOCS_DIR, PROJECT_ROOT,
    SPARSE_EMBEDDING_MODEL,
)

from haystack import Pipeline, Document
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder


def build_documents(chunks):
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
    from livedocs.ingest.code_graph import qualified_id
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


_CHECKPOINT_DIR = PROJECT_ROOT / "tmp" / "checkpoints"
_WRITE_BATCH = 10


def _checkpoint_path(collection):
    return _CHECKPOINT_DIR / f"{collection}_progress.json"


def index_documents(documents, collection, recreate=True, resume=False, log=print):
    """Embed+write+checkpoint per batch. Crash-safe: resume from last completed batch."""
    import json
    from livedocs.settings import get_settings
    from livedocs.ingest.embedders import make_document_embedder

    s = get_settings()
    qdrant_url = s.qdrant.url
    qdrant_api_key = s.qdrant.api_key
    emb_dim = s.embedding.dim

    checkpoint = _checkpoint_path(collection)
    start_batch = 0

    if resume and checkpoint.exists():
        data = json.loads(checkpoint.read_text())
        start_batch = data.get("done", 0)
        log(f"\nResuming {collection} from batch {start_batch}/{data.get('total')}")
        recreate = False

    batches = [documents[i:i + _WRITE_BATCH] for i in range(0, len(documents), _WRITE_BATCH)]
    total_batches = len(batches)

    log(f"\n--- {collection}: {len(documents)} docs in {total_batches} batches ---")
    log(f"--- Dense: {s.embedding.provider}/{s.embedding.model} dim={emb_dim} | Sparse: BM25 ---")

    dense_embedder = make_document_embedder()
    if hasattr(dense_embedder, 'warm_up'):
        dense_embedder.warm_up()
    sparse_embedder = FastembedSparseDocumentEmbedder(model=SPARSE_EMBEDDING_MODEL)
    sparse_embedder.warm_up()

    _api_key = Secret.from_token(qdrant_api_key) if qdrant_api_key else None
    log(f"--- Qdrant at {qdrant_url} ---")

    doc_store = QdrantDocumentStore(
        url=qdrant_url,
        index=collection,
        embedding_dim=emb_dim,
        recreate_index=(recreate and start_batch == 0),
        use_sparse_embeddings=True,
        api_key=_api_key,
    )
    writer = DocumentWriter(document_store=doc_store)

    written_total = 0
    for i, batch in enumerate(batches):
        if i < start_batch:
            log(f"  Batch {i+1}/{total_batches} — skipped (already written)")
            continue

        log(f"\n  Batch {i+1}/{total_batches} ({len(batch)} docs) — embedding...")
        embedded = dense_embedder.run(documents=batch)["documents"]
        embedded = sparse_embedder.run(documents=embedded)["documents"]

        if i > 0 or not (recreate and start_batch == 0):
            doc_store.recreate_index = False

        result = writer.run(documents=embedded)
        n = result.get("documents_written", len(embedded))
        written_total += n

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"done": i + 1, "total": total_batches}))
        log(f"  Written {n} docs. Progress: {i+1}/{total_batches} batches.")

        if i + 1 < total_batches and s.embedding.batch_sleep > 0:
            time.sleep(s.embedding.batch_sleep)

    checkpoint.unlink(missing_ok=True)
    log(f"\nDone. {written_total} total documents written to '{collection}'.")
    return written_total
