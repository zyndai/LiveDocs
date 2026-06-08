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
from chunker import parse_docs_tree, chunk_sections
from config import (
    DOCS_DIR,
    DENSE_EMBEDDING_MODEL, DENSE_EMBEDDING_DIM, SPARSE_EMBEDDING_MODEL,
    QDRANT_URL, DOCS_QDRANT_COLLECTION,
)

from haystack import Pipeline, Document
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder


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


def index_documents(documents, collection, recreate=True):
    """Embed (dense OpenAI + sparse BM25) and write documents to a Qdrant collection.

    Shared by the markdown and code build paths.
    Dense: OpenAI API calls (batched by Haystack). Sparse: BM25 fastembed (CPU, instant).
    """
    import os
    print(f"\n--- Connecting to Qdrant at {QDRANT_URL} (collection: {collection}) ---")
    document_store = QdrantDocumentStore(
        url=QDRANT_URL,
        index=collection,
        embedding_dim=DENSE_EMBEDDING_DIM,
        recreate_index=recreate,
        use_sparse_embeddings=True,
    )

    print(f"--- Dense embedder: Gemini {DENSE_EMBEDDING_MODEL} (free tier) ---")
    dense_embedder = GoogleGenAIDocumentEmbedder(
        model=DENSE_EMBEDDING_MODEL,
        api_key=Secret.from_env_var("GOOGLE_API_KEY"),
    )
    print(f"--- Sparse embedder: BM25 fastembed (CPU, no download) ---")
    sparse_embedder = FastembedSparseDocumentEmbedder(model=SPARSE_EMBEDDING_MODEL)
    writer = DocumentWriter(document_store=document_store)

    pipeline = Pipeline()
    pipeline.add_component("dense_embedder", dense_embedder)
    pipeline.add_component("sparse_embedder", sparse_embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    print(f"\n--- Embedding & writing {len(documents)} documents ---")
    result = pipeline.run({"dense_embedder": {"documents": documents}})
    written = result.get("writer", {}).get("documents_written")
    print(f"\nDone. {written} documents written.")
    return written


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
