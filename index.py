"""Step 1: Haystack indexing pipeline.

  fetch docs (ragpipline/fetch_docs.py)
       v
  parse + chunk (ragpipline/chunker.py)
       v
  build Haystack Documents (content + metadata)
       v
  Pipeline:  dense_embedder -> sparse_embedder -> writer
       v
  Qdrant collection 'zynd_docs_haystack'

Run from inside the ragpipline folder:
    ..\\.venv\\Scripts\\python.exe index.py
"""
from chunker import parse_docs_tree, chunk_sections
from fetch_docs import fetch
from config import (
    DOCS_DIR,
    DENSE_EMBEDDING_MODEL, DENSE_EMBEDDING_DIM, SPARSE_EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_COLLECTION,
)

from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder


def build_documents(chunks):
    """Convert our chunk dicts into Haystack Document objects."""
    docs = []
    for c in chunks:
        docs.append(Document(
            content=c["content"],
            meta={
                "source": c["source"],
                "product_area": c["product_area"],
                "heading": c["heading"],
                "subheading": c["subheading"],
                "chunk_index": c["chunk_index"],
                "chunk_total": c["chunk_total"],
                "token_count": c["token_count"],
            },
        ))
    return docs


def main():
    # 1. Make sure docs are present locally.
    fetch()

    # 2. Chunk.
    print("\n--- Chunking ---")
    sections = parse_docs_tree(DOCS_DIR)
    chunks = chunk_sections(sections)
    print(f"{len(sections)} sections -> {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No chunks. Check DOCS_DIR.")

    documents = build_documents(chunks)

    # 3. Qdrant document store with both dense + sparse enabled.
    print(f"\n--- Connecting to Qdrant at {QDRANT_URL} (collection: {QDRANT_COLLECTION}) ---")
    document_store = QdrantDocumentStore(
        url=QDRANT_URL,
        index=QDRANT_COLLECTION,
        embedding_dim=DENSE_EMBEDDING_DIM,
        recreate_index=True,            # full re-index every run
        use_sparse_embeddings=True,
    )

    # 4. Embedders.
    print(f"--- Loading dense embedder: {DENSE_EMBEDDING_MODEL} ---")
    dense_embedder = SentenceTransformersDocumentEmbedder(model=DENSE_EMBEDDING_MODEL)
    print(f"--- Loading sparse embedder: {SPARSE_EMBEDDING_MODEL} (downloads ~500MB on first use) ---")
    sparse_embedder = FastembedSparseDocumentEmbedder(model=SPARSE_EMBEDDING_MODEL)

    writer = DocumentWriter(document_store=document_store)

    # 5. Wire the indexing pipeline.
    pipeline = Pipeline()
    pipeline.add_component("dense_embedder", dense_embedder)
    pipeline.add_component("sparse_embedder", sparse_embedder)
    pipeline.add_component("writer", writer)
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    # 6. Run.
    print(f"\n--- Embedding & writing {len(documents)} documents ---")
    result = pipeline.run({"dense_embedder": {"documents": documents}})

    written = result.get("writer", {}).get("documents_written")
    print(f"\nDone. Haystack reports {written} documents written.")
    print(f"Verify in Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()
