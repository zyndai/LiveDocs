"""One command to build both code and docs knowledge bases.

    python build.py

Steps:
  1. walk CODE_DIR for source files              (code_walker)
  2. tree-sitter chunk + extract call edges      (code_chunker)
  3. build + serialize the code graph            (code_graph -> tmp/code_graph.pkl)
  4. embed (bge-m3 + SPLADE) + write Qdrant      collection: codebase_rag
  5. parse + chunk DOCS_DIR markdown files       (chunker)
  6. embed + write Qdrant                        collection: docs_rag

Prereqs: Qdrant running on localhost:6333, deps installed, API key in .env.
Put repos in code/ and .md files in docs/ before running.
"""
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from pathlib import Path

from code_walker import walk_code
from code_chunker import chunk_file
from code_graph import build_graph, save_graph, CodeGraph
from index import build_code_documents, build_documents, index_documents
from chunker import parse_docs_tree, chunk_sections
from config import CODE_QDRANT_COLLECTION, DOCS_QDRANT_COLLECTION, CODE_DIR, DOCS_DIR


def build_code(resume=False):
    print(f"\n=== [1/2] Building CODE index from {CODE_DIR} ===")

    print("\n--- Walking + chunking source files ---")
    all_chunks = []
    imports_by_file = {}
    file_count = 0
    by_lang = defaultdict(int)
    for rec in walk_code():
        chunks, modules = chunk_file(rec)
        if not chunks:
            continue
        file_count += 1
        by_lang[rec["lang"]] += 1
        all_chunks.extend(chunks)
        if modules:
            imports_by_file[(rec["repo"], rec["path"])] = modules

    print(f"{file_count} files -> {len(all_chunks)} chunks  (by lang: {dict(by_lang)})")
    if not all_chunks:
        print(f"  WARNING: No chunks from {CODE_DIR}. Skipping code index.")
        print("  Tip: Drop repos into code/ with .py/.js/.ts/.go files.")
        return

    print("\n--- Building call graph ---")
    graph = build_graph(all_chunks, imports_by_file)
    path = save_graph(graph)
    print(f"Graph stats: {CodeGraph(graph).stats()}")
    print(f"Serialized -> {path}")

    documents = build_code_documents(all_chunks)
    index_documents(documents, CODE_QDRANT_COLLECTION, recreate=not resume, resume=resume)
    print(f"Code index done. Collection: {CODE_QDRANT_COLLECTION}")


def build_docs(resume=False):
    print(f"\n=== [2/2] Building DOCS index from {DOCS_DIR} ===")

    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*.md")):
        print(f"  WARNING: No .md files found in {DOCS_DIR}. Skipping docs index.")
        print("  Tip: Drop markdown files into docs/ then re-run build.py.")
        return

    print("\n--- Parsing + chunking markdown ---")
    sections = parse_docs_tree(DOCS_DIR)
    chunks = chunk_sections(sections)
    print(f"{len(sections)} sections -> {len(chunks)} chunks")

    documents = build_documents(chunks)
    index_documents(documents, DOCS_QDRANT_COLLECTION, recreate=not resume, resume=resume)
    print(f"Docs index done. Collection: {DOCS_QDRANT_COLLECTION}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build code + docs Qdrant indexes.")
    parser.add_argument("--code", action="store_true", help="Build code index only")
    parser.add_argument("--docs", action="store_true", help="Build docs index only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint — skip already-written batches")
    args = parser.parse_args()

    run_code = args.code or (not args.code and not args.docs)
    run_docs = args.docs or (not args.code and not args.docs)

    if run_code:
        build_code(resume=args.resume)
    if run_docs:
        build_docs(resume=args.resume)
    print("\n=== Build complete. Start the server: uvicorn app:app --port 8002 ===")


if __name__ == "__main__":
    main()
