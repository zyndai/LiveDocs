"""Importable build functions. Wraps ingest pipeline with injectable log callback.

Called by livedocs/jobs.py (background) and scripts/build.py (CLI).
"""
from collections import defaultdict

from livedocs.ingest.code_walker import walk_code
from livedocs.ingest.code_chunker import chunk_file
from livedocs.ingest.code_graph import build_graph, save_graph, CodeGraph
from livedocs.ingest.index import build_code_documents, build_documents, index_documents
from livedocs.ingest.chunker import parse_docs_tree, chunk_sections
from livedocs.config import CODE_QDRANT_COLLECTION, DOCS_QDRANT_COLLECTION, CODE_DIR, DOCS_DIR


def run_build_code(log=print, resume=False) -> dict:
    """Walk code/ → chunk → graph → embed → Qdrant. Returns counts dict."""
    log(f"\n=== [CODE] Building index from {CODE_DIR} ===")

    log("\n--- Walking + chunking source files ---")
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

    log(f"{file_count} files -> {len(all_chunks)} chunks  (by lang: {dict(by_lang)})")

    if not all_chunks:
        log(f"  WARNING: No chunks from {CODE_DIR}. Skipping code index.")
        log("  Tip: Drop repos into code/ with .py/.js/.ts/.go files.")
        return {"files": 0, "chunks": 0}

    log("\n--- Building call graph ---")
    graph = build_graph(all_chunks, imports_by_file)
    path = save_graph(graph)
    stats = CodeGraph(graph).stats()
    log(f"Graph stats: {stats}")
    log(f"Serialized -> {path}")

    documents = build_code_documents(all_chunks)
    written = index_documents(
        documents, CODE_QDRANT_COLLECTION, recreate=not resume, resume=resume, log=log
    )
    log(f"Code index done. Collection: {CODE_QDRANT_COLLECTION}")
    return {"files": file_count, "chunks": len(all_chunks), "written": written, "langs": dict(by_lang)}


def run_build_docs(log=print, resume=False) -> dict:
    """Parse docs/ markdown → chunk → embed → Qdrant. Returns counts dict."""
    log(f"\n=== [DOCS] Building index from {DOCS_DIR} ===")

    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*.md")):
        log(f"  WARNING: No .md files found in {DOCS_DIR}. Skipping docs index.")
        return {"sections": 0, "chunks": 0}

    log("\n--- Parsing + chunking markdown ---")
    sections = parse_docs_tree(DOCS_DIR)
    chunks = chunk_sections(sections)
    log(f"{len(sections)} sections -> {len(chunks)} chunks")

    documents = build_documents(chunks)
    written = index_documents(
        documents, DOCS_QDRANT_COLLECTION, recreate=not resume, resume=resume, log=log
    )
    log(f"Docs index done. Collection: {DOCS_QDRANT_COLLECTION}")
    return {"sections": len(sections), "chunks": len(chunks), "written": written}
