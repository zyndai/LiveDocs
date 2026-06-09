r"""Query path: hybrid retrieval over code + docs, graph/section expand, stream.

  question (+ history)
       v
  rewrite to standalone        (only if history present)
       v
  decompose into sub-queries   (conditional: LLM only for compound questions)
       v
  per sub-query: dense embed + BM25 sparse — both collections (code + docs)
       v
  merge + dedupe pool  ->  top RERANK_TOP_K by score
       v
  code hits: graph expansion (1-hop callers/callees, O(1) RAM lookup)
  doc  hits: section-neighbor expansion (seq±1 within same source file)
       v
  LLM explain + cite  (streaming via SSE callback bridge)
"""
import os
import queue
import threading

from dotenv import load_dotenv
load_dotenv()

from livedocs.config import (
    SPARSE_EMBEDDING_MODEL,
    CODE_QDRANT_COLLECTION, DOCS_QDRANT_COLLECTION,
)

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

from livedocs.query.llm import make_generator, make_lite_client
from livedocs.ingest.code_graph import CodeGraph
from livedocs.query.app_utils import _github_url, docs_to_source_dicts
from livedocs.ingest.embedders import make_text_embedder


SYSTEM_PROMPT = """You are an assistant that answers questions about a codebase and its documentation using the passages provided as context.

Rules:
1. Ground every concrete claim (function names, signatures, behavior, file locations) in the provided passages. Do NOT invent APIs, functions, or files that aren't in the context.
2. You MAY explain concepts, summarize how pieces fit together, and reason about the code and docs -- as long as the specifics come from the passages.
3. When you reference code, cite it inline using the exact location string shown in the passage header (GitHub URL if present, otherwise repo/file:start-end). When you reference docs, cite as `docs/source:heading`.
4. If the context genuinely lacks what's needed, say what's missing and which area to look in -- don't pretend.
5. Be direct and technical. Show short code/signatures or doc excerpts from the context when they answer the question.
6. End with a "Sources:" line listing each location you actually used, one per line.
"""

USER_PROMPT_TEMPLATE = """{% if history %}=== PRIOR CONVERSATION ===
{% for turn in history %}{{ turn.role }}: {{ turn.content }}
{% endfor %}
{% endif %}=== CONTEXT ===
{% for doc in documents %}
{% if doc.meta.get("node_id") or doc.meta.get("repo") %}[Passage {{ loop.index }}] {% if doc.meta.get("github_url") %}{{ doc.meta.github_url }}{% else %}{{ doc.meta.repo }}/{{ doc.meta.file }}:{{ doc.meta.start_line }}-{{ doc.meta.end_line }}{% endif %} ({{ doc.meta.symbol_type }} {{ doc.meta.symbol }}, {{ doc.meta.lang }})
{% else %}[Passage {{ loop.index }}] docs/{{ doc.meta.source }} [{{ doc.meta.heading }}{% if doc.meta.subheading %} > {{ doc.meta.subheading }}{% endif %}]
{% endif %}{{ doc.content }}

---
{% endfor %}
{% if related %}=== Related context ===
{% for doc in related %}
{% if doc.meta.get("node_id") or doc.meta.get("repo") %}[Related] {% if doc.meta.get("github_url") %}{{ doc.meta.github_url }}{% else %}{{ doc.meta.repo }}/{{ doc.meta.file }}:{{ doc.meta.start_line }}-{{ doc.meta.end_line }}{% endif %} ({{ doc.meta.symbol_type }} {{ doc.meta.symbol }})
{% else %}[Related] docs/{{ doc.meta.source }} [{{ doc.meta.heading }}{% if doc.meta.subheading %} > {{ doc.meta.subheading }}{% endif %}]
{% endif %}{{ doc.content }}

---
{% endfor %}
{% endif %}
=== QUESTION ===
{{ question }}

=== ANSWER ==="""


REWRITER_SYSTEM_PROMPT = """You rewrite the user's latest message into a single standalone question that captures all context from the conversation so it can be used for retrieval.

Rules:
- If the latest message is already self-contained, return it UNCHANGED.
- Resolve pronouns and implicit references ("that function", "it", "the second one") using the history.
- If the latest message clearly changes topic, return it unchanged.
- Output ONLY the rewritten question. No explanation, no quotes, no prefix.
"""

DECOMPOSE_SYSTEM_PROMPT = """You split a developer's question into the minimal set of independent sub-questions needed to answer it from a codebase and its documentation.

Rules:
- If the question asks about ONE thing, return it unchanged as a single line.
- If it genuinely asks about SEPARATE things (often joined by "and"), split into one sub-question per thing.
- Do NOT split phrases where "and" is part of one concept (e.g. "find and replace", "read and write", "drag and drop"). Those stay as one.
- Keep each sub-question self-contained and specific.
- Output one sub-question per line, nothing else. No numbering, no bullets, no commentary.
"""

_COMPOUND_SIGNALS = (" and ", " and\n", ";", "? and", "? how", "? what", "? where", "? why")

_code_store = None
_docs_store = None
_dense_emb = None
_sparse_emb = None
_code_retriever = None
_docs_retriever = None
_prompt_builder = None
_generator = None
_graph = None
_lite = None


def reset():
    """Clear cached components. Called after a build so next request reloads with fresh settings."""
    global _code_store, _docs_store, _dense_emb, _sparse_emb
    global _code_retriever, _docs_retriever, _prompt_builder, _generator, _graph, _lite
    _code_store = _docs_store = _dense_emb = _sparse_emb = None
    _code_retriever = _docs_retriever = _prompt_builder = _generator = _graph = _lite = None


def _ensure_loaded():
    global _code_store, _docs_store, _dense_emb, _sparse_emb
    global _code_retriever, _docs_retriever, _prompt_builder, _generator, _graph, _lite

    if _code_store is not None:
        return

    from livedocs.settings import get_settings
    from haystack.utils import Secret

    s = get_settings()
    provider = s.embedding.provider

    if provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set. Add it in the dashboard Settings tab.")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Add it in the dashboard Settings tab.")

    qdrant_url = s.qdrant.url
    qdrant_api_key = s.qdrant.api_key
    _qdrant_key = Secret.from_token(qdrant_api_key) if qdrant_api_key else None

    _code_store = QdrantDocumentStore(
        url=qdrant_url, index=CODE_QDRANT_COLLECTION,
        embedding_dim=s.embedding.dim, use_sparse_embeddings=True,
        api_key=_qdrant_key,
    )
    _docs_store = QdrantDocumentStore(
        url=qdrant_url, index=DOCS_QDRANT_COLLECTION,
        embedding_dim=s.embedding.dim, use_sparse_embeddings=True,
        api_key=_qdrant_key,
    )

    _dense_emb = make_text_embedder()
    _dense_emb.warm_up()

    _sparse_emb = FastembedSparseTextEmbedder(model=SPARSE_EMBEDDING_MODEL)
    _sparse_emb.warm_up()

    _code_retriever = QdrantHybridRetriever(document_store=_code_store, top_k=s.retrieval.retrieve_top_k)
    _docs_retriever = QdrantHybridRetriever(document_store=_docs_store, top_k=s.retrieval.retrieve_top_k)

    _prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system(SYSTEM_PROMPT),
            ChatMessage.from_user(USER_PROMPT_TEMPLATE),
        ],
        required_variables=["documents", "question"],
    )
    _generator = make_generator()
    _lite = make_lite_client()

    _graph = CodeGraph.load()
    if _graph is None:
        print("  (no code graph — graph expansion disabled; run a build)")
    else:
        print(f"  Code graph: {_graph.stats()}")


def warm_up():
    _ensure_loaded()


def _is_compound(question):
    q = question.lower()
    return any(s in q for s in _COMPOUND_SIGNALS) or q.count("?") > 1


def rewrite_question(question, history):
    if not history:
        return question
    history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    user_prompt = f"Conversation so far:\n{history_text}\n\nLatest user message: {question}\n\nStandalone question:"
    try:
        return _lite(REWRITER_SYSTEM_PROMPT, user_prompt) or question
    except Exception as e:
        print(f"  (rewrite failed: {type(e).__name__}: {e})")
        return question


def decompose_question(question):
    from livedocs.settings import get_settings
    if not _is_compound(question):
        return [question]
    try:
        user_prompt = f"Question: {question}\n\nSub-questions:"
        result = _lite(DECOMPOSE_SYSTEM_PROMPT, user_prompt)
        lines = [l.strip(" -*\t") for l in (result or "").splitlines() if l.strip()]
        subs = [l for l in lines if l]
        max_sq = get_settings().retrieval.max_subqueries
        return subs[:max_sq] if subs else [question]
    except Exception as e:
        print(f"  (decompose failed: {type(e).__name__}: {e})")
        return [question]


def _retrieve_one(subquery):
    dense = _dense_emb.run(text=subquery)["embedding"]
    sparse = _sparse_emb.run(text=subquery)["sparse_embedding"]
    code_docs = _code_retriever.run(query_embedding=dense, query_sparse_embedding=sparse)["documents"]
    docs_docs = _docs_retriever.run(query_embedding=dense, query_sparse_embedding=sparse)["documents"]
    return code_docs + docs_docs


def _dedupe(docs):
    best = {}
    for d in docs:
        m = d.meta or {}
        primary = m.get("node_id") or m.get("source", "")
        key = (primary, m.get("chunk_index", 0))
        if key not in best or (d.score or 0) > (best[key].score or 0):
            best[key] = d
    return list(best.values())


def _expand_with_graph(reranked):
    from livedocs.settings import get_settings
    if _graph is None:
        return []
    s = get_settings()
    have = {d.meta.get("node_id") for d in reranked}
    neighbor_ids = []
    for d in reranked:
        nid = d.meta.get("node_id")
        if not nid:
            continue
        for n in _graph.neighbors(nid):
            if n not in have and n not in neighbor_ids:
                neighbor_ids.append(n)
    if not neighbor_ids:
        return []
    filters = {"field": "meta.node_id", "operator": "in", "value": neighbor_ids}
    related = _code_store.filter_documents(filters=filters)
    seen, out = set(), []
    for d in related:
        nid = d.meta.get("node_id")
        if nid in seen:
            continue
        seen.add(nid)
        out.append(d)
    return out[:s.retrieval.rerank_top_k]


def _expand_docs(reranked):
    from livedocs.settings import get_settings
    s = get_settings()
    have_keys = {
        (d.meta.get("source", ""), d.meta.get("chunk_index", 0))
        for d in reranked
    }
    out = []
    seen_keys = set(have_keys)

    for d in reranked:
        m = d.meta or {}
        source = m.get("source")
        seq = m.get("seq")
        if not source or seq is None or m.get("node_id"):
            continue
        for neighbor_seq in (seq - 1, seq + 1):
            if neighbor_seq < 0:
                continue
            key = (source, neighbor_seq)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            filters = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.source", "operator": "==", "value": source},
                    {"field": "meta.seq", "operator": "==", "value": neighbor_seq},
                ],
            }
            neighbors = _docs_store.filter_documents(filters=filters)
            out.extend(neighbors)

    return out[:s.retrieval.rerank_top_k]


def prepare(question, history=None):
    from livedocs.settings import get_settings
    _ensure_loaded()
    history = history or []
    s = get_settings()

    standalone = rewrite_question(question, history)
    subqueries = decompose_question(standalone)

    pool = []
    for sq in subqueries:
        pool.extend(_retrieve_one(sq))
    pool = _dedupe(pool)

    reranked = sorted(pool, key=lambda d: d.score or 0, reverse=True)[:s.retrieval.rerank_top_k]

    graph_related = _expand_with_graph(reranked)
    doc_related = _expand_docs(reranked)
    related = graph_related + doc_related

    for d in reranked + related:
        m = d.meta or {}
        if m.get("repo") and m.get("file") and "github_url" not in m:
            url = _github_url(m["repo"], m["file"], m.get("start_line"), m.get("end_line"))
            if url:
                d.meta["github_url"] = url

    msgs = _prompt_builder.run(
        documents=reranked,
        related=related,
        question=question,
        history=history,
    )["prompt"]

    return msgs, reranked, related, subqueries


def ask_stream(question, history=None):
    """Generator yielding (event_type, data) pairs for SSE."""
    try:
        msgs, reranked, related, subqueries = prepare(question, history)
    except Exception as e:
        yield ("error", f"Retrieval failed: {type(e).__name__}: {e}")
        return

    token_queue = queue.Queue()
    _DONE = object()

    def _callback(chunk):
        text = getattr(chunk, "content", "") or ""
        if text:
            token_queue.put(text)

    def _run_gen():
        try:
            gen = make_generator(streaming_callback=_callback)
            gen.run(messages=msgs)
        except Exception as exc:
            token_queue.put(exc)
        finally:
            token_queue.put(_DONE)

    thread = threading.Thread(target=_run_gen, daemon=True)
    thread.start()

    while True:
        item = token_queue.get()
        if item is _DONE:
            break
        if isinstance(item, Exception):
            yield ("error", f"Generation failed: {type(item).__name__}: {item}")
            return
        yield ("token", item)

    yield ("meta", {
        "sources": docs_to_source_dicts(reranked, with_score=True),
        "related": docs_to_source_dicts(related, with_score=False),
        "sub_queries": subqueries,
    })


def ask(question, history=None):
    """Blocking wrapper — used by /ask endpoint and CLI."""
    _ensure_loaded()
    msgs, reranked, related, subqueries = prepare(question, history)
    answer = _generator.run(messages=msgs)["replies"][0].text
    return reranked, related, answer, subqueries


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "how does the code walker decide which files to index?"
    reranked, related, answer, subs = ask(q)
    print(f"\nQuery: {q}")
    print(f"Sub-queries: {subs}")
    print(f"\n--- {len(reranked)} reranked hits ---")
    for d in reranked:
        m = d.meta
        if m.get("node_id") or m.get("repo"):
            print(f"  {m['repo']}/{m['file']}:{m['start_line']}-{m['end_line']} score={d.score:.3f}")
        else:
            print(f"  docs/{m['source']} [{m['heading']}] score={d.score:.3f}")
    print(f"\n--- Answer ---\n{answer}")
