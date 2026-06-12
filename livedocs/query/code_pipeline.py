r"""Query path: hybrid retrieval over code + docs, graph/section expand, stream.

  question (+ history)
       v
  rewrite to standalone        (only if history present)
       v
  decompose into sub-queries   (conditional: LLM only for compound questions)
       v
  per sub-query: dense embed + BM25 sparse — both collections (code + docs)
       v
  merge + dedupe pool  ->  cross-encoder rerank (sigmoid scores, comparable
                           across code + docs)  ->  confidence gate (below
                           MIN_CONFIDENCE: answer "insufficient evidence",
                           skip generation)  ->  docs-first rerank: top
                           DOCS_TOP_K docs, code fills remaining slots up to
                           RERANK_TOP_K
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
8. Always format your entire response in Markdown. Use headers, bullet lists, bold, and fenced code blocks where appropriate.
7. Authority hierarchy: DOCUMENTATION passages describe the supported product behavior, workflows, and user-facing requirements; CODE passages show how things are implemented. When the documentation states that a step, setting, or prerequisite is required as part of a workflow, treat the documentation as authoritative -- even if the code suggests it is technically optional or unenforced. If docs and code conflict, answer according to the documentation first, then briefly note what the code actually does. Only fall back to code-derived behavior when the documentation does not address the question.
"""

USER_PROMPT_TEMPLATE = """{% if history %}=== PRIOR CONVERSATION ===
{% for turn in history %}{{ turn.role }}: {{ turn.content }}
{% endfor %}
{% endif %}{% if doc_passages %}=== CONTEXT: DOCUMENTATION (authoritative for product behavior) ===
{% for doc in doc_passages %}
[Passage {{ loop.index }}] docs/{{ doc.meta.source }} [{{ doc.meta.heading }}{% if doc.meta.subheading %} > {{ doc.meta.subheading }}{% endif %}]
{{ doc.content }}

---
{% endfor %}
{% endif %}{% if code_passages %}=== CONTEXT: CODE (implementation detail) ===
{% for doc in code_passages %}
[Passage {{ loop.index + doc_passages|length }}] {% if doc.meta.get("github_url") %}{{ doc.meta.github_url }}{% else %}{{ doc.meta.repo }}/{{ doc.meta.file }}:{{ doc.meta.start_line }}-{{ doc.meta.end_line }}{% endif %} ({{ doc.meta.symbol_type }} {{ doc.meta.symbol }}, {{ doc.meta.lang }})
{{ doc.content }}

---
{% endfor %}
{% endif %}{% if related %}=== Related context ===
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

INSUFFICIENT_EVIDENCE_ANSWER = (
    "No sufficiently relevant passages were found in the indexed docs/code for "
    "this question, so I won't guess. The closest matches are listed under "
    "Sources — try rephrasing, or name the specific feature, file, or symbol."
)

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
_ranker = None


def reset():
    """Clear cached components. Called after a build so next request reloads with fresh settings."""
    global _code_store, _docs_store, _dense_emb, _sparse_emb
    global _code_retriever, _docs_retriever, _prompt_builder, _generator, _graph, _lite, _ranker
    _code_store = _docs_store = _dense_emb = _sparse_emb = None
    _code_retriever = _docs_retriever = _prompt_builder = _generator = _graph = _lite = None
    _ranker = None


def _ensure_loaded():
    global _code_store, _docs_store, _dense_emb, _sparse_emb
    global _code_retriever, _docs_retriever, _prompt_builder, _generator, _graph, _lite, _ranker

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
    if hasattr(_dense_emb, 'warm_up'):
        _dense_emb.warm_up()

    _sparse_emb = FastembedSparseTextEmbedder(model=SPARSE_EMBEDDING_MODEL)
    _sparse_emb.warm_up()

    if s.retrieval.reranker_provider == "local" and s.retrieval.reranker_model:
        try:
            from haystack_integrations.components.rankers.fastembed import FastembedRanker
            _ranker = FastembedRanker(
                model_name=s.retrieval.reranker_model,
                top_k=s.retrieval.rerank_candidates,
            )
            _ranker.warm_up()
            print(f"  Reranker (local): {s.retrieval.reranker_model}")
        except Exception as e:
            print(f"  (local reranker load failed, falling back to retrieval scores: {type(e).__name__}: {e})")
            _ranker = None
    else:
        _ranker = None
        if s.retrieval.reranker_provider in ("cloudflare", "cohere", "jina"):
            print(f"  Reranker (API): {s.retrieval.reranker_provider} / {s.retrieval.reranker_model}")

    _code_retriever = QdrantHybridRetriever(document_store=_code_store, top_k=s.retrieval.retrieve_top_k)
    _docs_retriever = QdrantHybridRetriever(document_store=_docs_store, top_k=s.retrieval.retrieve_top_k)

    _prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system(SYSTEM_PROMPT),
            ChatMessage.from_user(USER_PROMPT_TEMPLATE),
        ],
        required_variables=["doc_passages", "code_passages", "question"],
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
    if dense is None:
        raise RuntimeError("Embedding returned None — check API key, rate limits, or base_url for your embedding provider.")
    sparse = _sparse_emb.run(text=subquery)["sparse_embedding"]
    code_docs = _code_retriever.run(query_embedding=dense, query_sparse_embedding=sparse)["documents"]
    docs_docs = _docs_retriever.run(query_embedding=dense, query_sparse_embedding=sparse)["documents"]
    return code_docs + docs_docs


def _is_code_hit(d):
    m = d.meta or {}
    return bool(m.get("node_id") or m.get("repo"))


def _rerank_docs_first(pool, docs_top_k, total_k):
    """Docs-priority selection: reserve up to docs_top_k slots for documentation
    (authoritative for product behavior), fill the rest with code, backfill
    symmetrically when either side runs short. Docs come first in output order."""
    doc_hits = sorted((d for d in pool if not _is_code_hit(d)), key=lambda d: d.score or 0, reverse=True)
    code_hits = sorted((d for d in pool if _is_code_hit(d)), key=lambda d: d.score or 0, reverse=True)

    n_docs = min(docs_top_k, len(doc_hits), total_k)
    selected_docs = doc_hits[:n_docs]
    remaining = total_k - n_docs
    selected_code = code_hits[:remaining]
    shortfall = remaining - len(selected_code)
    if shortfall > 0:
        selected_docs = doc_hits[:n_docs + shortfall]
    return selected_docs, selected_code


def _rerank_cloudflare(question, candidates):
    """Call Cloudflare Workers AI reranker (@cf/baai/bge-reranker-base).
    Endpoint: POST /accounts/{id}/ai/run/{model}
    Scores are raw logits — apply sigmoid to normalize to 0-1."""
    import math
    import requests
    from livedocs.settings import get_settings
    s = get_settings()
    token = s.keys.CLOUDFLARE_API_TOKEN
    if not token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN not set — add it in Settings")
    # Derive base from embedding base_url: strip trailing /v1
    emb_base = s.embedding.base_url.rstrip("/")
    if emb_base.endswith("/v1"):
        cf_ai_base = emb_base[:-3]  # .../accounts/{id}/ai
    else:
        cf_ai_base = emb_base  # fallback: use as-is
    model = s.retrieval.reranker_model or "@cf/baai/bge-reranker-base"
    url = f"{cf_ai_base}/run/{model}"
    contexts = [{"text": d.content or ""} for d in candidates]
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": question, "contexts": contexts},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("result", [])
    # CF returns [{score, index}, ...] — scores are raw logits
    for r in results:
        candidates[r["index"]].score = 1 / (1 + math.exp(-float(r["score"])))
    ranked = sorted(candidates, key=lambda d: d.score or 0, reverse=True)
    return ranked


def _rerank_cohere(question, candidates):
    """Call Cohere Rerank API. Returns list of docs with updated scores (0-1)."""
    import requests
    from livedocs.settings import get_settings
    s = get_settings()
    api_key = s.keys.COHERE_API_KEY
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set — add it in Settings")
    model = s.retrieval.reranker_model or "rerank-v3.5"
    texts = [d.content or "" for d in candidates]
    resp = requests.post(
        "https://api.cohere.com/v2/rerank",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": question, "documents": texts,
              "top_n": len(texts), "return_documents": False},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    # results: [{index, relevance_score}, ...] sorted by score desc
    ordered = [None] * len(candidates)
    for r in results:
        doc = candidates[r["index"]]
        doc.score = float(r["relevance_score"])
        ordered[r["index"]] = doc
    ranked = [d for d in ordered if d is not None]
    ranked.sort(key=lambda d: d.score or 0, reverse=True)
    return ranked


def _rerank_jina(question, candidates):
    """Call Jina Rerank API. Returns list of docs with updated scores (0-1)."""
    import requests
    from livedocs.settings import get_settings
    s = get_settings()
    api_key = s.keys.JINA_API_KEY
    if not api_key:
        raise RuntimeError("JINA_API_KEY not set — add it in Settings")
    model = s.retrieval.reranker_model or "jina-reranker-v2-base-multilingual"
    texts = [d.content or "" for d in candidates]
    resp = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": question, "documents": texts,
              "top_n": len(texts)},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    for r in results:
        candidates[r["index"]].score = float(r["relevance_score"])
    ranked = sorted(candidates, key=lambda d: d.score or 0, reverse=True)
    return ranked


def _cross_encode(question, pool):
    """Deep-score query<->chunk pairs. Overwrites each doc's .score with a
    0-1 relevance probability. Returns (pool, top_score); top_score is None
    when the ranker is unavailable so the confidence gate stays inert."""
    import math
    from livedocs.settings import get_settings

    s = get_settings()
    provider = s.retrieval.reranker_provider

    if not pool:
        return pool, None

    candidates = sorted(pool, key=lambda d: d.score or 0, reverse=True)
    candidates = candidates[:s.retrieval.rerank_candidates]

    try:
        if provider == "cloudflare":
            ranked = _rerank_cloudflare(question, candidates)
        elif provider == "cohere":
            ranked = _rerank_cohere(question, candidates)
        elif provider == "jina":
            ranked = _rerank_jina(question, candidates)
        else:
            # local fastembed cross-encoder
            if _ranker is None:
                return pool, None
            ranked = _ranker.run(query=question, documents=candidates)["documents"]
            # fastembed returns raw logits; convert to sigmoid probabilities
            for d in ranked:
                d.score = 1 / (1 + math.exp(-(d.score or 0)))
    except Exception as e:
        print(f"  (cross-encode [{provider}] failed, using retrieval scores: {type(e).__name__}: {e})")
        return pool, None

    top3 = ", ".join(f"{d.score:.4f}" for d in ranked[:3])
    print(f"  Rerank [{provider}] scores (top 3): {top3}")
    return ranked, (ranked[0].score if ranked else 0.0)


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


def _attach_github_urls(docs):
    for d in docs:
        m = d.meta or {}
        if m.get("repo") and m.get("file") and "github_url" not in m:
            url = _github_url(m["repo"], m["file"], m.get("start_line"), m.get("end_line"))
            if url:
                d.meta["github_url"] = url


def prepare(question, history=None):
    """Returns (msgs, reranked, related, subqueries, confidence).
    msgs is None when the confidence gate fires — reranked then holds the
    near-miss candidates and no prompt is built (generation must be skipped)."""
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

    pool, confidence = _cross_encode(standalone, pool)
    if confidence is not None and confidence < s.retrieval.min_confidence:
        near_misses = sorted(pool, key=lambda d: d.score or 0, reverse=True)[:3]
        _attach_github_urls(near_misses)
        return None, near_misses, [], subqueries, confidence

    selected_docs, selected_code = _rerank_docs_first(
        pool, s.retrieval.docs_top_k, s.retrieval.rerank_top_k,
    )
    reranked = selected_docs + selected_code

    graph_related = _expand_with_graph(reranked)
    doc_related = _expand_docs(reranked)
    related = doc_related + graph_related

    _attach_github_urls(reranked + related)

    msgs = _prompt_builder.run(
        doc_passages=selected_docs,
        code_passages=selected_code,
        related=related,
        question=question,
        history=history,
    )["prompt"]

    return msgs, reranked, related, subqueries, confidence


def ask_stream(question, history=None):
    """Generator yielding (event_type, data) pairs for SSE."""
    try:
        msgs, reranked, related, subqueries, confidence = prepare(question, history)
    except Exception as e:
        yield ("error", f"Retrieval failed: {type(e).__name__}: {e}")
        return

    if msgs is None:
        yield ("token", INSUFFICIENT_EVIDENCE_ANSWER)
        yield ("meta", {
            "sources": docs_to_source_dicts(reranked, with_score=True),
            "related": [],
            "sub_queries": subqueries,
            "confidence": confidence,
            "low_confidence": True,
        })
        return

    token_queue = queue.Queue()
    _DONE = object()

    def _callback(chunk):
        # Cloudflare sometimes sends delta.content as int. Haystack stores the same chunk
        # object and later does "".join([c.content for c in chunks]) — coerce to str here
        # so the stored reference is also fixed before Haystack tries to join them.
        if not isinstance(chunk.content, str):
            chunk.content = str(chunk.content) if chunk.content is not None else ""
        if chunk.content:
            token_queue.put(chunk.content)

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
        "confidence": confidence,
        "low_confidence": False,
    })


def ask(question, history=None):
    """Blocking wrapper — used by /ask endpoint and CLI."""
    _ensure_loaded()
    msgs, reranked, related, subqueries, confidence = prepare(question, history)
    if msgs is None:
        return reranked, related, INSUFFICIENT_EVIDENCE_ANSWER, subqueries
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
