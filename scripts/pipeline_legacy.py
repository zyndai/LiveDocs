r"""Step 4: Haystack retrieval + reranking + LLM (grounded answer with citations).

  question text
       v
  dense_text_embedder  +  sparse_text_embedder    (run in parallel)
       v                       v
       \_______________________/
                  v
        QdrantHybridRetriever          (fuses dense+sparse with RRF internally)
                  v
       up to RETRIEVE_TOP_K Documents
                  v
   TransformersSimilarityRanker        (cross-encoder, narrows to RERANK_TOP_K)
                  v
        ChatPromptBuilder              (Jinja2 template: system + context + question)
                  v
   GoogleGenAIChatGenerator            (Gemma 4 / Gemini via Google AI Studio)
                  v
         grounded answer + citations

Run from inside the ragpipline folder:
    python pipeline.py "how do I deploy an agent?"
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from dotenv import load_dotenv
load_dotenv()

from livedocs.config import (
    DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_COLLECTION,
    RETRIEVE_TOP_K, RERANK_TOP_K, RERANKER_MODEL, DENSE_EMBEDDING_DIM,
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_OUTPUT_TOKENS,
    REWRITER_MODEL,
)

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Direct google-genai SDK used for the rewriter (cheaper than spinning up another Haystack pipeline).
from google import genai
from google.genai import types as genai_types


SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question using ONLY the context passages provided.

Rules:
1. Answer ONLY from the provided context. If the context does not contain the answer, reply exactly: "The documentation does not cover this." Do not guess, invent, or use outside knowledge.
2. Be concise and direct. Prefer code/commands from the context verbatim when relevant.
3. After your answer, add a "Sources:" line listing the source citations of the passages you actually used, one per line, in the format: `<source> > <heading> > <subheading>`.
4. Do not mention "context", "passage numbers", or your own constraints in the answer body. Just answer the question.
"""

USER_PROMPT_TEMPLATE = """{% if history %}=== PRIOR CONVERSATION ===
{% for turn in history %}{{ turn.role }}: {{ turn.content }}
{% endfor %}
{% endif %}=== CONTEXT ===
{% for doc in documents %}
[Passage {{ loop.index }}] (source: {{ doc.meta.source }} > {{ doc.meta.heading }} > {{ doc.meta.subheading }})
{{ doc.content }}

---
{% endfor %}

=== QUESTION ===
{{ question }}

=== ANSWER ==="""


REWRITER_SYSTEM_PROMPT = """You rewrite the user's latest message into a single standalone question that captures all context from the conversation so it can be used for document retrieval.

Rules:
- If the latest message is already self-contained, return it UNCHANGED.
- Resolve pronouns and implicit references ("that", "it", "the second one") using the history.
- If the latest message clearly changes topic, return it unchanged (don't bleed in old context).
- Output ONLY the rewritten question. No explanation, no quotes, no "Standalone question:" prefix.
"""


def rewrite_question(question, history):
    """Rewrite a possibly-context-dependent question as a standalone question using chat history.
    Returns the original question if history is empty (saves an LLM call)."""
    if not history:
        return question

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    prompt = (
        f"{REWRITER_SYSTEM_PROMPT}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message: {question}\n\n"
        f"Standalone question:"
    )
    response = client.models.generate_content(
        model=REWRITER_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
        ),
    )
    return (response.text or question).strip()


_pipeline = None


def build_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set. Copy .env.example to .env and add your key.")

    document_store = QdrantDocumentStore(
        url=QDRANT_URL,
        index=QDRANT_COLLECTION,
        embedding_dim=DENSE_EMBEDDING_DIM,
        use_sparse_embeddings=True,
    )

    dense_emb = SentenceTransformersTextEmbedder(model=DENSE_EMBEDDING_MODEL)
    sparse_emb = FastembedSparseTextEmbedder(model=SPARSE_EMBEDDING_MODEL)

    retriever = QdrantHybridRetriever(
        document_store=document_store,
        top_k=RETRIEVE_TOP_K,
    )

    ranker = TransformersSimilarityRanker(
        model=RERANKER_MODEL,
        top_k=RERANK_TOP_K,
        scale_score=True,
    )

    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system(SYSTEM_PROMPT),
            ChatMessage.from_user(USER_PROMPT_TEMPLATE),
        ],
        required_variables=["documents", "question"],
    )

    generator = GoogleGenAIChatGenerator(
        model=LLM_MODEL,
        api_key=Secret.from_env_var("GOOGLE_API_KEY"),
        generation_kwargs={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
        },
    )

    p = Pipeline()
    p.add_component("dense_emb", dense_emb)
    p.add_component("sparse_emb", sparse_emb)
    p.add_component("retriever", retriever)
    p.add_component("ranker", ranker)
    p.add_component("prompt_builder", prompt_builder)
    p.add_component("generator", generator)

    p.connect("dense_emb.embedding", "retriever.query_embedding")
    p.connect("sparse_emb.sparse_embedding", "retriever.query_sparse_embedding")
    p.connect("retriever.documents", "ranker.documents")
    p.connect("ranker.documents", "prompt_builder.documents")
    p.connect("prompt_builder.prompt", "generator.messages")

    _pipeline = p
    return p


def ask(question, history=None):
    """Run the full pipeline. Returns (retrieved, reranked, answer_text, retrieval_query).

    history: optional list of dicts like [{"role": "user"|"assistant", "content": "..."}].
    When present, the question is first rewritten into a standalone form for retrieval,
    and the history is included in the final answer prompt for context.
    """
    history = history or []
    retrieval_query = rewrite_question(question, history)

    p = build_pipeline()
    result = p.run(
        {
            "dense_emb": {"text": retrieval_query},
            "sparse_emb": {"text": retrieval_query},
            "ranker": {"query": retrieval_query},
            # Final prompt sees ORIGINAL question + full history (richer than the rewrite).
            "prompt_builder": {"question": question, "history": history},
        },
        include_outputs_from={"retriever", "ranker"},
    )
    retrieved = result["retriever"]["documents"]
    reranked = result["ranker"]["documents"]
    answer = result["generator"]["replies"][0].text
    return retrieved, reranked, answer, retrieval_query


def print_results(question, retrieval_query, retrieved, reranked, answer):
    print(f"\nQuery: {question}")
    if retrieval_query != question:
        print(f"Rewritten for retrieval: {retrieval_query}")

    print(f"\n--- Stage 1: hybrid retrieval (RRF) -> {len(retrieved)} chunks ---")
    for rank, d in enumerate(retrieved, 1):
        m = d.meta
        cite = f"{m.get('source')} > {m.get('heading')} > {m.get('subheading')}"
        print(f"[{rank:>2}] rrf={d.score:.4f}  {cite}")

    print(f"\n--- Stage 2: cross-encoder rerank -> top {len(reranked)} ---")
    for rank, d in enumerate(reranked, 1):
        m = d.meta
        cite = f"{m.get('source')} > {m.get('heading')} > {m.get('subheading')}"
        print(f"[{rank}] rerank={d.score:.4f}  {cite}")

    print(f"\n--- Stage 3: LLM answer ({LLM_MODEL}) ---")
    print(answer)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "how do I deploy an agent?"
    retrieved, reranked, answer, retrieval_query = ask(question)
    print_results(question, retrieval_query, retrieved, reranked, answer)
