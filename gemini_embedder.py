"""Gemini embedders using REST API directly — no SDK model-path mangling."""
import os
import time
import requests
from typing import List

from haystack import component
from haystack.dataclasses import Document

from config import DENSE_EMBEDDING_MODEL, DENSE_EMBEDDING_DIM

_BATCH_SIZE = 100  # internal chunking for large standalone calls; index_documents controls pacing
_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_MAX_RETRIES = 6


def _key():
    k = os.environ.get("GOOGLE_API_KEY")
    if not k:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return k


def _post(url, body, timeout):
    delay = 5
    for attempt in range(_MAX_RETRIES):
        resp = requests.post(url, json=body, params={"key": _key()}, timeout=timeout)
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{_MAX_RETRIES}...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # raise on final attempt


def _embed_single(text: str, model: str) -> List[float]:
    resp = _post(
        f"{_BASE}/{model}:embedContent",
        {"content": {"parts": [{"text": text}]}, "outputDimensionality": DENSE_EMBEDDING_DIM},
        timeout=30,
    )
    return resp.json()["embedding"]["values"]


def _embed_batch(texts: List[str], model: str) -> List[List[float]]:
    resp = _post(
        f"{_BASE}/{model}:batchEmbedContents",
        {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": DENSE_EMBEDDING_DIM,
                }
                for t in texts
            ]
        },
        timeout=120,
    )
    return [e["values"] for e in resp.json()["embeddings"]]


@component
class GeminiDocumentEmbedder:
    """Embeds Documents via Gemini batchEmbedContents REST endpoint. Build-time."""

    def __init__(self, model: str = DENSE_EMBEDDING_MODEL):
        self.model = model

    def warm_up(self):
        pass  # requests uses per-call connections; nothing to initialize

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        for i in range(0, len(documents), _BATCH_SIZE):
            batch = documents[i : i + _BATCH_SIZE]
            texts = [d.content or "" for d in batch]
            embeddings = _embed_batch(texts, self.model)
            for doc, emb in zip(batch, embeddings):
                doc.embedding = emb
        return {"documents": documents}


@component
class GeminiTextEmbedder:
    """Embeds single query string via Gemini embedContent REST endpoint. Query-time."""

    def __init__(self, model: str = DENSE_EMBEDDING_MODEL):
        self.model = model

    def warm_up(self):
        pass

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        return {"embedding": _embed_single(text, self.model)}
