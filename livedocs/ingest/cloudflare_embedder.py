"""Cloudflare Workers AI embedders using openai SDK directly — bypasses Haystack's usage=None crash."""
import os
from typing import List

from haystack import component
from haystack.dataclasses import Document

_BATCH_SIZE = 100


def _client():
    from openai import OpenAI
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN not set")
    base_url = os.environ.get("CLOUDFLARE_EMBEDDING_BASE_URL")
    if not base_url:
        raise RuntimeError("CLOUDFLARE_EMBEDDING_BASE_URL not set")
    return OpenAI(api_key=token, base_url=base_url)


@component
class CloudflareDocumentEmbedder:
    def __init__(self, model: str = "@cf/baai/bge-m3"):
        self.model = model

    def warm_up(self):
        pass

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        client = _client()
        for i in range(0, len(documents), _BATCH_SIZE):
            batch = documents[i: i + _BATCH_SIZE]
            texts = [d.content or "" for d in batch]
            resp = client.embeddings.create(model=self.model, input=texts)
            for doc, item in zip(batch, resp.data):
                doc.embedding = item.embedding
        return {"documents": documents}


@component
class CloudflareTextEmbedder:
    def __init__(self, model: str = "@cf/baai/bge-m3"):
        self.model = model

    def warm_up(self):
        pass

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        client = _client()
        resp = client.embeddings.create(model=self.model, input=text)
        return {"embedding": resp.data[0].embedding}
