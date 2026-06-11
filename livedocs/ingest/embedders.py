"""Dense embedder provider factory.

make_document_embedder() -> Haystack component (build-time, batch)
make_text_embedder()     -> Haystack component (query-time, single text)

Provider selection reads from get_settings() if not overridden.
"""


def make_document_embedder(provider=None, model=None, dim=None):
    from livedocs.settings import get_settings, EMBEDDING_PRESETS
    s = get_settings()
    provider = provider or s.embedding.provider
    model = model or s.embedding.model
    dim = dim or s.embedding.dim

    if provider == "google":
        from livedocs.ingest.gemini_embedder import GeminiDocumentEmbedder
        return GeminiDocumentEmbedder(model=model, dim=dim)

    if provider == "openai":
        import os
        from haystack.components.embedders import OpenAIDocumentEmbedder
        from haystack.utils import Secret
        return OpenAIDocumentEmbedder(
            model=model,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            dimensions=dim,
        )

    if provider == "local":
        from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
        return FastembedDocumentEmbedder(model=model)

    if provider == "cloudflare":
        from livedocs.ingest.cloudflare_embedder import CloudflareDocumentEmbedder
        base_url = s.embedding.base_url
        if not base_url:
            raise ValueError(
                "Cloudflare embedding provider needs embedding.base_url set to your OpenAI-compatible "
                "endpoint, e.g. https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/v1. "
                "Set it in the dashboard Settings tab."
            )
        import os
        os.environ["CLOUDFLARE_EMBEDDING_BASE_URL"] = base_url
        return CloudflareDocumentEmbedder(model=model)

    raise ValueError(
        f"Unknown embedding provider {provider!r}. Use 'google', 'openai', 'local', or 'cloudflare'."
    )


def make_text_embedder(provider=None, model=None, dim=None):
    from livedocs.settings import get_settings
    s = get_settings()
    provider = provider or s.embedding.provider
    model = model or s.embedding.model
    dim = dim or s.embedding.dim

    if provider == "google":
        from livedocs.ingest.gemini_embedder import GeminiTextEmbedder
        return GeminiTextEmbedder(model=model, dim=dim)

    if provider == "openai":
        from haystack.components.embedders import OpenAITextEmbedder
        from haystack.utils import Secret
        return OpenAITextEmbedder(
            model=model,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            dimensions=dim,
        )

    if provider == "local":
        from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
        return FastembedTextEmbedder(model=model)

    if provider == "cloudflare":
        from livedocs.ingest.cloudflare_embedder import CloudflareTextEmbedder
        base_url = s.embedding.base_url
        if not base_url:
            raise ValueError(
                "Cloudflare embedding provider needs embedding.base_url set to your OpenAI-compatible "
                "endpoint, e.g. https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/v1. "
                "Set it in the dashboard Settings tab."
            )
        import os
        os.environ["CLOUDFLARE_EMBEDDING_BASE_URL"] = base_url
        return CloudflareTextEmbedder(model=model)

    raise ValueError(
        f"Unknown embedding provider {provider!r}. Use 'google', 'openai', 'local', or 'cloudflare'."
    )
