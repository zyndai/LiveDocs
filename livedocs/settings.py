"""Runtime-editable settings backed by config/settings.json.

get_settings() -> Settings (cached, reload on save)
save_settings(patch) -> merge + write + inject env + return new Settings
reload() -> force reload from disk (called after build)
"""
import json
import os
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from livedocs import config as _cfg

_SETTINGS_PATH = _cfg.PROJECT_ROOT / "config" / "settings.json"
_lock = threading.Lock()
_current: Optional["Settings"] = None


@dataclass
class LLMSettings:
    provider: str = _cfg.LLM_PROVIDER
    model: str = _cfg.LLM_MODEL
    temperature: float = _cfg.LLM_TEMPERATURE
    max_output_tokens: int = _cfg.LLM_MAX_OUTPUT_TOKENS
    thinking_budget: int = _cfg.LLM_THINKING_BUDGET
    rewriter_model: str = _cfg.REWRITER_MODEL
    # OpenAI-compatible endpoint base URL (cloudflare provider only).
    base_url: str = ""


@dataclass
class EmbeddingSettings:
    provider: str = "google"
    model: str = _cfg.DENSE_EMBEDDING_MODEL
    dim: int = _cfg.DENSE_EMBEDDING_DIM


@dataclass
class KeysSettings:
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    CLOUDFLARE_API_TOKEN: str = ""


@dataclass
class QdrantSettings:
    url: str = _cfg.QDRANT_URL
    api_key: Optional[str] = _cfg.QDRANT_API_KEY


@dataclass
class RetrievalSettings:
    retrieve_top_k: int = _cfg.RETRIEVE_TOP_K
    rerank_top_k: int = _cfg.RERANK_TOP_K
    max_subqueries: int = _cfg.MAX_SUBQUERIES
    docs_top_k: int = _cfg.DOCS_TOP_K


@dataclass
class GraphSettings:
    expand_hops: int = _cfg.GRAPH_EXPAND_HOPS
    max_neighbors: int = _cfg.GRAPH_MAX_NEIGHBORS


# Embedding provider → (default model, vector dim)
EMBEDDING_PRESETS: dict[str, tuple[str, int]] = {
    "google": ("gemini-embedding-2", 768),
    "openai": ("text-embedding-3-small", 1536),
    "local": ("BAAI/bge-m3", 1024),
}

# LLM provider → suggested default models
LLM_MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
    "cloudflare": [
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/openai/gpt-oss-120b",
        "@cf/qwen/qwen2.5-coder-32b-instruct",
    ],
}


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    keys: KeysSettings = field(default_factory=KeysSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    # list of source dicts: {id, kind, corpus, name, url, branch, github_base, status, counts}
    sources: list = field(default_factory=list)
    deployed: bool = False
    last_build: Optional[dict] = None


def _deep_update(base: dict, patch: dict) -> dict:
    result = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def _from_dict(d: dict) -> Settings:
    def _pick(cls, src):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in src.items() if k in fields})

    s = Settings()
    if "llm" in d:
        s.llm = _pick(LLMSettings, d["llm"])
    if "embedding" in d:
        s.embedding = _pick(EmbeddingSettings, d["embedding"])
    if "keys" in d:
        s.keys = _pick(KeysSettings, d["keys"])
    if "qdrant" in d:
        s.qdrant = _pick(QdrantSettings, d["qdrant"])
    if "retrieval" in d:
        s.retrieval = _pick(RetrievalSettings, d["retrieval"])
    if "graph" in d:
        s.graph = _pick(GraphSettings, d["graph"])
    s.sources = d.get("sources", [])
    s.deployed = d.get("deployed", False)
    s.last_build = d.get("last_build")
    return s


def _to_dict(s: Settings) -> dict:
    return {
        "llm": asdict(s.llm),
        "embedding": asdict(s.embedding),
        "keys": asdict(s.keys),
        "qdrant": asdict(s.qdrant),
        "retrieval": asdict(s.retrieval),
        "graph": asdict(s.graph),
        "sources": s.sources,
        "deployed": s.deployed,
        "last_build": s.last_build,
    }


def _inject_env(s: Settings) -> None:
    """Push API keys + Qdrant config into os.environ so Secret.from_env_var() works."""
    for key, val in asdict(s.keys).items():
        if val:
            os.environ[key] = val
    os.environ["QDRANT_URL"] = s.qdrant.url
    if s.qdrant.api_key:
        os.environ["QDRANT_API_KEY"] = s.qdrant.api_key


def _load() -> Settings:
    if _SETTINGS_PATH.exists():
        try:
            data = json.loads(_SETTINGS_PATH.read_text())
            return _from_dict(data)
        except Exception as e:
            print(f"[settings] Load failed ({e}) — using defaults")
    return Settings()


def get_settings() -> Settings:
    global _current
    if _current is None:
        with _lock:
            if _current is None:
                _current = _load()
                _inject_env(_current)
    return _current


def save_settings(patch: dict) -> Settings:
    """Deep-merge patch into current settings, write JSON, inject env."""
    global _current
    with _lock:
        base = _to_dict(_current) if _current else _to_dict(Settings())
        merged = _deep_update(base, patch)
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
        _current = _from_dict(merged)
        _inject_env(_current)
        return _current


def reload() -> Settings:
    """Force reload from disk. Called after a successful build."""
    global _current
    with _lock:
        _current = _load()
        _inject_env(_current)
        return _current


def get_settings_dict() -> dict:
    return _to_dict(get_settings())
