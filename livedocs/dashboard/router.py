"""FastAPI router for /dashboard/* — HTMX + Jinja2 self-hosted UI."""
from pathlib import Path

from fastapi import APIRouter, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from livedocs.config import PROJECT_ROOT

router = APIRouter(prefix="/dashboard")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


# ── helpers ──────────────────────────────────────────────────────────────────

def _qdrant_ok() -> bool:
    import requests
    from livedocs.settings import get_settings
    try:
        r = requests.get(f"{get_settings().qdrant.url}/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _base_ctx(request: Request) -> dict:
    from livedocs.settings import get_settings
    from livedocs import jobs
    s = get_settings()
    return {
        "deployed": s.deployed,
        "qdrant_ok": _qdrant_ok(),
        "job": jobs.get_status(),
        "api_base": str(request.base_url).rstrip("/"),
    }


def _tr(request, name, ctx):
    """Shorthand for the Starlette 1.x TemplateResponse signature."""
    return templates.TemplateResponse(request, name, ctx)


# ── overview ─────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    from livedocs.settings import get_settings
    s = get_settings()
    ctx = _base_ctx(request)
    ctx["sources"] = s.sources
    ctx["last_build"] = s.last_build
    ctx["active"] = "overview"
    return _tr(request, "overview.html", ctx)


# ── settings form ─────────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    from livedocs.settings import get_settings, LLM_MODEL_SUGGESTIONS, EMBEDDING_PRESETS
    s = get_settings()
    ctx = _base_ctx(request)
    ctx.update({
        "s": s,
        "llm_suggestions": LLM_MODEL_SUGGESTIONS,
        "embedding_presets": EMBEDDING_PRESETS,
        "saved": False,
        "needs_reindex": False,
        "error": None,
        "active": "settings",
    })
    return _tr(request, "settings_form.html", ctx)


@router.post("/settings", response_class=HTMLResponse)
async def settings_save(
    request: Request,
    llm_provider: str = Form(...),
    llm_model: str = Form(...),
    llm_temperature: float = Form(0.2),
    llm_max_output_tokens: int = Form(8192),
    llm_thinking_budget: int = Form(1024),
    llm_rewriter_model: str = Form(...),
    emb_provider: str = Form(...),
    emb_model: str = Form(...),
    emb_dim: int = Form(...),
    google_api_key: str = Form(""),
    openai_api_key: str = Form(""),
    anthropic_api_key: str = Form(""),
    qdrant_url: str = Form("http://localhost:6333"),
    qdrant_api_key: str = Form(""),
    retrieve_top_k: int = Form(10),
    rerank_top_k: int = Form(5),
    max_subqueries: int = Form(4),
    graph_expand_hops: int = Form(1),
    graph_max_neighbors: int = Form(6),
):
    from livedocs.settings import get_settings, save_settings, EMBEDDING_PRESETS, LLM_MODEL_SUGGESTIONS

    old_emb = get_settings().embedding
    needs_reindex = (
        old_emb.provider != emb_provider or
        old_emb.model != emb_model or
        old_emb.dim != emb_dim
    )

    patch = {
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "temperature": llm_temperature,
            "max_output_tokens": llm_max_output_tokens,
            "thinking_budget": llm_thinking_budget,
            "rewriter_model": llm_rewriter_model,
        },
        "embedding": {"provider": emb_provider, "model": emb_model, "dim": emb_dim},
        "keys": {
            "GOOGLE_API_KEY": google_api_key,
            "OPENAI_API_KEY": openai_api_key,
            "ANTHROPIC_API_KEY": anthropic_api_key,
        },
        "qdrant": {"url": qdrant_url, "api_key": qdrant_api_key or None},
        "retrieval": {
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_k": rerank_top_k,
            "max_subqueries": max_subqueries,
        },
        "graph": {"expand_hops": graph_expand_hops, "max_neighbors": graph_max_neighbors},
    }

    if needs_reindex:
        patch["deployed"] = False

    try:
        s = save_settings(patch)
        from livedocs.query import code_pipeline
        code_pipeline.reset()
        ctx = _base_ctx(request)
        ctx.update({
            "s": s,
            "llm_suggestions": LLM_MODEL_SUGGESTIONS,
            "embedding_presets": EMBEDDING_PRESETS,
            "saved": True,
            "needs_reindex": needs_reindex,
            "error": None,
            "active": "settings",
        })
    except Exception as e:
        from livedocs.settings import get_settings
        ctx = _base_ctx(request)
        ctx.update({
            "s": get_settings(),
            "llm_suggestions": LLM_MODEL_SUGGESTIONS,
            "embedding_presets": EMBEDDING_PRESETS,
            "saved": False,
            "needs_reindex": False,
            "error": str(e),
            "active": "settings",
        })
    return _tr(request, "settings_form.html", ctx)


# ── sources ───────────────────────────────────────────────────────────────────

@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    from livedocs.settings import get_settings
    s = get_settings()
    ctx = _base_ctx(request)
    ctx.update({"sources": s.sources, "error": None, "active": "sources"})
    return _tr(request, "sources.html", ctx)


@router.post("/sources/github", response_class=HTMLResponse)
async def add_github_source(
    request: Request,
    name: str = Form(...),
    url: str = Form(...),
    corpus: str = Form("code"),
    branch: str = Form("main"),
    github_base: str = Form(""),
    token: str = Form(""),
):
    from livedocs.sources import add_github
    from livedocs.settings import get_settings
    try:
        add_github(name=name, url=url, corpus=corpus, branch=branch,
                   github_base=github_base, token=token)
    except Exception as e:
        ctx = _base_ctx(request)
        ctx.update({"sources": get_settings().sources, "error": str(e), "active": "sources"})
        return _tr(request, "sources.html", ctx)
    return RedirectResponse("/dashboard/sources", status_code=303)


@router.post("/sources/local", response_class=HTMLResponse)
async def add_local_source(
    request: Request,
    name: str = Form(...),
    path: str = Form(...),
    corpus: str = Form("code"),
):
    from livedocs.sources import add_local
    from livedocs.settings import get_settings
    try:
        add_local(name=name, path=path, corpus=corpus)
    except Exception as e:
        ctx = _base_ctx(request)
        ctx.update({"sources": get_settings().sources, "error": str(e), "active": "sources"})
        return _tr(request, "sources.html", ctx)
    return RedirectResponse("/dashboard/sources", status_code=303)


@router.post("/sources/upload", response_class=HTMLResponse)
async def upload_source(
    request: Request,
    name: str = Form(...),
    corpus: str = Form("docs"),
    file: UploadFile = File(...),
):
    from livedocs.sources import add_upload
    from livedocs.settings import get_settings
    try:
        contents = await file.read()
        add_upload(name=name, corpus=corpus, file_bytes=contents, filename=file.filename)
    except Exception as e:
        ctx = _base_ctx(request)
        ctx.update({"sources": get_settings().sources, "error": str(e), "active": "sources"})
        return _tr(request, "sources.html", ctx)
    return RedirectResponse("/dashboard/sources", status_code=303)


@router.post("/sources/{src_id}/delete", response_class=HTMLResponse)
async def delete_source(request: Request, src_id: str):
    from livedocs.sources import remove_source
    remove_source(src_id, delete_files=True)
    return RedirectResponse("/dashboard/sources", status_code=303)


# ── build ─────────────────────────────────────────────────────────────────────

@router.get("/build", response_class=HTMLResponse)
async def build_page(request: Request):
    from livedocs import jobs
    ctx = _base_ctx(request)
    ctx.update({"log_history": jobs.get_log_history(), "active": "build"})
    return _tr(request, "build.html", ctx)


@router.post("/build/start", response_class=HTMLResponse)
async def start_build(
    request: Request,
    corpus: str = Form("both"),
    resume: bool = Form(False),
):
    from livedocs import jobs
    try:
        jobs.start_build(corpus=corpus, resume=resume)
    except RuntimeError as e:
        ctx = _base_ctx(request)
        ctx.update({
            "log_history": jobs.get_log_history(),
            "start_error": str(e),
            "active": "build",
        })
        return _tr(request, "build.html", ctx)
    return RedirectResponse("/dashboard/build", status_code=303)


@router.get("/build/stream")
async def build_log_stream(request: Request):
    from livedocs import jobs

    def generate():
        for line in jobs.log_stream():
            if line:
                yield f"data: {line}\n\n"
            else:
                yield ": keepalive\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/build/status")
async def build_status_json(request: Request):
    from livedocs import jobs
    return jobs.get_status() or {}


# ── chat playground ───────────────────────────────────────────────────────────

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    ctx = _base_ctx(request)
    ctx["active"] = "chat"
    return _tr(request, "chat.html", ctx)
