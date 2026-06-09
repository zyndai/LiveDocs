"""FastAPI entrypoint.

Endpoints:
  GET  /health
  POST /ask/stream   SSE streaming (text/event-stream)
  POST /ask          Blocking JSON response
  /dashboard/*       HTMX admin UI (settings, sources, build, chat)
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from livedocs.config import PROJECT_ROOT
from livedocs.query.code_pipeline import ask_stream, warm_up


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from livedocs.settings import get_settings
    s = get_settings()
    if s.deployed:
        print("Warming up RAG pipeline ...")
        try:
            warm_up()
            print("Ready.")
        except Exception as e:
            print(f"  (warm-up failed: {e} — queries will fail until a build completes)")
    else:
        print("Not deployed yet. Visit /dashboard to configure and build.")
    yield


app = FastAPI(title="LiveDocs RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Static files (CSS, JS)
_static_dir = PROJECT_ROOT / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Dashboard router
from livedocs.dashboard.router import router as dashboard_router
app.include_router(dashboard_router)


# ── Request/response models ────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    sources: list
    related: list
    sub_queries: list[str]


# ── Guards ────────────────────────────────────────────────────────────────────

def _require_deployed():
    from livedocs.settings import get_settings
    if not get_settings().deployed:
        raise HTTPException(
            status_code=503,
            detail="Not deployed — run a build in the dashboard first: /dashboard/build",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from livedocs.settings import get_settings
    s = get_settings()
    return {
        "status": "ok",
        "deployed": s.deployed,
        "llm": s.llm.provider,
        "embedding": s.embedding.provider,
    }


@app.post("/ask/stream")
async def ask_stream_endpoint(req: AskRequest):
    _require_deployed()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    def generate():
        try:
            for event_type, data in ask_stream(req.question, history=history):
                if event_type == "token":
                    yield _sse("token", json.dumps(data))
                elif event_type == "meta":
                    yield _sse("meta", json.dumps(data))
                elif event_type == "error":
                    yield _sse("error", json.dumps({"detail": data}))
                    return
            yield _sse("done", "")
        except Exception as e:
            yield _sse("error", json.dumps({"detail": f"{type(e).__name__}: {e}"}))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    """Non-streaming endpoint. Returns full answer + sources in one JSON response."""
    _require_deployed()
    from livedocs.query.code_pipeline import ask
    from livedocs.query.app_utils import docs_to_source_dicts

    history = [{"role": m.role, "content": m.content} for m in req.history]
    try:
        reranked, related, answer, sub_queries = ask(req.question, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return AskResponse(
        answer=answer,
        sources=docs_to_source_dicts(reranked, with_score=True),
        related=docs_to_source_dicts(related, with_score=False),
        sub_queries=sub_queries,
    )
