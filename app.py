"""HTTP wrapper — streaming SSE endpoint.

POST /ask/stream  body={"question": "...", "history": [...]}
  -> text/event-stream

  event: token   data: "partial answer text"
  event: token   data: "..."
  ...
  event: meta    data: {"sources":[...], "related":[...], "sub_queries":[...]}
  event: done    data: ""

GET /health  -> {"status":"ok"}

Run:
    uvicorn app:app --port 8002
"""
import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from code_pipeline import ask_stream, warm_up


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Warming up RAG (bge-m3 + SPLADE + reranker + graph) ...")
    warm_up()
    print("Ready.")
    yield


app = FastAPI(title="Code+Docs RAG — streaming", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "impl": "code-docs-rag-stream"}


@app.post("/ask/stream")
async def ask_stream_endpoint(req: AskRequest):
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
