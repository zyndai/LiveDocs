"""HTTP wrapper around the Haystack pipeline.

POST /ask  body={"question": "..."}  -> {"answer": "...", "sources": [...]}

Same contract as the main pipeline's app.py, but on port 8002 so both can
run side-by-side.

Run from inside the ragpipline folder:
    uvicorn app:app --port 8002
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import ask, build_pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm-load the Haystack pipeline so first request doesn't pay the model-load cost.
    print("Warming up Haystack pipeline (bge-m3 + SPLADE + reranker) ...")
    build_pipeline()
    print("Ready.")
    yield


app = FastAPI(title="Zynd Docs RAG (Haystack)", lifespan=lifespan)

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
    question: str = Field(min_length=1, max_length=1000)
    # Optional chat history. Most recent turn last. Omit on the first question.
    history: list[Message] = Field(default_factory=list)


class Source(BaseModel):
    source: str | None
    heading: str | None
    subheading: str | None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    # The standalone question used for retrieval. Same as `question` if no rewriting was needed.
    retrieval_query: str


@app.get("/health")
def health():
    return {"status": "ok", "impl": "haystack"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        _retrieved, reranked, answer, retrieval_query = ask(req.question, history=history)
    except Exception as e:
        print(f"!! /ask failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="generation failed")

    sources = [
        Source(
            source=(d.meta or {}).get("source"),
            heading=(d.meta or {}).get("heading"),
            subheading=(d.meta or {}).get("subheading"),
            score=float(d.score),
        )
        for d in reranked
    ]

    return AskResponse(answer=answer, sources=sources, retrieval_query=retrieval_query)
