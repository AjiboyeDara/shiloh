import json
import os

from dotenv import load_dotenv

load_dotenv()  # must run before app.rag reads provider/model settings at import

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    ChapterResponse,
    ChapterVerse,
    ChatRequest,
    ChatResponse,
    PassageResult,
    SearchRequest,
    SearchResponse,
)
import requests

from app.rag import (
    GEMINI_MODEL,
    MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROVIDER,
    answer_question,
    stream_answer,
)
from app.retrieval import get_chapter, retrieve

OLLAMA_DOWN_MSG = (
    "Couldn't reach the local Ollama server. Start it with `ollama serve` "
    "(or switch to another provider in the model picker)."
)


def _check_provider_key(provider: str):
    required_key = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider)
    if required_key and not os.environ.get(required_key):
        raise HTTPException(
            status_code=500,
            detail=f"{required_key} is not set. Add it to your .env file.",
        )

app = FastAPI(title="Open Bible Study AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Bible Study AI API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def list_models():
    """Providers/models the user can pick from in the UI. A provider is
    included only if it's usable right now (key set / server reachable)."""
    providers = []

    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
        local_models = sorted(m["name"].removesuffix(":latest") for m in tags.get("models", []))
        if local_models:
            providers.append({
                "id": "ollama", "label": "Local (Ollama)",
                "models": local_models,
                "default_model": OLLAMA_MODEL if OLLAMA_MODEL in local_models else local_models[0],
            })
    except Exception:
        pass

    if os.environ.get("GEMINI_API_KEY"):
        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
        if GEMINI_MODEL not in models:
            models.insert(0, GEMINI_MODEL)
        providers.append({
            "id": "gemini", "label": "Google Gemini",
            "models": models, "default_model": GEMINI_MODEL,
        })

    if os.environ.get("ANTHROPIC_API_KEY"):
        models = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-8"]
        if MODEL not in models:
            models.insert(0, MODEL)
        providers.append({
            "id": "anthropic", "label": "Anthropic Claude",
            "models": models, "default_model": MODEL,
        })

    return {"providers": providers, "default_provider": PROVIDER}


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        results = retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search index not available yet. Run scripts/build_index.py first. ({e})",
        )
    return SearchResponse(results=[PassageResult(**r) for r in results])


@app.get("/api/chapter", response_model=ChapterResponse)
def chapter(book: str, chapter: int):
    verses = get_chapter(book, chapter)
    if not verses:
        raise HTTPException(
            status_code=404,
            detail=f"No verses found for {book} {chapter}.",
        )
    return ChapterResponse(
        book=book,
        chapter=chapter,
        verses=[ChapterVerse(**v) for v in verses],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    provider = (req.provider or PROVIDER).lower()
    _check_provider_key(provider)
    try:
        answer_text, passages = answer_question(
            req.message, history=req.history, top_k=req.top_k,
            provider=provider, model=req.model,
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=OLLAMA_DOWN_MSG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        answer=answer_text,
        passages=[PassageResult(**p) for p in passages],
    )


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-sent events: one `passages` event, then `data` deltas of the
    answer text, then a `done` event. Errors mid-stream arrive as an
    `error` event since the 200 status is already committed."""
    provider = (req.provider or PROVIDER).lower()
    _check_provider_key(provider)

    def sse(event, payload):
        prefix = f"event: {event}\n" if event else ""
        return f"{prefix}data: {json.dumps(payload)}\n\n"

    def event_stream():
        try:
            passages, deltas = stream_answer(
                req.message, history=req.history, top_k=req.top_k,
                provider=provider, model=req.model,
            )
            yield sse("passages", {"passages": passages})
            for chunk in deltas:
                yield sse(None, {"delta": chunk})
            yield sse("done", {})
        except requests.exceptions.ConnectionError:
            yield sse("error", {"message": OLLAMA_DOWN_MSG})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
