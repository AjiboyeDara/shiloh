import base64
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # must run before app.rag reads provider/model settings at import

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    ChapterResponse,
    ChapterVerse,
    ChatRequest,
    ChatResponse,
    PassageResult,
    PassageTextResponse,
    SearchRequest,
    SearchResponse,
    WordStudyResponse,
)
import requests

from app.rag import (
    GEMINI_MODEL,
    MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROVIDER,
    answer_question,
    repair_quotes,
    stream_answer,
)
from app.retrieval import (
    canonical_book,
    get_chapter,
    get_passage_text,
    has_bsb,
    retrieve,
)
from app.verify import verify_quotes
from app.word_study import word_study

OLLAMA_DOWN_MSG = (
    "Couldn't reach the local Ollama server. Start it with `ollama serve` "
    "(or switch to another provider in the model picker)."
)


def _safe_verify(answer_text, passages):
    """Quote verification is an annotation, never a reason to fail a reply."""
    try:
        return verify_quotes(answer_text, passages)
    except Exception:
        return []


def _check_provider_key(provider: str):
    required_key = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider)
    if required_key and not os.environ.get(required_key):
        raise HTTPException(
            status_code=500,
            detail=f"{required_key} is not set. Add it to your .env file.",
        )

def _parse_origins(raw: str):
    return [o.strip() for o in raw.split(",") if o.strip()]


# Per-IP rate limit on the chat endpoints (they're the ones that spend LLM
# calls). Disabled by default so localhost dev is unaffected; set
# CHAT_RATE_LIMIT to a requests-per-minute cap when hosting publicly.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("CHAT_RATE_LIMIT", 0))
TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"
_rate_buckets: dict = defaultdict(deque)


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request):
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    now = time.monotonic()
    ip = _client_ip(request)
    bucket = _rate_buckets[ip]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded; please wait a moment and try again.",
        )
    bucket.append(now)
    # Drop other IPs' stale buckets so the dict can't grow unboundedly.
    for other in [k for k, v in _rate_buckets.items() if v and now - v[-1] > 60]:
        del _rate_buckets[other]


def _warm_retrieval():
    """The first query normally pays ~5s of embedding-model load plus the
    lazy BM25 build; doing it at startup makes the first question as fast
    as the rest. Failures are fine — the lazy path still works."""
    try:
        from app.retrieval import _bm25_index, get_embedder
        get_embedder().encode(["warm"])
        _bm25_index()
    except Exception:
        pass


@asynccontextmanager
async def _lifespan(app):
    if os.environ.get("WARM_ON_STARTUP", "1") != "0":
        threading.Thread(target=_warm_retrieval, daemon=True).start()
    yield


app = FastAPI(title="Open Bible Study AI", lifespan=_lifespan)

# Optional password gate for public hosting: when APP_PASSWORD is set,
# everything but /health requires HTTP Basic auth (any username). The
# browser's native prompt handles the frontend; fetch() then reuses the
# cached credentials automatically.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


@app.middleware("http")
async def _basic_auth(request: Request, call_next):
    if APP_PASSWORD and request.url.path != "/health":
        supplied = ""
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                supplied = base64.b64decode(header[6:]).decode().partition(":")[2]
            except Exception:
                supplied = ""
        if not secrets.compare_digest(supplied, APP_PASSWORD):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Shiloh"'},
            )
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(os.environ.get("CORS_ORIGINS", "*")),
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

MEDIA_DIR = os.path.join(os.path.dirname(__file__), "..", "media")
if os.path.isdir(MEDIA_DIR):
    app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")


@app.get("/")
def root():
    """Landing page first; the chat app lives at /app."""
    landing_path = os.path.join(FRONTEND_DIR, "landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path)
    return chat_app()


@app.get("/app")
def chat_app():
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
        book=canonical_book(book),
        chapter=chapter,
        verses=[ChapterVerse(**v) for v in verses],
    )


@app.get("/api/passage-text", response_model=PassageTextResponse)
def passage_text(book: str, chapter: int, translation: str = "kjv",
                 start: int | None = None, end: int | None = None):
    """One passage's verses in a given translation (whole chapter when no
    range). BSB is display-on-demand: the chat/search payloads stay KJV."""
    translation = translation.lower()
    if translation not in ("kjv", "bsb"):
        raise HTTPException(status_code=422, detail="translation must be 'kjv' or 'bsb'.")
    if translation == "bsb" and not has_bsb():
        raise HTTPException(
            status_code=404,
            detail="BSB text not downloaded; run scripts/download_bible.py.",
        )
    verses = get_passage_text(translation, book, chapter, start, end)
    if not verses:
        ref = f"{book} {chapter}" + (f":{start}-{end}" if start else "")
        raise HTTPException(status_code=404, detail=f"No verses found for {ref}.")
    return PassageTextResponse(
        translation=translation,
        book=canonical_book(book),
        chapter=chapter,
        verse_start=start,
        verse_end=end,
        verses=[ChapterVerse(**v) for v in verses],
    )


@app.get("/api/word-study", response_model=WordStudyResponse)
def word_study_endpoint(word: str, book: str | None = None,
                        chapter: int | None = None,
                        start: int | None = None, end: int | None = None):
    """Strong's entries + KJV concordance for one English word. With a
    verse context (book/chapter/start/end) and the tagged KJV downloaded,
    the entries are the exact numbers tagged on the word there."""
    word = word.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]{0,23}", word):
        raise HTTPException(
            status_code=422,
            detail="Provide a single English word (letters only).",
        )
    return word_study(word, book=book, chapter=chapter,
                      verse_start=start, verse_end=end)


@app.post("/api/chat", response_model=ChatResponse,
          dependencies=[Depends(rate_limit)])
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
        quote_checks=_safe_verify(answer_text, passages),
    )


@app.post("/api/chat/stream", dependencies=[Depends(rate_limit)])
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
            passages, deltas, messages = stream_answer(
                req.message, history=req.history, top_k=req.top_k,
                provider=provider, model=req.model,
            )
            yield sse("passages", {"passages": passages})
            answer_parts = []
            for chunk in deltas:
                answer_parts.append(chunk)
                yield sse(None, {"delta": chunk})
            answer_text = "".join(answer_parts)
            # If the answer misquotes scripture, run one repair pass and
            # replace the streamed text client-side via a `revision` event.
            revised = repair_quotes(messages, answer_text, passages,
                                    provider=provider, model=req.model)
            if revised != answer_text:
                answer_text = revised
                yield sse("revision", {"answer": answer_text})
            checks = _safe_verify(answer_text, passages)
            if checks:
                yield sse("quotes", {"quotes": checks})
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
