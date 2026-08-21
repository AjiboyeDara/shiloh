"""API tests with the LLM call mocked out."""
import os

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.retrieval import BSB_VERSES_PATH, VERSES_PATH

client = TestClient(app_main.app)

needs_data = pytest.mark.skipif(
    not os.path.exists(VERSES_PATH),
    reason="data/kjv_verses.json missing; run scripts/download_bible.py",
)

needs_bsb = pytest.mark.skipif(
    not os.path.exists(BSB_VERSES_PATH),
    reason="data/bsb_verses.json missing; run scripts/download_bible.py",
)

FAKE_PASSAGES = [{
    "reference": "John 3:16", "text": "For God so loved the world...",
    "book": "John", "chapter": 3, "verse_start": 16, "verse_end": 16,
    "cross_references": ["Romans 5:8"],
}]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_root_serves_landing():
    # Structural, not brand copy — renaming the app shouldn't turn CI red.
    res = client.get("/")
    assert res.status_code == 200
    assert "<html" in res.text.lower()
    assert '/app' in res.text  # the landing page's way into the chat UI


def test_app_serves_chat():
    res = client.get("/app")
    assert res.status_code == 200
    assert "Search the scriptures" in res.text


def test_chat_returns_answer_and_passages(monkeypatch):
    monkeypatch.setattr(
        app_main, "answer_question",
        lambda message, **kw: ("God loves the world [1].", FAKE_PASSAGES),
    )
    res = client.post("/api/chat", json={"message": "What does John 3:16 say?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"].endswith("[1].")
    assert body["passages"][0]["reference"] == "John 3:16"
    assert body["passages"][0]["cross_references"] == ["Romans 5:8"]


def test_chat_stream_emits_passages_deltas_done(monkeypatch):
    monkeypatch.setattr(
        app_main, "stream_answer",
        lambda message, **kw: (FAKE_PASSAGES, iter(["God ", "loves ", "[1]."]), []),
    )
    res = client.post("/api/chat/stream", json={"message": "hi"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    text = res.text
    assert "event: passages" in text
    assert '{"delta": "God "}' in text
    assert "event: done" in text


def test_chat_missing_key_rejected(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = client.post("/api/chat", json={"message": "hi", "provider": "anthropic"})
    assert res.status_code == 500
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_top_k_out_of_range_rejected():
    res = client.post("/api/chat", json={"message": "hi", "top_k": 50})
    assert res.status_code == 422


@needs_data
def test_chapter_endpoint():
    res = client.get("/api/chapter", params={"book": "John", "chapter": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["book"] == "John"
    assert body["verses"][15]["verse"] == 16
    assert "God so loved" in body["verses"][15]["text"]


def test_chapter_endpoint_404():
    res = client.get("/api/chapter", params={"book": "Nonexistent", "chapter": 1})
    assert res.status_code == 404


@needs_bsb
def test_passage_text_bsb_range():
    res = client.get("/api/passage-text", params={
        "translation": "bsb", "book": "John", "chapter": 3, "start": 16, "end": 17,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["translation"] == "bsb"
    assert [v["verse"] for v in body["verses"]] == [16, 17]
    # Modern wording, not the KJV's "only begotten Son".
    assert "begotten" not in body["verses"][0]["text"]


@needs_data
def test_passage_text_kjv_matches_chapter_slice():
    res = client.get("/api/passage-text", params={
        "book": "John", "chapter": 3, "start": 16, "end": 16,
    })
    assert res.status_code == 200
    chapter = client.get("/api/chapter", params={"book": "John", "chapter": 3}).json()
    assert res.json()["verses"] == [chapter["verses"][15]]


def test_passage_text_unknown_translation_422():
    res = client.get("/api/passage-text", params={
        "translation": "niv", "book": "John", "chapter": 3,
    })
    assert res.status_code == 422


def test_passage_text_404():
    res = client.get("/api/passage-text", params={"book": "Nonexistent", "chapter": 1})
    assert res.status_code == 404


def test_parse_origins():
    assert app_main._parse_origins("*") == ["*"]
    assert app_main._parse_origins("https://a.com, https://b.com,") == [
        "https://a.com", "https://b.com",
    ]


def _mock_answer(monkeypatch):
    monkeypatch.setattr(
        app_main, "answer_question",
        lambda message, **kw: ("An answer.", FAKE_PASSAGES),
    )


def test_rate_limit_disabled_by_default(monkeypatch):
    _mock_answer(monkeypatch)
    app_main._rate_buckets.clear()
    for _ in range(5):
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200


def test_basic_auth_disabled_by_default():
    assert client.get("/api/models").status_code == 200


def test_basic_auth_enforced(monkeypatch):
    import base64

    monkeypatch.setattr(app_main, "APP_PASSWORD", "sw0rdfish")
    res = client.get("/api/models")
    assert res.status_code == 401
    assert res.headers["www-authenticate"].startswith("Basic")
    # /health stays open for probes
    assert client.get("/health").status_code == 200
    # correct password (any username) gets through
    token = base64.b64encode(b"anyone:sw0rdfish").decode()
    assert client.get("/api/models", headers={"Authorization": f"Basic {token}"}).status_code == 200
    # wrong password stays out
    bad = base64.b64encode(b"anyone:wrong").decode()
    assert client.get("/api/models", headers={"Authorization": f"Basic {bad}"}).status_code == 401


needs_tagged = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(VERSES_PATH), "..",
                                    "resources", "strongs", "kjv_tagged.json")),
    reason="tagged KJV missing; run scripts/fetch_strongs_kjv.py",
)


@needs_tagged
def test_word_study_exact_with_verse_context():
    res = client.get("/api/word-study", params={
        "word": "loved", "book": "John", "chapter": 3, "start": 16, "end": 16,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["exact"] is True
    # John 3:16 "loved" is agapaō (G25), and it should be the top entry.
    assert body["strongs"][0]["number"] == "G25"


@needs_data
def test_word_study_without_context_not_exact():
    res = client.get("/api/word-study", params={"word": "loved"})
    assert res.status_code == 200
    assert res.json()["exact"] is False


def test_rate_limit_enforced(monkeypatch):
    _mock_answer(monkeypatch)
    monkeypatch.setattr(app_main, "RATE_LIMIT_PER_MINUTE", 2)
    app_main._rate_buckets.clear()
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200
    res = client.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 429
    assert "Rate limit" in res.json()["detail"]
    # Non-chat endpoints are not limited.
    assert client.get("/health").status_code == 200
    app_main._rate_buckets.clear()


def test_search_shares_the_chat_limit(monkeypatch):
    """Search runs the embedding model, so it spends from the chat budget."""
    monkeypatch.setattr(app_main, "retrieve", lambda query, top_k=6: FAKE_PASSAGES)
    monkeypatch.setattr(app_main, "RATE_LIMIT_PER_MINUTE", 1)
    app_main._rate_buckets.clear()
    assert client.post("/api/search", json={"query": "love"}).status_code == 200
    assert client.post("/api/search", json={"query": "love"}).status_code == 429
    app_main._rate_buckets.clear()


@needs_data
def test_word_study_has_its_own_roomier_bucket(monkeypatch):
    """Word study is click-driven, so it must not 429 at the chat limit."""
    monkeypatch.setattr(app_main, "retrieve", lambda query, top_k=6: FAKE_PASSAGES)
    monkeypatch.setattr(app_main, "RATE_LIMIT_PER_MINUTE", 1)
    app_main._rate_buckets.clear()
    # Spend the entire chat budget...
    assert client.post("/api/search", json={"query": "love"}).status_code == 200
    assert client.post("/api/search", json={"query": "love"}).status_code == 429
    # ...word study keeps answering, up to its own 6× cap.
    for _ in range(6):
        assert client.get("/api/word-study", params={"word": "loved"}).status_code == 200
    assert client.get("/api/word-study", params={"word": "loved"}).status_code == 429
    app_main._rate_buckets.clear()


# ── Reading plans ────────────────────────────────────────────────────────
def test_plan_returns_days_with_references(monkeypatch):
    monkeypatch.setattr(
        app_main, "plan_days",
        lambda theme, days=7, provider=None, model=None: (
            [{"title": "Bearing with one another", "references": ["Colossians 3:13"]}], None),
    )
    res = client.post("/api/plan", json={"theme": "forgiveness", "days": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["theme"] == "forgiveness"
    assert body["days"][0]["references"] == ["Colossians 3:13"]


def test_plan_refuses_harmful_theme():
    # The gate runs before any model call, so this needs no mocking.
    res = client.post("/api/plan", json={"theme": "how to make a bomb", "days": 3})
    assert res.status_code == 400


def test_plan_rejects_out_of_range_day_count():
    assert client.post("/api/plan", json={"theme": "grace", "days": 40}).status_code == 422


def test_plan_502_when_no_day_survives(monkeypatch):
    monkeypatch.setattr(
        app_main, "plan_days",
        lambda theme, days=7, provider=None, model=None: ([], None),
    )
    assert client.post("/api/plan", json={"theme": "grace", "days": 3}).status_code == 502


def test_plan_rejects_empty_and_oversized_themes():
    assert client.post("/api/plan", json={"theme": "", "days": 3}).status_code == 422
    assert client.post("/api/plan", json={"theme": "x" * 201, "days": 3}).status_code == 422


# ── Free-tier exhaustion is a routine event on a public deployment ───────
# The whole chain is under test here, not just the message: rag raises, the
# endpoint catches it as an unknown exception, and it reaches the browser as
# an SSE `error` event the frontend renders verbatim.
def test_daily_quota_reaches_the_browser_as_a_sentence(monkeypatch):
    from app import rag

    class _Quota429:
        status_code = 429
        text = '{"error":{"details":[{"quotaId":"GenerateRequestsPerDayPerProject"}]}}'

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(rag.requests, "post", lambda *a, **kw: _Quota429())
    monkeypatch.setattr(rag, "retrieve", lambda query, top_k=6: FAKE_PASSAGES)

    with client.stream("POST", "/api/chat/stream",
                       json={"message": "What is grace?", "provider": "gemini"}) as res:
        body = "".join(res.iter_text())

    assert "event: error" in body
    assert "used up for today" in body
    assert "429 Client Error" not in body   # no raw HTTP leaking through
