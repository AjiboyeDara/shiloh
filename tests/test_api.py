"""API tests with the LLM call mocked out."""
import os

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.retrieval import VERSES_PATH

client = TestClient(app_main.app)

needs_data = pytest.mark.skipif(
    not os.path.exists(VERSES_PATH),
    reason="data/kjv_verses.json missing; run scripts/download_bible.py",
)

FAKE_PASSAGES = [{
    "reference": "John 3:16", "text": "For God so loved the world...",
    "book": "John", "chapter": 3, "verse_start": 16, "verse_end": 16,
    "cross_references": ["Romans 5:8"],
}]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


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
