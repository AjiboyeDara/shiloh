"""
Generation layer: takes a user question + retrieved passages and produces
a study-oriented answer via an LLM. Supports three providers, selected by
the LLM_PROVIDER env var:
  - "ollama" (default): a local model served by Ollama, free, no API key.
  - "gemini": the Google Gemini API (requires GEMINI_API_KEY; has a free tier).
  - "anthropic": the Anthropic API (requires ANTHROPIC_API_KEY).
"""
import json
import os

import requests

from app.retrieval import load_commentary, retrieve

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
MODEL = os.environ.get("CHAT_MODEL", "claude-sonnet-4-5")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are a Bible study assistant. You help people understand \
scripture more deeply. You are given retrieved Bible passages (King James \
Version) relevant to the user's question, and sometimes chapter commentary.

Guidelines:
- Ground your answer in the provided passages. The passages are numbered; \
cite them inline with the bracketed number, e.g. [1] or [2], placed right \
after the claim it supports. You may also name the reference \
(Book Chapter:Verse) in prose, but always include the bracketed number too \
so the citation can be linked to the passage. Quote sparingly.
- If the passages don't fully answer the question, say so plainly rather \
than filling gaps with speculation.
- Offer historical/literary context when it aids understanding (authorship, \
audience, genre) but distinguish that from the text itself.
- Different Christian traditions interpret many passages differently. \
Where a question touches a doctrinally contested point, present the \
Bible's material and note that interpretations vary across traditions, \
rather than asserting one reading as the only correct one.
- Keep answers focused and readable. This is for personal study, not a \
sermon.
"""


def build_context(passages, commentary_snippets=None):
    parts = []
    for i, p in enumerate(passages, start=1):
        parts.append(f"[{i}] {p['reference']}\n{p['text']}")
    context = "\n\n".join(parts)
    if commentary_snippets:
        context += "\n\nCommentary notes:\n" + "\n\n".join(commentary_snippets)
    return context


def _prepare(message: str, history=None, top_k: int = 6):
    """Shared retrieval + prompt assembly for both the blocking and
    streaming paths. Returns (messages, passages)."""
    passages = retrieve(message, top_k=top_k)

    # Pull commentary for chapters that showed up in retrieval, if the user
    # has populated resources/commentary/.
    seen_chapters = set()
    commentary_snippets = []
    for p in passages:
        key = (p["book"], p["chapter"])
        if key in seen_chapters:
            continue
        seen_chapters.add(key)
        note = load_commentary(p["book"], p["chapter"])
        if note:
            commentary_snippets.append(f"[{p['book']} {p['chapter']}] {note}")

    context = build_context(passages, commentary_snippets)

    messages = []
    for turn in (history or []):
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({
        "role": "user",
        "content": (
            f"Retrieved passages:\n{context}\n\n"
            f"User question: {message}"
        ),
    })
    return messages, passages


def answer_question(message: str, history=None, top_k: int = 6,
                    provider: str = None, model: str = None):
    messages, passages = _prepare(message, history=history, top_k=top_k)

    provider = (provider or PROVIDER).lower()
    if provider == "ollama":
        answer_text = _generate_ollama(messages, model or OLLAMA_MODEL)
    elif provider == "gemini":
        answer_text = _generate_gemini(messages, model or GEMINI_MODEL)
    else:
        answer_text = _generate_anthropic(messages, model or MODEL)

    return answer_text, passages


def stream_answer(message: str, history=None, top_k: int = 6,
                  provider: str = None, model: str = None):
    """Like answer_question, but returns (passages, generator) where the
    generator yields the answer text incrementally."""
    messages, passages = _prepare(message, history=history, top_k=top_k)

    provider = (provider or PROVIDER).lower()
    if provider == "ollama":
        gen = _stream_ollama(messages, model or OLLAMA_MODEL)
    elif provider == "gemini":
        gen = _stream_gemini(messages, model or GEMINI_MODEL)
    else:
        gen = _stream_anthropic(messages, model or MODEL)
    return passages, gen


def _generate_gemini(messages, model):
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
            ],
            "generationConfig": {"maxOutputTokens": 1200},
        },
        timeout=120,
    )
    response.raise_for_status()
    candidate = response.json()["candidates"][0]
    return "".join(
        part.get("text", "") for part in candidate["content"]["parts"]
    )


def _generate_ollama(messages, model):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
            "options": {"num_predict": 1200},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _generate_anthropic(messages, model):
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def _stream_ollama(messages, model):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": True,
            "options": {"num_predict": 1200},
        },
        timeout=300,
        stream=True,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        chunk = data.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if data.get("done"):
            break


def _stream_gemini(messages, model):
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
            ],
            "generationConfig": {"maxOutputTokens": 1200},
        },
        timeout=120,
        stream=True,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if payload == b"[DONE]":
            break
        data = json.loads(payload)
        for candidate in data.get("candidates", [])[:1]:
            for part in candidate.get("content", {}).get("parts", []):
                if part.get("text"):
                    yield part["text"]


def _stream_anthropic(messages, model):
    from anthropic import Anthropic

    client = Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        yield from stream.text_stream
