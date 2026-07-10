---
name: verify
description: Build/launch/drive recipe for verifying Shiloh (FastAPI + single-file frontend) end to end.
---

# Verifying Shiloh

## Launch

```bash
.venv/bin/uvicorn app.main:app --port 8901   # serves API + frontend at /
```

Needs `data/kjv_verses.json` + `data/chroma_index/` (run `scripts/setup.py` once —
it fetches KJV + BSB texts, cross-references, Strong's, and builds both indexes). Default LLM provider is local Ollama —
check it's up with `curl -s localhost:11434/api/tags`. First request loads the
embedding model (~5s) and builds the BM25 index lazily.

## API surface

- `POST /api/chat/stream` — SSE: `event: passages`, then `data: {"delta":...}`,
  then `event: done` (errors mid-stream come as `event: error`). Use `curl -sN`.
- `POST /api/chat` — blocking JSON. `POST /api/search`, `GET /api/chapter?book=&chapter=`,
  `GET /api/models`, `GET /health`.
- Good probes: `top_k: 50` → 422; `provider: "anthropic"` with no key → 500 naming
  the env var; unreachable Ollama (`OLLAMA_URL=http://localhost:19999` on a second
  port) → 503 on /api/chat, `event: error` on the stream.

## Browser drive (frontend)

Playwright against system Chrome — no browser download:

```bash
cd "$SCRATCHPAD" && npm i playwright   # then in the script:
# chromium.launch({ channel: 'chrome', headless: true })
```

Flows worth driving: submit a question (thinking indicator → streamed answer →
`[n]` citation sups → passage cards + "See also" xref chips); click citation
(card `.lit`); "Read full chapter" panel + xref chip opens referenced chapter;
reload (localStorage persistence restores transcript); `#newChat` resets;
390px viewport → passages become a bottom sheet (`#sheetFab` / `#sheetClose`,
citation tap opens it). Await `#sendBtn` text back to "Ask" to detect
end-of-stream.

## Gotchas

- Ollama answers take 10–60s on llama3.2; use generous Playwright timeouts.
- `.env` sets real provider keys; don't echo it.
- `tests/` exist (`python -m pytest`) but verification means driving the app,
  not rerunning them.
