# Open Bible Study AI

An open-source, self-hostable AI chat assistant for Bible study. It answers
questions by retrieving relevant King James Version passages first (RAG),
then generating a grounded answer, with the source passages always shown
alongside the response so you can check the text yourself.

## How it works

```
question ──▶ embed query ──▶ search local vector index ──▶ top passages
                                                                 │
                                                                 ▼
                                          passages + question ──▶ LLM ──▶ answer
```

- **Bible text**: King James Version (public domain), sourced from a plain
  JSON mirror and normalized into `book/chapter/verse` records.
- **Retrieval**: hybrid. Verses are grouped into overlapping 5-verse
  windows, embedded locally with `sentence-transformers`
  (`all-MiniLM-L6-v2`, runs on CPU, no API key needed), and stored in a
  local Chroma vector index; a BM25 lexical index over the same chunks is
  fused in with reciprocal-rank fusion so exact KJV wording still matches.
  To close the archaic-vocabulary gap ("anxiety" vs "take no thought"),
  the Berean Standard Bible (public domain, modern English) is indexed in
  parallel as a retrieval-only mirror — search runs over both, the app
  always displays KJV. Fused results carry a light per-chapter diversity
  cap so thematic questions span the canon. Scripture references written
  in the question ("Romans 8", "John 3:16") are parsed out and always
  included first, and a small synonym map bridges the most common
  remaining gaps ("Holy Spirit" → "Holy Ghost"). No Bible text ever
  leaves your machine during retrieval. Retrieval quality is measured by
  a golden-set eval (`scripts/eval_retrieval.py`) — run it before and
  after any retrieval change.
- **Generation**: retrieved passages + your question go to an LLM to produce
  the answer, streamed token by token to the UI (`/api/chat/stream`), with
  numbered `[n]` citations that link back to the retrieved passages.
  Providers are selected via `LLM_PROVIDER` in `.env`: a free local model
  via [Ollama](https://ollama.com) (the default, with no API key so nothing
  leaves your machine), Google Gemini, or the Anthropic API. The generation
  calls are isolated in `app/rag.py` so adding another provider is easy.
- **Quote verification**: every scripture quotation in an answer is
  checked word-for-word against the KJV (`app/verify.py`). Verbatim quotes
  are marked verified with their verse reference; near-quotes are flagged
  with the real wording; inventions are called out. Verse mentions in
  answers ("v. 26") are clickable and open the chapter at that verse.
- **Word study**: click any word in a retrieved passage to see the Strong's
  entries behind it (original Hebrew/Greek, transliteration, definition,
  how the KJV renders it) plus a concordance of everywhere it occurs.
- **Extras**: cross-references and Strong's dictionaries are fetched by
  `scripts/setup.py` out of the box; chapter commentary is supported as an
  optional local resource file that the app picks up automatically.

## Quickstart (local)

```bash
git clone <this-repo>
cd bible-study-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Default is a free local model via Ollama: install it from https://ollama.com
# (or `brew install ollama`), start it (`ollama serve`), then:
ollama pull llama3.2
# To use Claude instead, set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY in .env.

# One command: fetches the KJV + BSB texts, cross-references, and Strong's
# dictionaries, then builds the local embedding indexes (a few minutes).
python scripts/setup.py

uvicorn app.main:app --reload
```

Open http://localhost:8000 to use the chat UI, which is served directly by the API.

## Quickstart (Docker)

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker compose up --build
```

The image downloads the Bible text and builds the index at build time, so
the container is ready to serve as soon as it starts.

## Cross-references

Verse cross-references (openbible.info's dataset, derived from the
public-domain Treasury of Scripture Knowledge, CC-BY) are fetched and
converted with one command:

```bash
python scripts/fetch_cross_references.py
```

After that, every retrieved passage in the UI shows "See also" chips that
open the referenced chapter inline.

## Adding commentary

Chapter commentary is optional and off by default — it's the one resource
without a bundled fetch script (a reliably machine-readable public-domain
source is still wanted; writing that fetcher is a great first
contribution). Drop in a file and the app picks it up automatically:

| Resource | Path | Format |
|---|---|---|
| Commentary | `resources/commentary/<Book>.json` | `{"1": "commentary for chapter 1", "2": "...", ...}` |

Good public-domain source: Matthew Henry's Concise Commentary, available
via [ccel.org](https://ccel.org).

Strong's dictionaries are already fetched by `scripts/setup.py` (or
directly via `scripts/fetch_strongs.py`) from
[OpenScriptures](https://github.com/openscriptures/strongs) (CC-BY-SA) and
power the click-a-word study panel.

## Project structure

```
app/
  main.py         FastAPI app + routes (/api/chat, /api/chat/stream, /api/search, /api/chapter, /api/word-study)
  rag.py          Prompting + LLM calls (blocking + streaming)
  retrieval.py    Hybrid search, reference parsing, resource lookups
  verify.py       Word-for-word quote verification against the KJV
  word_study.py   Strong's lookups + concordance for a KJV word
  models.py       Request/response schemas
scripts/
  setup.py                   One-command setup (runs everything below)
  download_bible.py          Fetches + normalizes the KJV and BSB texts
  build_index.py             Chunks, embeds, and indexes both translations
  fetch_cross_references.py  Fetches + converts cross-reference data
  fetch_strongs.py           Fetches + merges the Strong's dictionaries
  eval_retrieval.py          Golden-set retrieval eval (recall@k, hit@k)
  golden_set.json            ~30 thematic questions with expected passages
frontend/
  index.html      Single-file chat UI (no build step)
tests/            pytest suite (retrieval, chunking, verification, API)
resources/        Cross-refs + Strong's + optional commentary
data/             Generated at setup time (gitignored)
```

Run the tests with `python -m pytest`.

## Swapping the LLM

Set `LLM_PROVIDER` in `.env` to `ollama` (default; free local model, configure
with `OLLAMA_MODEL`/`OLLAMA_URL`) or `anthropic` (requires `ANTHROPIC_API_KEY`,
configure the model with `CHAT_MODEL`). Each provider is a single small
function in `app/rag.py` (`_generate_ollama` / `_generate_anthropic`), so
adding another is a few lines, since the retrieval and prompt-building logic is
provider-agnostic.

## Swapping the translation

The default is KJV because it's unambiguously public domain. To use a
different public-domain translation (ASV, WEB, Douay-Rheims, etc.), point
`RAW_URL` in `scripts/download_bible.py` at a JSON source for that
translation and adjust `normalize()` if the shape differs. Do **not** use a
copyrighted modern translation (NIV, ESV, NLT, etc.) without a license from
its publisher.

## Self-hosting

Two env vars harden a publicly exposed instance (both optional; defaults
keep localhost development frictionless):

- `CORS_ORIGINS` — comma-separated list of allowed browser origins
  (default `*`). Set it to your site, e.g.
  `CORS_ORIGINS=https://bible.example.com`.
- `CHAT_RATE_LIMIT` — per-IP requests per minute on the two chat endpoints
  (default `0` = disabled). `CHAT_RATE_LIMIT=10` is a sane public setting;
  over the limit returns HTTP 429. If the app sits behind a reverse proxy,
  also set `TRUST_PROXY=1` so the limit keys on `X-Forwarded-For` instead
  of the proxy's own address.

## Roadmap ideas

- Multi-translation comparison view
- Reading plans / study guides generated from a theme
- A Strong's-tagged KJV so word study resolves the exact original word per
  verse (today it lists candidates via the dictionaries' KJV renderings)
- Commentary fetch script (Matthew Henry) for the existing commentary hook
- A stronger embedding model, measured with `scripts/eval_retrieval.py`.
  `EMBED_MODEL` and an optional cross-encoder rerank stage (`RERANK_MODEL`)
  are env-configurable to make experiments cheap. Tried so far (2026-07),
  all at or below the `all-MiniLM-L6-v2` + RRF baseline on the golden set:
  `bge-small-en-v1.5` (with and without query prefix; re-tested against the
  expanded 58-question set and larger synonym map — still lost, recall@6
  0.655 vs 0.665, hit@6 0.931 vs 0.966), reranking with
  `ms-marco-MiniLM-L-6-v2` and `bge-reranker-base` — modern-English models
  seem to misjudge KJV text, so a candidate should likely be fine-tuned or
  chosen for archaic English

## License

Code: MIT (see `LICENSE`). The KJV Bible text is public domain. Any
additional resources you add carry whatever license their source specifies.
