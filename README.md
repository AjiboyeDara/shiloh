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
- **Retrieval**: verses are grouped into overlapping 5-verse windows,
  embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`, runs on
  CPU, no API key needed), and stored in a local Chroma vector index. No
  Bible text ever leaves your machine during retrieval.
- **Generation**: retrieved passages + your question go to an LLM to produce
  the answer. Two providers are supported out of the box, selected via
  `LLM_PROVIDER` in `.env`: a free local model via [Ollama](https://ollama.com)
  (the default, with no API key so nothing leaves your machine) or the Anthropic
  API. The generation call is isolated in `app/rag.py` so adding another
  provider is easy.
- **Extras**: cross-references, chapter commentary, and Strong's word
  definitions are supported as optional local resource files (see below).
  None are bundled by default since their best public-domain sources aren't
  reliably mirrored on a single site, but wiring them in is a couple of
  JSON files, and `app/retrieval.py` already knows how to read them.

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

python scripts/download_bible.py   # fetches KJV text (~5MB, one-time)
python scripts/build_index.py      # builds the local embedding index (a few minutes)

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

## Adding cross-references, commentary, or Strong's data

These are optional and off by default. Drop in a file at any of these paths
and the app will pick it up automatically, with no code changes needed:

| Resource | Path | Format |
|---|---|---|
| Cross-references | `resources/cross_references/cross_references.json` | `{"John 3:16": ["Romans 5:8", "1 John 4:9"], ...}` |
| Commentary | `resources/commentary/<Book>.json` | `{"1": "commentary for chapter 1", "2": "...", ...}` |
| Strong's numbers | `resources/strongs/strongs.json` | `{"G26": "agape - love...", "H430": "Elohim - God...", ...}` |

Good public-domain sources to build these from:
- **Cross-references**: [openbible.info/labs/cross-references](https://www.openbible.info/labs/cross-references/)
  (derived from the Treasury of Scripture Knowledge) or the
  [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases)
  GitHub repo, which bundles the same data in several formats.
- **Commentary**: Matthew Henry's Concise Commentary, available via
  [ccel.org](https://ccel.org) or bundled in the scrollmapper repo above.
- **Strong's**: [OpenScriptures Strong's dictionary data](https://github.com/openscriptures/strongs)
  (GitHub, public domain).

A short conversion script that reshapes any of these into the formats above
is a natural first contribution if you want to help extend this project.

## Project structure

```
app/
  main.py         FastAPI app + routes (/api/chat, /api/search)
  rag.py          Prompting + LLM call
  retrieval.py    Embedding search + optional resource lookups
  models.py       Request/response schemas
scripts/
  download_bible.py   Fetches + normalizes the KJV text
  build_index.py      Chunks, embeds, and indexes it
frontend/
  index.html      Single-file chat UI (no build step)
resources/        Optional user-supplied cross-refs/commentary/Strong's
data/             Generated at setup time (gitignored)
```

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

## Roadmap ideas

- Verse-level cross-reference display in the sidebar once resources are loaded
- Multi-translation comparison view
- Reading plans / study guides generated from a theme
- Original-language word study mode (Strong's + morphology)

## License

Code: MIT (see `LICENSE`). The KJV Bible text is public domain. Any
additional resources you add carry whatever license their source specifies.
