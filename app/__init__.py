"""Load .env before anything else in the package reads a setting.

app.rag and app.retrieval read provider, model, and tuning settings from the
environment at import time, so .env has to be loaded before those imports —
not just by the web app. Doing it here covers every entry point (uvicorn, the
scripts/ tools, pytest) instead of relying on each one to remember. Without
it, `python scripts/eval_answers.py` silently ignored GEMINI_API_KEY and let
the answering model grade its own output, and scripts/build_index.py would
build the index with a different EMBED_MODEL than the app queries it with.
"""
from dotenv import load_dotenv

load_dotenv()
