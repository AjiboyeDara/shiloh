"""
One-command setup: fetches every text and resource, then builds the vector
indexes. Safe to re-run — completed steps are skipped (pass --rebuild to
force the index to be rebuilt).

    python scripts/setup.py

Steps:
  1. KJV text               (display + retrieval; public domain)
  2. BSB text               (modern-English retrieval mirror; public domain)
  3. Cross-references       (openbible.info, CC-BY)
  4. Strong's dictionaries  (OpenScriptures, CC-BY-SA)
  5. Vector indexes         (KJV + BSB; a few minutes of local embedding)

Chapter commentary (resources/commentary/) has no bundled fetcher yet; see
the README's "Adding commentary" section for the drop-in format.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_index
import download_bible
import fetch_cross_references
import fetch_strongs


def _optional(label, fn, done_path):
    """Resources are enrichment: a network failure warns and moves on
    instead of killing setup."""
    if os.path.exists(done_path):
        print(f"[{label}] already present, skipping.")
        return
    try:
        fn()
    except Exception as e:
        print(f"[{label}] failed ({e}); continuing. Re-run this script later to retry.")


def _index_ready():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=build_index.INDEX_DIR)
        names = {c.name for c in client.list_collections()}
        want = {"bible_kjv"}
        if os.path.exists(build_index.MODERN_PATH):
            want.add("bible_modern")
        return want <= names
    except Exception:
        return False


def main():
    download_bible.download()
    download_bible.normalize()
    download_bible.download_bsb()
    download_bible.normalize_bsb()

    _optional("cross-references", fetch_cross_references.main,
              fetch_cross_references.OUT_PATH)
    _optional("Strong's", fetch_strongs.main, fetch_strongs.OUT_PATH)

    if _index_ready() and "--rebuild" not in sys.argv:
        print("[index] already built, skipping (pass --rebuild to force).")
    else:
        build_index.build()

    print("\nSetup complete. Start the app with: uvicorn app.main:app")


if __name__ == "__main__":
    main()
