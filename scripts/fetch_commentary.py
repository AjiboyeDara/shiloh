"""
Downloads Matthew Henry's commentary (public domain, via the free-use
HelloAO Bible API) into the per-book format app/retrieval.py's
load_commentary already reads:

    resources/commentary/<Book>.json
    { "1": "commentary text for chapter 1", "2": "...", ... }

Each chapter's text is Matthew Henry's chapter introduction — a compact
overview that fits a prompt; the full verse-by-verse exposition would blow
the context budget. Chapters with no introduction fall back to the opening
of the first verse section.

Already-fetched books are skipped so an interrupted run resumes; pass
--force to re-download everything.
"""
import argparse
import json
import os
import time

import requests

API = "https://bible.helloao.org/api/c/matthew-henry"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "commentary")

MAX_CHARS = 1500  # keep each chapter note prompt-sized


def _trim(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS].rsplit(" ", 1)[0] + "…"


def _chapter_note(payload: dict) -> str:
    chapter = payload.get("chapter", {})
    intro = (chapter.get("introduction") or "").strip()
    if intro:
        return _trim(intro)
    for section in chapter.get("content", []):
        for para in section.get("content", []):
            if para.strip():
                return _trim(para)
    return ""


def _get(session, url, retries=3, required=False):
    """None (instead of raising) for chapters that return junk — a handful
    of chapters are missing upstream and come back as non-JSON."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                if required:
                    raise
                print(f"    skipping {url.rsplit('/', 2)[-2:]}: no usable data")
                return None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--force", action="store_true",
                    help="re-download books that already have a file")
    # parse_known_args so setup.py can call this with its own flags present
    args = ap.parse_known_args()[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()
    books = _get(session, f"{API}/books.json", required=True)["books"]
    print(f"Matthew Henry commentary: {len(books)} books")

    for book in books:
        out_path = os.path.join(OUT_DIR, f"{book['name']}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"  {book['name']}: already fetched, skipping")
            continue
        notes = {}
        for chapter in range(1, book["numberOfChapters"] + 1):
            payload = _get(session, f"{API}/{book['id']}/{chapter}.json")
            if payload is None:
                continue
            note = _chapter_note(payload)
            if note:
                notes[str(chapter)] = note
            time.sleep(0.02)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False)
        print(f"  {book['name']}: {len(notes)}/{book['numberOfChapters']} chapters")

    print(f"Done. Commentary stored in {OUT_DIR}")


if __name__ == "__main__":
    main()
