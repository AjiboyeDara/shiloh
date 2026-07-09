"""
Retrieval layer: embeds a query and pulls the most relevant Bible passages
from the local Chroma index. Also exposes hooks for cross-references,
commentary, and Strong's data if the user has populated resources/.
"""
import json
import os
import re
from functools import lru_cache

import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(DATA_DIR, "chroma_index")
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "..", "resources")

EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=INDEX_DIR)
    return client.get_collection("bible_kjv")


def search_passages(query: str, top_k: int = 6):
    """Returns a list of dicts: reference, text, book, chapter,
    verse_start, verse_end."""
    model = get_embedder()
    collection = get_collection()

    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    passages = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        passages.append({
            "reference": meta["reference"],
            "text": doc,
            "book": meta["book"],
            "chapter": meta["chapter"],
            "verse_start": meta.get("verse_start"),
            "verse_end": meta.get("verse_end"),
        })
    return passages


VERSES_PATH = os.path.join(DATA_DIR, "kjv_verses.json")


@lru_cache(maxsize=1)
def _load_all_verses():
    """Load the full KJV verse list once and index it by (book, chapter)."""
    with open(VERSES_PATH, encoding="utf-8") as f:
        verses = json.load(f)
    by_chapter = {}
    for v in verses:
        by_chapter.setdefault((v["book"], v["chapter"]), []).append(v)
    for vlist in by_chapter.values():
        vlist.sort(key=lambda v: v["verse"])
    return by_chapter


def get_chapter(book: str, chapter: int):
    """Return the ordered verses (verse, text) for a given book + chapter."""
    vlist = _load_all_verses().get((book, chapter), [])
    return [{"verse": v["verse"], "text": v["text"]} for v in vlist]


# ── Reference-aware retrieval ────────────────────────────────────────────
# Semantic search alone can miss an explicitly named passage ("What does
# Romans 8 say about the Spirit?" isn't guaranteed to retrieve Romans 8).
# So references written in the question are parsed out and their text is
# always included, with vector search filling the remaining slots.

MAX_REF_PASSAGES = 2       # at most this many direct references per query
MAX_CHAPTER_VERSES = 60    # cap whole-chapter references (Psalm 119 is 176)


@lru_cache(maxsize=1)
def _reference_pattern():
    """Regex matching 'Book Chapter', 'Book Chapter:Verse', and
    'Book Chapter:Start-End', built from the book names in the data."""
    canonical = {b.lower(): b for b, _ in _load_all_verses().keys()}
    canonical["psalm"] = "Psalms"  # people usually write "Psalm 23"
    # Longest first so "1 John" wins over "John", "Psalms" over "Psalm".
    alts = "|".join(re.escape(n) for n in sorted(canonical, key=len, reverse=True))
    pattern = re.compile(
        rf"\b({alts})\s+(\d{{1,3}})(?:\s*:\s*(\d{{1,3}})(?:\s*[-–—]\s*(\d{{1,3}}))?)?",
        re.IGNORECASE,
    )
    return pattern, canonical


def reference_passages(query: str):
    """Passages for scripture references written directly in the query."""
    pattern, canonical = _reference_pattern()
    passages = []
    seen = set()
    for m in pattern.finditer(query):
        book = canonical[m.group(1).lower()]
        chapter = int(m.group(2))
        verses = _load_all_verses().get((book, chapter))
        if not verses:
            continue
        if m.group(3):
            v_start = int(m.group(3))
            v_end = int(m.group(4)) if m.group(4) else v_start
            if v_end < v_start:
                v_end = v_start
            window = [v for v in verses if v_start <= v["verse"] <= v_end]
        else:
            window = verses[:MAX_CHAPTER_VERSES]
        if not window:
            continue
        key = (book, chapter, window[0]["verse"], window[-1]["verse"])
        if key in seen:
            continue
        seen.add(key)
        whole_chapter = len(window) == len(verses)
        if whole_chapter:
            ref = f"{book} {chapter}"
        elif len(window) > 1:
            ref = f"{book} {chapter}:{window[0]['verse']}-{window[-1]['verse']}"
        else:
            ref = f"{book} {chapter}:{window[0]['verse']}"
        passages.append({
            "reference": ref,
            "text": " ".join(f"{v['verse']}. {v['text']}" for v in window),
            "book": book,
            "chapter": chapter,
            "verse_start": window[0]["verse"],
            "verse_end": window[-1]["verse"],
        })
        if len(passages) >= MAX_REF_PASSAGES:
            break
    return passages


def _overlaps(a, b):
    if a["book"] != b["book"] or a["chapter"] != b["chapter"]:
        return False
    return not (a["verse_end"] < b["verse_start"] or b["verse_end"] < a["verse_start"])


def retrieve(query: str, top_k: int = 6):
    """Direct references from the query first, then semantic results
    (skipping any that overlap a direct reference) up to top_k total."""
    direct = reference_passages(query)
    results = list(direct)
    for p in search_passages(query, top_k=top_k):
        if len(results) >= top_k:
            break
        if any(_overlaps(p, d) for d in direct):
            continue
        results.append(p)
    return results


def load_cross_references(book: str, chapter: int, verse: int):
    """
    Optional: looks up cross-references if the user has placed a file at
    resources/cross_references/cross_references.json in the format:
    { "Book Chapter:Verse": ["Book Chapter:Verse", ...], ... }

    Source suggestion (public domain, derived from the Treasury of Scripture
    Knowledge): https://www.openbible.info/labs/cross-references/
    or the scrollmapper/bible_databases GitHub repo.
    """
    path = os.path.join(RESOURCES_DIR, "cross_references", "cross_references.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    key = f"{book} {chapter}:{verse}"
    return data.get(key, [])


def load_commentary(book: str, chapter: int):
    """
    Optional: looks up commentary if the user has placed files at
    resources/commentary/<Book>.json in the format:
    { "1": "commentary text for chapter 1", "2": "...", ... }

    Source suggestion (public domain): Matthew Henry's Concise Commentary,
    available via ccel.org or bundled in the scrollmapper repo's commentary
    formats.
    """
    path = os.path.join(RESOURCES_DIR, "commentary", f"{book}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(str(chapter))


def load_strongs(strongs_number: str):
    """
    Optional: looks up a Strong's number if the user has placed a file at
    resources/strongs/strongs.json in the format:
    { "H430": "Elohim - God, gods...", "G26": "agape - love...", ... }

    Source suggestion (public domain): OpenScriptures Strong's dictionary
    data, or Blue Letter Bible's downloadable lexicon.
    """
    path = os.path.join(RESOURCES_DIR, "strongs", "strongs.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(strongs_number)
