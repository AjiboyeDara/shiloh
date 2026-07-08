"""
Retrieval layer: embeds a query and pulls the most relevant Bible passages
from the local Chroma index. Also exposes hooks for cross-references,
commentary, and Strong's data if the user has populated resources/.
"""
import json
import os
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
