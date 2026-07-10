"""
Word study: map a KJV English word to the Strong's dictionary entries
behind it, plus its concordance (everywhere the word occurs in the KJV).

There is no Strong's-tagged Bible in the data, but each Strong's entry's
kjv_def field lists how the KJV translates that original word — so matching
the English word against kjv_def recovers the Hebrew/Greek candidates
("charity" -> G26 agápē). Requires resources/strongs/strongs.json (run
scripts/fetch_strongs.py); degrades to concordance-only without it.
"""
import json
import os
import re
from functools import lru_cache

from app.retrieval import RESOURCES_DIR, _load_all_verses

MAX_STRONGS = 6
MAX_OCCURRENCES = 8
DEFINITION_LIMIT = 240


@lru_cache(maxsize=1)
def _strongs_data():
    path = os.path.join(RESOURCES_DIR, "strongs", "strongs.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _word_re(word: str):
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def _trim(text: str, limit: int = DEFINITION_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def strongs_for_word(word: str):
    """Strong's entries whose KJV translation list contains the word,
    most-primary first (position in kjv_def approximates how central the
    word is to that entry's translation)."""
    rx = _word_re(word)
    hits = []
    for number, entry in _strongs_data().items():
        kjv_def = entry.get("kjv_def", "")
        m = rx.search(kjv_def)
        if not m:
            continue
        hits.append((m.start(), len(kjv_def), number, entry))
    hits.sort(key=lambda h: (h[0], h[1]))
    return [{
        "number": number,
        "lemma": entry.get("lemma", ""),
        "translit": entry.get("translit", ""),
        "pron": entry.get("pron", ""),
        "definition": _trim(entry.get("strongs_def", "").strip(" ,;")),
        "kjv_def": _trim(entry.get("kjv_def", "")),
    } for _, _, number, entry in hits[:MAX_STRONGS]]


def concordance(word: str, limit: int = MAX_OCCURRENCES):
    """(total verse count, first `limit` occurrences in canonical order)."""
    rx = _word_re(word)
    count, samples = 0, []
    for (book, chapter), verses in _load_all_verses().items():
        for v in verses:
            if rx.search(v["text"]):
                count += 1
                if len(samples) < limit:
                    samples.append({
                        "reference": f"{book} {chapter}:{v['verse']}",
                        "text": v["text"],
                    })
    return count, samples


def word_study(word: str):
    word = word.strip()
    count, occurrences = concordance(word)
    return {
        "word": word,
        "strongs": strongs_for_word(word),
        "count": count,
        "occurrences": occurrences,
    }
