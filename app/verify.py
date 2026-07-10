"""
Quote verification: quotation marks in a generated answer are a claim that
the enclosed words are scripture. Every such span is checked word-for-word
against the KJV text, so the UI can mark verbatim quotes as verified and
flag near-quotes with the real wording. Models — small local ones
especially — mangle KJV quotes constantly; this keeps the app accountable
to the text.
"""
import re
from bisect import bisect_right
from difflib import SequenceMatcher
from functools import lru_cache

from app.retrieval import _load_all_verses

# Quoted spans with fewer words than this are vocabulary mentions
# ("charity"), not scripture quotations.
MIN_QUOTE_WORDS = 4
MAX_QUOTE_CHARS = 400
# A sliding-window similarity at or above this counts as a near-quote of
# the verse it landed on; below it the quote is reported as not found.
# (Wholly invented quotes score ~0.5 against their best window; real
# quotes with a few altered words score ~0.7.)
FUZZY_THRESHOLD = 0.62
ACTUAL_TEXT_LIMIT = 220

_QUOTE_RE = re.compile(r'[“"]([^“”"]+?)[”"]')
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")


def _normalize(text: str) -> str:
    """Case, punctuation, and quote-style differences are not misquotes."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " ".join(re.findall(r"[a-z']+", text))


@lru_cache(maxsize=1)
def _chapter_corpus():
    """[(book, chapter, normalized_chapter_text, [(char_offset, verse)])],
    so a character position in a match maps back to a verse number."""
    corpus = []
    for (book, chapter), verses in _load_all_verses().items():
        parts, offsets, pos = [], [], 0
        for v in verses:
            norm = _normalize(v["text"])
            offsets.append((pos, v["verse"]))
            parts.append(norm)
            pos += len(norm) + 1
        corpus.append((book, chapter, " ".join(parts), offsets))
    return corpus


def _verse_at(offsets, pos):
    i = bisect_right(offsets, (pos, float("inf"))) - 1
    return offsets[max(i, 0)][1]


def _fmt_ref(book, chapter, v1, v2):
    return f"{book} {chapter}:{v1}" + (f"-{v2}" if v2 != v1 else "")


def _iter_chapters(preferred):
    """All chapters, the retrieved passages' chapters first, so a quote that
    appears in several books resolves to the passage that was cited."""
    pref, rest = [], []
    for entry in _chapter_corpus():
        (pref if (entry[0], entry[1]) in preferred else rest).append(entry)
    return pref + rest


def _find_in_chapter(text, fragments):
    """Word-aligned positions of each fragment, in order, or None. Multiple
    fragments come from a quote elided with "..."."""
    spans, pos = [], 0
    for frag in fragments:
        i = pos
        while True:
            i = text.find(frag, i)
            if i == -1:
                return None
            end = i + len(frag)
            if (i == 0 or text[i - 1] == " ") and (end == len(text) or text[end] == " "):
                spans.append((i, end))
                pos = end
                break
            i += 1
    return spans


def _find_exact(fragments, preferred):
    for book, chapter, text, offsets in _iter_chapters(preferred):
        spans = _find_in_chapter(text, fragments)
        if spans:
            return (book, chapter,
                    _verse_at(offsets, spans[0][0]),
                    _verse_at(offsets, spans[-1][1] - 1))
    return None


def _find_fuzzy(norm_quote, preferred):
    """Best near-match for the quote within the retrieved passages'
    chapters (whole-Bible fuzzy search would be slow and rarely right)."""
    qwords = norm_quote.split()
    best_ratio, best = 0.0, None
    for book, chapter, text, offsets in _chapter_corpus():
        if (book, chapter) not in preferred:
            continue
        words = text.split()
        starts, pos = [], 0
        for w in words:
            starts.append(pos)
            pos += len(w) + 1
        # The altered quote may have gained or lost words relative to the
        # verse, so score windows a couple of words narrower and wider too.
        sizes = {max(3, len(qwords) - 2), len(qwords), len(qwords) + 2}
        for size in sizes:
            for i in range(max(1, len(words) - size + 1)):
                window = " ".join(words[i:i + size])
                m = SequenceMatcher(None, norm_quote, window)
                if m.quick_ratio() <= best_ratio:
                    continue
                ratio = m.ratio()
                if ratio > best_ratio:
                    s = starts[i]
                    e = starts[min(i + size, len(words)) - 1]
                    best_ratio = ratio
                    best = (book, chapter, _verse_at(offsets, s), _verse_at(offsets, e))
    return best if best_ratio >= FUZZY_THRESHOLD else None


def _verse_text(book, chapter, v1, v2):
    verses = _load_all_verses()[(book, chapter)]
    joined = " ".join(v["text"] for v in verses if v1 <= v["verse"] <= v2)
    if len(joined) > ACTUAL_TEXT_LIMIT:
        joined = joined[:ACTUAL_TEXT_LIMIT].rsplit(" ", 1)[0] + "…"
    return joined


def verify_quotes(answer: str, passages=None):
    """Check every quoted span in the answer against the KJV. Returns one
    {quote, status, reference?, actual?} per distinct quote: "verified"
    (word-for-word, elisions marked with "..." allowed), "mismatch" (close
    to a real verse but altered; `actual` holds the KJV wording), or
    "not_found"."""
    preferred = {(p["book"], p["chapter"]) for p in passages or []}
    results, seen = [], set()
    for m in _QUOTE_RE.finditer(answer):
        quote = m.group(1).strip()
        if len(quote) > MAX_QUOTE_CHARS or quote in seen:
            continue
        norm = _normalize(quote)
        if len(norm.split()) < MIN_QUOTE_WORDS:
            continue
        seen.add(quote)

        fragments = [f for f in (_normalize(p) for p in _ELLIPSIS_RE.split(quote)) if f]
        hit = _find_exact(fragments, preferred)
        if hit:
            results.append({"quote": quote, "status": "verified",
                            "reference": _fmt_ref(*hit)})
            continue
        near = _find_fuzzy(norm, preferred)
        if near:
            book, chapter, v1, v2 = near
            results.append({"quote": quote, "status": "mismatch",
                            "reference": _fmt_ref(book, chapter, v1, v2),
                            "actual": _verse_text(book, chapter, v1, v2)})
        else:
            results.append({"quote": quote, "status": "not_found"})
    return results
