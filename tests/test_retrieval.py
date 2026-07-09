"""Tests for reference parsing, overlap logic, and retrieval merging.

Reference-parsing tests need data/kjv_verses.json (run
scripts/download_bible.py once); they're skipped if it's missing.
The pure-function tests always run.
"""
import os

import pytest

from app import retrieval

needs_data = pytest.mark.skipif(
    not os.path.exists(retrieval.VERSES_PATH),
    reason="data/kjv_verses.json missing; run scripts/download_bible.py",
)


# ── reference_passages ───────────────────────────────────────────────────

@needs_data
def test_whole_chapter_reference():
    refs = retrieval.reference_passages("What does Romans 8 say about the Spirit?")
    assert len(refs) == 1
    assert refs[0]["reference"] == "Romans 8"
    assert refs[0]["verse_start"] == 1
    assert "no condemnation" in refs[0]["text"]


@needs_data
def test_single_verse_reference():
    refs = retrieval.reference_passages("Explain John 3:16 please")
    assert refs[0]["reference"] == "John 3:16"
    assert refs[0]["verse_start"] == 16
    assert refs[0]["verse_end"] == 16
    assert "God so loved the world" in refs[0]["text"]


@needs_data
def test_verse_range_reference():
    refs = retrieval.reference_passages("Compare 1 John 4:7-9 with something")
    assert refs[0]["reference"] == "1 John 4:7-9"
    assert refs[0]["book"] == "1 John"  # not parsed as John


@needs_data
def test_psalm_singular_alias():
    refs = retrieval.reference_passages("walk me through psalm 23")
    assert refs[0]["book"] == "Psalms"
    assert refs[0]["chapter"] == 23


@needs_data
def test_long_chapter_is_capped():
    refs = retrieval.reference_passages("Thoughts on Psalms 119?")
    assert refs[0]["verse_end"] - refs[0]["verse_start"] + 1 == retrieval.MAX_CHAPTER_VERSES


@needs_data
def test_no_reference_in_plain_question():
    assert retrieval.reference_passages("Explain the parable of the prodigal son.") == []


@needs_data
def test_nonexistent_chapter_ignored():
    # "Jude 9" parses as chapter 9, but Jude only has one chapter — the
    # reference must be skipped without crashing.
    assert retrieval.reference_passages("What about Jude 9?") == []


@needs_data
def test_backwards_range_clamped():
    refs = retrieval.reference_passages("Read John 3:16-2")
    assert refs[0]["verse_start"] == 16
    assert refs[0]["verse_end"] == 16


# ── _overlaps / retrieve ─────────────────────────────────────────────────

def _p(book, chapter, vs, ve):
    return {"book": book, "chapter": chapter, "verse_start": vs, "verse_end": ve,
            "reference": f"{book} {chapter}:{vs}-{ve}", "text": ""}


def test_overlaps_same_chapter():
    assert retrieval._overlaps(_p("John", 3, 14, 18), _p("John", 3, 16, 20))
    assert not retrieval._overlaps(_p("John", 3, 1, 5), _p("John", 3, 6, 10))
    assert not retrieval._overlaps(_p("John", 3, 1, 5), _p("John", 4, 1, 5))
    assert not retrieval._overlaps(_p("John", 3, 1, 5), _p("Luke", 3, 1, 5))


@needs_data
def test_retrieve_puts_direct_reference_first_and_dedupes(monkeypatch):
    semantic = [_p("Romans", 8, 1, 5), _p("Luke", 15, 11, 15), _p("Acts", 2, 1, 5)]
    monkeypatch.setattr(retrieval, "search_passages", lambda q, top_k=6: semantic)
    results = retrieval.retrieve("What does Romans 8 teach?", top_k=3)
    assert results[0]["reference"] == "Romans 8"
    # The overlapping Romans 8:1-5 semantic chunk must be dropped.
    assert [r["reference"] for r in results[1:]] == ["Luke 15:11-15", "Acts 2:1-5"]


# ── chapter lookup ───────────────────────────────────────────────────────

@needs_data
def test_get_chapter_case_insensitive():
    assert retrieval.get_chapter("psalms", 23) == retrieval.get_chapter("Psalms", 23)
    assert retrieval.get_chapter("PSALM", 23)  # singular alias too
    assert retrieval.canonical_book("1 john") == "1 John"
    assert retrieval.canonical_book("Nonexistent") == "Nonexistent"


# ── query expansion ──────────────────────────────────────────────────────

def test_expand_query_kjv_synonyms():
    assert "holy ghost" in retrieval._expand_query("the Holy Spirit's role").lower()
    assert "charity" in retrieval._expand_query("what is love?").lower()
    # Already-KJV wording isn't double-expanded.
    q = "faith hope and charity"
    assert retrieval._expand_query(q) == q
    # Unrelated queries pass through untouched.
    q2 = "the walls of Jericho"
    assert retrieval._expand_query(q2) == q2
