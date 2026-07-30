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


def _rewrite(monkeypatch, reply, query):
    """Run _rewrite_query against a canned provider reply."""
    from app import rag
    monkeypatch.setattr(rag, "_generate", lambda *a, **kw: reply)
    retrieval._rewrite_query.cache_clear()
    return retrieval._rewrite_query(query)


def test_rewrite_query_adds_to_the_synonym_map(monkeypatch):
    """Additive: the curated terms are a floor the model only adds to."""
    query = "What is the Great Commission?"
    out = _rewrite(monkeypatch, "Go ye therefore, teach all nations", query)
    assert out == f"{retrieval._expand_query(query)} (Go ye therefore, teach all nations)"
    assert "preach the gospel" in out  # the mapped terms survived


def test_rewrite_query_strips_preamble_and_bullets(monkeypatch):
    out = _rewrite(monkeypatch, "Search terms:\n- take no thought\n- careful for nothing",
                   "anxiety")
    assert out.endswith("(take no thought careful for nothing)")


def test_rewrite_query_falls_back_when_provider_fails(monkeypatch):
    """A dead provider must degrade to the synonym map, not break search."""
    from app import rag

    def boom(*a, **kw):
        raise RuntimeError("provider down")
    monkeypatch.setattr(rag, "_generate", boom)
    retrieval._rewrite_query.cache_clear()
    query = "the Holy Spirit's role"
    assert retrieval._rewrite_query(query) == retrieval._expand_query(query)


def test_rewrite_query_falls_back_on_empty_reply(monkeypatch):
    assert _rewrite(monkeypatch, "   \n", "worry") == retrieval._expand_query("worry")


# ── context expansion ────────────────────────────────────────────────────

@needs_data
def test_expanded_text_widens_to_whole_chapter():
    # The regression that motivated expansion: 1 Kings 3:16-20 cuts the
    # Solomon story off before the ruling in v24-27.
    p = _p("1 Kings", 3, 16, 20)
    text, ref, count = retrieval.expanded_text(p)
    assert ref == "1 Kings 3"
    assert count == 28  # the whole chapter
    assert "Divide the living child" in text          # v25: the ruse
    assert "she is the mother thereof" in text        # v27: the ruling


@needs_data
def test_expanded_text_caps_long_chapters_around_hit():
    p = _p("Psalms", 119, 100, 104)
    text, ref, count = retrieval.expanded_text(p)
    assert ref.startswith("Psalms 119:")
    assert count == retrieval.EXPAND_MAX_VERSES
    first, last = ref.split(":")[1].split("-")
    assert int(last) - int(first) + 1 == retrieval.EXPAND_MAX_VERSES
    assert int(first) <= 100 <= int(last)


@needs_data
def test_expanded_text_falls_back_without_verse_info():
    p = {"book": "John", "chapter": 3, "verse_start": None, "verse_end": None,
         "reference": "John 3", "text": "original"}
    assert retrieval.expanded_text(p) == ("original", "John 3", 0)


# ── relevance floor ──────────────────────────────────────────────────────

def _scored(*scores):
    return [[s, f"doc{i}", {}] for i, s in enumerate(scores)]


def test_floor_drops_low_scoring_tail():
    ranked = _scored(0.06, 0.05, 0.04, 0.035, 0.02, 0.01)
    kept = retrieval._apply_floor(ranked)
    assert [r[0] for r in kept] == [0.06, 0.05, 0.04, 0.035]


def test_floor_keeps_flat_curves_whole():
    ranked = _scored(0.031, 0.030, 0.029, 0.029, 0.028, 0.028)
    assert retrieval._apply_floor(ranked) == ranked


def test_floor_always_keeps_minimum():
    ranked = _scored(0.06, 0.01, 0.005)
    assert len(retrieval._apply_floor(ranked)) == retrieval.MIN_KEEP


# ── reranking ────────────────────────────────────────────────────────────

def _c(doc, book="John", chapter=3):
    return [0.5, doc, {"book": book, "chapter": chapter}]


class _FakeReranker:
    def predict(self, pairs):
        # Score by document length: longest doc wins.
        return [len(doc) for _, doc in pairs]


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    monkeypatch.setattr(retrieval, "get_reranker", lambda: _FakeReranker())
    candidates = [_c("short"), _c("the longest document"), _c("mid doc")]
    reranked = retrieval._rerank("query", candidates)
    assert [c[1] for c in reranked] == ["the longest document", "mid doc", "short"]


def test_rerank_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(retrieval, "get_reranker", lambda: None)
    candidates = [_c("a"), _c("bb"), _c("ccc")]
    assert retrieval._rerank("query", candidates) == candidates
    assert "charity" in retrieval._expand_query("what is love?").lower()
    # Already-KJV wording isn't double-expanded.
    q = "faith hope and charity"
    assert retrieval._expand_query(q) == q
    # Unrelated queries pass through untouched.
    q2 = "the walls of Jericho"
    assert retrieval._expand_query(q2) == q2


# ── cross-reference expansion ────────────────────────────────────────────

def test_ref_to_candidate_parses_single_and_range():
    doc, meta = retrieval._ref_to_candidate("Romans 8:28")
    assert meta["book"] == "Romans" and meta["chapter"] == 8
    assert meta["verse_start"] == 28 and meta["verse_end"] == 28
    assert doc.startswith("28.")

    _, meta = retrieval._ref_to_candidate("Luke 15:20-24")
    assert meta["verse_start"] == 20 and meta["verse_end"] == 24
    assert meta["reference"] == "Luke 15:20-24"


def test_ref_to_candidate_rejects_garbage():
    assert retrieval._ref_to_candidate("not a reference") is None
    assert retrieval._ref_to_candidate("Nowhere 99:1") is None
