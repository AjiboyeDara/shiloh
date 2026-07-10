"""Tests for quote verification (app/verify.py).

All matching tests need data/kjv_verses.json (run scripts/download_bible.py
once); they're skipped if it's missing. The normalization tests always run.
"""
import os

import pytest

from app import verify
from app.retrieval import VERSES_PATH

needs_data = pytest.mark.skipif(
    not os.path.exists(VERSES_PATH),
    reason="data/kjv_verses.json missing; run scripts/download_bible.py",
)

PASSAGES = [{"book": "Romans", "chapter": 8}, {"book": "John", "chapter": 3}]


def check_one(answer, passages=PASSAGES):
    results = verify.verify_quotes(answer, passages)
    assert len(results) == 1, results
    return results[0]


# ── normalization ────────────────────────────────────────────────────────

def test_normalize_strips_case_and_punctuation():
    assert verify._normalize("For God so loved the world,") == \
        "for god so loved the world"


def test_normalize_keeps_apostrophes():
    assert verify._normalize("God’s elect") == "god's elect"


# ── quote extraction rules ───────────────────────────────────────────────

@needs_data
def test_short_mentions_are_skipped():
    assert verify.verify_quotes('Paul speaks of "charity" here.', PASSAGES) == []


@needs_data
def test_curly_and_straight_quotes_both_extracted():
    r = check_one('Jesus said “For God so loved the world, that he gave '
                  'his only begotten Son” to Nicodemus.')
    assert r["status"] == "verified"
    assert r["reference"] == "John 3:16"


# ── verification outcomes ────────────────────────────────────────────────

@needs_data
def test_exact_quote_verified_with_verse_reference():
    r = check_one('The Spirit "maketh intercession for us with groanings '
                  'which cannot be uttered" for the saints.')
    assert r["status"] == "verified"
    assert r["reference"] == "Romans 8:26"


@needs_data
def test_quote_spanning_verses_gets_a_range():
    r = check_one('"But if the Spirit of him that raised up Jesus from the '
                  'dead dwell in you, he that raised up Christ from the dead '
                  'shall also quicken your mortal bodies by his Spirit that '
                  'dwelleth in you. Therefore, brethren, we are debtors" [1]')
    assert r["status"] == "verified"
    assert r["reference"] == "Romans 8:11-12"


@needs_data
def test_elided_quote_verified():
    r = check_one('We are "debtors...to live after the flesh" no longer.')
    assert r["status"] == "verified"
    assert r["reference"] == "Romans 8:12"


@needs_data
def test_altered_quote_flagged_with_actual_wording():
    r = check_one('"God so loved the earth that he sent his one and only Son"')
    assert r["status"] == "mismatch"
    assert r["reference"] == "John 3:16"
    assert "God so loved the world" in r["actual"]


@needs_data
def test_invented_quote_not_found():
    r = check_one('"and the spirit of tranquility descended upon the marketplace"')
    assert r["status"] == "not_found"
    assert r.get("reference") is None


@needs_data
def test_exact_quote_found_outside_retrieved_passages():
    # Whole-Bible exact search still works when the quote isn't in the
    # retrieved chapters (here: Psalm 23 vs. Romans 8 / John 3).
    r = check_one('"The LORD is my shepherd; I shall not want" is beloved.')
    assert r["status"] == "verified"
    assert r["reference"] == "Psalms 23:1"


@needs_data
def test_duplicate_quotes_reported_once():
    answer = ('"maketh intercession for us with groanings which cannot be '
              'uttered" and again "maketh intercession for us with groanings '
              'which cannot be uttered"')
    assert len(verify.verify_quotes(answer, PASSAGES)) == 1


@needs_data
def test_preferred_chapters_win_reference_ambiguity():
    # "take up his cross and follow me" appears in Matthew, Mark, and Luke;
    # with Mark 8 retrieved, the citation should resolve there.
    r = check_one('"take up his cross, and follow me" [1]',
                  passages=[{"book": "Mark", "chapter": 8}])
    assert r["status"] == "verified"
    assert r["reference"].startswith("Mark 8:")
