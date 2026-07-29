"""Tests for the quote-repair pass and context assembly in app.rag."""
import os

import pytest

from app import rag
from app.retrieval import VERSES_PATH

needs_data = pytest.mark.skipif(
    not os.path.exists(VERSES_PATH),
    reason="data/kjv_verses.json missing; run scripts/download_bible.py",
)

PASSAGES = [{
    "reference": "1 Kings 3:16-20", "text": "…",
    "book": "1 Kings", "chapter": 3, "verse_start": 16, "verse_end": 20,
}]

FABRICATED = ('Solomon ruled at once: "Give your bondwoman to thy bondwoman '
              'proverb her hand that doth not suck up milk" [1].')
CORRECTED = 'Solomon said, "Divide the living child in two" [1].'


@needs_data
def test_repair_returns_original_when_quotes_verify(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("no regeneration should happen")
    monkeypatch.setattr(rag, "_generate", boom)
    answer = 'The king said, "Divide the living child in two" [1].'
    assert rag.repair_quotes([], answer, PASSAGES) == answer


@needs_data
def test_repair_replaces_fabricated_quote(monkeypatch):
    monkeypatch.setattr(rag, "_generate", lambda msgs, p=None, m=None: CORRECTED)
    assert rag.repair_quotes([], FABRICATED, PASSAGES) == CORRECTED


@needs_data
def test_repair_keeps_original_when_revision_no_better(monkeypatch):
    monkeypatch.setattr(rag, "_generate", lambda msgs, p=None, m=None: FABRICATED)
    assert rag.repair_quotes([], FABRICATED, PASSAGES) == FABRICATED


@needs_data
def test_repair_survives_generation_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("provider down")
    monkeypatch.setattr(rag, "_generate", boom)
    assert rag.repair_quotes([], FABRICATED, PASSAGES) == FABRICATED


JOHN3 = [{
    "reference": "John 3:16", "text": "…",
    "book": "John", "chapter": 3, "verse_start": 16, "verse_end": 16,
}]
QUOTE = "For God so loved the world, that he gave his only begotten Son"


@needs_data
def test_attach_citations_adds_missing_marker():
    answer = f'Jesus taught that, "{QUOTE}."'
    out = rag.attach_citations(answer, JOHN3)
    assert out == f'Jesus taught that, "{QUOTE}."[1]'


@needs_data
def test_attach_citations_leaves_cited_quote_alone():
    answer = f'Jesus taught that, "{QUOTE}" [1].'
    assert rag.attach_citations(answer, JOHN3) == answer


@needs_data
def test_attach_citations_skips_unverifiable_quote():
    answer = 'The text says, "the aardvark danced upon the printing press."'
    assert rag.attach_citations(answer, JOHN3) == answer


@needs_data
def test_build_context_expands_top_passages():
    context = rag.build_context(PASSAGES)
    # The chunk ends at v20, but the prompt must include the story's ending.
    assert "Divide the living child" in context
    assert "[1] 1 Kings 3\n" in context
