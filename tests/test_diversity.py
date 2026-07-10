"""Tests for the per-chapter diversity cap in fused ranking (pure function,
no data files needed)."""
from app.retrieval import _diversify


def item(score, book, chapter, n):
    return [score, f"doc-{book}-{chapter}-{n}", {"book": book, "chapter": chapter}]


def chapters(picked):
    return [(it[2]["book"], it[2]["chapter"]) for it in picked]


def test_caps_chunks_per_chapter():
    ranked = [item(0.9 - i * 0.1, "Romans", 8, i) for i in range(4)] + \
             [item(0.4, "John", 3, 0), item(0.3, "Psalms", 23, 0)]
    picked = chapters(_diversify(ranked, 4))
    assert picked == [("Romans", 8), ("Romans", 8), ("John", 3), ("Psalms", 23)]


def test_backfills_when_chapters_run_out():
    # Only one chapter available: the cap must not shrink the result set.
    ranked = [item(0.9 - i * 0.1, "Romans", 8, i) for i in range(5)]
    picked = _diversify(ranked, 4)
    assert len(picked) == 4
    # Backfill preserves score order.
    assert [it[0] for it in picked] == sorted([it[0] for it in picked], reverse=True)


def test_order_preserved_for_diverse_input():
    ranked = [item(0.9, "Romans", 8, 0), item(0.8, "John", 3, 0),
              item(0.7, "James", 1, 0)]
    assert _diversify(ranked, 3) == ranked
