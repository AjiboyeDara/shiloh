"""Tests for the verse-window chunking in scripts/build_index.py."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "build_index",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "build_index.py"),
)
build_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_index)


def verses(book, chapter, n):
    return [{"book": book, "chapter": chapter, "verse": i, "text": f"v{i}"}
            for i in range(1, n + 1)]


def test_windows_overlap_and_cover_all_verses():
    chunks = build_index.make_chunks(verses("Test", 1, 12))
    # stride 3, size 5: windows start at 1, 4, 7, 10
    assert [c["verse_start"] for c in chunks] == [1, 4, 7, 10]
    assert chunks[0]["verse_end"] == 5
    assert chunks[-1]["verse_end"] == 12  # short final window still emitted
    covered = set()
    for c in chunks:
        covered.update(range(c["verse_start"], c["verse_end"] + 1))
    assert covered == set(range(1, 13))


def test_short_chapter_single_chunk():
    chunks = build_index.make_chunks(verses("Test", 1, 3))
    assert len(chunks) == 1
    assert chunks[0]["reference"] == "Test 1:1-3"


def test_single_verse_chapter_reference_format():
    chunks = build_index.make_chunks(verses("Test", 1, 1))
    assert len(chunks) == 1
    assert chunks[0]["reference"] == "Test 1:1"


def test_chunks_do_not_cross_chapters():
    vs = verses("Test", 1, 6) + verses("Test", 2, 6)
    chunks = build_index.make_chunks(vs)
    for c in chunks:
        assert c["verse_start"] >= 1 and c["verse_end"] <= 6
    assert {c["chapter"] for c in chunks} == {1, 2}
