"""Tests for KJV text normalization in scripts/download_bible.py."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "download_bible",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "download_bible.py"),
)
download_bible = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_bible)

strip_annotations = download_bible.strip_annotations


def test_strips_single_annotation():
    assert strip_annotations(
        "he leadeth me beside the still waters. {still...: Heb. waters of quietness}"
    ) == "he leadeth me beside the still waters."


def test_strips_multiple_annotations_and_collapses_spaces():
    text = ("green pastures: {green: Heb. pastures of tender grass} "
            "{still: Heb. quietness} He restoreth my soul")
    assert strip_annotations(text) == "green pastures: He restoreth my soul"


def test_plain_text_untouched():
    assert strip_annotations("For God so loved the world") == "For God so loved the world"


def test_supplied_words_kept_without_braces():
    # KJV italicized (translator-supplied) words have no colon and belong
    # in the verse text.
    assert strip_annotations("The LORD {is} my shepherd; I shall not want.") == \
        "The LORD is my shepherd; I shall not want."
    assert strip_annotations("for thou {art} with me") == "for thou art with me"


def test_mixed_note_and_supplied_word():
    text = "thou anointest my head with oil; my cup runneth over. {anointest: Heb. makest fat} It {is} good."
    assert strip_annotations(text) == \
        "thou anointest my head with oil; my cup runneth over. It is good."


def test_nested_supplied_word_inside_note():
    # Micah 7:12 in the source: a note whose content itself contains a
    # supplied-word brace pair.
    text = "and from sea to sea. {{and from} the fortified cities: or, even to the fortified cities}"
    assert strip_annotations(text) == "and from sea to sea."


def test_epistle_colophon_removed():
    text = "Amen. «{Written to the Romans from Corinthus, and sent} by Phebe servant of the church.}»"
    assert strip_annotations(text) == "Amen."


def test_malformed_note_with_dangling_tail():
    # Hebrews 10:34 in the source: a note followed by a stray ' word}'.
    text = "an enduring substance. {in yourselves...: or, that ye have in or, for} yourselves}"
    assert strip_annotations(text) == "an enduring substance."
