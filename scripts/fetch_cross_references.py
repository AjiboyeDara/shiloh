"""
Downloads the openbible.info cross-reference dataset (derived from the
public-domain Treasury of Scripture Knowledge, ~340k links with community
votes) and converts it into the format app/retrieval.py reads:

    resources/cross_references/cross_references.json
    { "Genesis 1:1": ["John 1:1", "Hebrews 11:3", ...], ... }

Per source verse, references are sorted by vote count and capped at
MAX_REFS_PER_VERSE. Range targets ("Gen.1.1-Gen.1.3") keep the range when
it stays within one chapter, otherwise just the starting verse.

Data license: openbible.info cross references are CC-BY (attribution:
openbible.info).
"""
import io
import json
import os
import zipfile
from collections import defaultdict

import requests

ZIP_URL = "https://a.openbible.info/data/cross-references.zip"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "cross_references")
OUT_PATH = os.path.join(OUT_DIR, "cross_references.json")

MAX_REFS_PER_VERSE = 8
MIN_VOTES = 0  # drop links the community voted below zero

# openbible.info book abbreviations -> the full names used in our data.
BOOK_ABBREV = {
    "Gen": "Genesis", "Exod": "Exodus", "Lev": "Leviticus", "Num": "Numbers",
    "Deut": "Deuteronomy", "Josh": "Joshua", "Judg": "Judges", "Ruth": "Ruth",
    "1Sam": "1 Samuel", "2Sam": "2 Samuel", "1Kgs": "1 Kings", "2Kgs": "2 Kings",
    "1Chr": "1 Chronicles", "2Chr": "2 Chronicles", "Ezra": "Ezra",
    "Neh": "Nehemiah", "Esth": "Esther", "Job": "Job", "Ps": "Psalms",
    "Prov": "Proverbs", "Eccl": "Ecclesiastes", "Song": "Song of Solomon",
    "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations",
    "Ezek": "Ezekiel", "Dan": "Daniel", "Hos": "Hosea", "Joel": "Joel",
    "Amos": "Amos", "Obad": "Obadiah", "Jonah": "Jonah", "Mic": "Micah",
    "Nah": "Nahum", "Hab": "Habakkuk", "Zeph": "Zephaniah", "Hag": "Haggai",
    "Zech": "Zechariah", "Mal": "Malachi", "Matt": "Matthew", "Mark": "Mark",
    "Luke": "Luke", "John": "John", "Acts": "Acts", "Rom": "Romans",
    "1Cor": "1 Corinthians", "2Cor": "2 Corinthians", "Gal": "Galatians",
    "Eph": "Ephesians", "Phil": "Philippians", "Col": "Colossians",
    "1Thess": "1 Thessalonians", "2Thess": "2 Thessalonians",
    "1Tim": "1 Timothy", "2Tim": "2 Timothy", "Titus": "Titus",
    "Phlm": "Philemon", "Heb": "Hebrews", "Jas": "James", "1Pet": "1 Peter",
    "2Pet": "2 Peter", "1John": "1 John", "2John": "2 John",
    "3John": "3 John", "Jude": "Jude", "Rev": "Revelation",
}


def parse_single(ref):
    """'Gen.1.1' -> ('Genesis', 1, 1), or None if the book is unknown."""
    parts = ref.split(".")
    if len(parts) != 3 or parts[0] not in BOOK_ABBREV:
        return None
    return BOOK_ABBREV[parts[0]], int(parts[1]), int(parts[2])


def format_target(ref):
    """Format a target ref (possibly a range) as 'Book C:V' or 'Book C:V-V'."""
    if "-" in ref:
        start_s, end_s = ref.split("-", 1)
        start, end = parse_single(start_s), parse_single(end_s)
        if not start:
            return None
        if end and end[0] == start[0] and end[1] == start[1] and end[2] > start[2]:
            return f"{start[0]} {start[1]}:{start[2]}-{end[2]}"
        return f"{start[0]} {start[1]}:{start[2]}"
    single = parse_single(ref)
    if not single:
        return None
    return f"{single[0]} {single[1]}:{single[2]}"


def main():
    print(f"Downloading {ZIP_URL} ...")
    resp = requests.get(ZIP_URL, timeout=120)
    resp.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    txt_name = next(n for n in archive.namelist() if n.endswith(".txt"))
    print(f"Parsing {txt_name} ...")

    links = defaultdict(list)  # "Genesis 1:1" -> [(votes, "John 1:1"), ...]
    total = 0
    with archive.open(txt_name) as f:
        for raw in io.TextIOWrapper(f, encoding="utf-8"):
            if raw.startswith("From Verse"):  # header
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            source = parse_single(fields[0])
            target = format_target(fields[1])
            try:
                votes = int(fields[2])
            except ValueError:
                continue
            if not source or not target or votes < MIN_VOTES:
                continue
            key = f"{source[0]} {source[1]}:{source[2]}"
            links[key].append((votes, target))
            total += 1

    out = {}
    for key, refs in links.items():
        refs.sort(key=lambda r: -r[0])
        seen, kept = set(), []
        for _, target in refs:
            if target in seen:
                continue
            seen.add(target)
            kept.append(target)
            if len(kept) >= MAX_REFS_PER_VERSE:
                break
        out[key] = kept

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Kept {total} links across {len(out)} source verses.")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
