"""
Downloads the King James Version (public domain) as structured JSON and
normalizes it into data/kjv_verses.json, a flat list of:
    {"book": "Genesis", "chapter": 1, "verse": 1, "text": "..."}

Source: thiagobodruk/bible (MIT-licensed repo, KJV text is public domain).
"""
import json
import os
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "en_kjv_raw.json")
OUT_PATH = os.path.join(DATA_DIR, "kjv_verses.json")

# Standard 66-book order + full names, matching the order of the source file.
BOOK_NAMES = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John",
    "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
]


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"Raw file already exists at {RAW_PATH}, skipping download.")
        return
    print(f"Downloading KJV text from {RAW_URL} ...")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    print("Download complete.")


def normalize():
    with open(RAW_PATH, encoding="utf-8-sig") as f:
        books = json.load(f)

    if len(books) != len(BOOK_NAMES):
        raise ValueError(
            f"Expected {len(BOOK_NAMES)} books, source has {len(books)}. "
            "The source format may have changed — check BOOK_NAMES ordering."
        )

    verses = []
    for book, name in zip(books, BOOK_NAMES):
        for chapter_idx, chapter in enumerate(book["chapters"], start=1):
            for verse_idx, text in enumerate(chapter, start=1):
                verses.append({
                    "book": name,
                    "chapter": chapter_idx,
                    "verse": verse_idx,
                    "text": text.strip(),
                })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(verses)} verses to {OUT_PATH}")


if __name__ == "__main__":
    download()
    normalize()
