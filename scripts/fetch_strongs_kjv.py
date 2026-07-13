"""
Downloads a Strong's-tagged KJV so word study can resolve the exact
Hebrew/Greek word behind an English word in a specific verse, instead of
listing every candidate entry.

Source: the data files of the King James with Strong bible app
(github.com/1John419/kjs, GPL-3.0 app code). The data itself derives from
public-domain material: the KJV text (1769) and Strong's Exhaustive
Concordance tagging (1890). Fetched as an optional local resource; the
README's license note applies.

Output format read by app/word_study.py:

    resources/strongs/kjv_tagged.json
    { "Genesis": { "1:1": [["In the beginning", ["H7225"]],
                            ["God", ["H430"]], ...], ... }, ... }

Each verse is a list of (KJV phrase, Strong's numbers) pairs in text order.
"""
import json
import os
import re

import requests

BASE = "https://raw.githubusercontent.com/1John419/kjs/master/json"
TEXT_URL = f"{BASE}/strong_pure.json"    # pure KJV wording (not divine-names)
LISTS_URL = f"{BASE}/kjv_lists.json"     # citations: verse index -> reference

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "strongs")
OUT_PATH = os.path.join(OUT_DIR, "kjv_tagged.json")

REF_RE = re.compile(r"^(.+?)\s+(\d+):(\d+)$")


def fetch_json(url):
    print(f"Downloading {url.rsplit('/', 1)[-1]} ...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    return resp.json()


def main():
    citations = fetch_json(LISTS_URL)["citations"]
    tagged = fetch_json(TEXT_URL)["maps"]
    if len(citations) != len(tagged):
        raise SystemExit(
            f"citation/text length mismatch: {len(citations)} vs {len(tagged)}"
        )

    by_book = {}
    for entry in tagged:
        ref = citations[entry["k"]]
        m = REF_RE.match(ref)
        if not m:
            raise SystemExit(f"Unparseable citation: {ref!r}")
        book, chapter, verse = m.group(1), m.group(2), m.group(3)
        pairs = [[phrase, nums] for phrase, nums in entry["v"] if nums]
        if pairs:
            by_book.setdefault(book, {})[f"{chapter}:{verse}"] = pairs

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(by_book, f, ensure_ascii=False)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Wrote {len(by_book)} books ({size_mb:.1f} MB) to {OUT_PATH}")


if __name__ == "__main__":
    main()
