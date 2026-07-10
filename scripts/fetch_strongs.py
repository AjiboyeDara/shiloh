"""
Downloads the OpenScriptures Strong's dictionaries (Hebrew + Greek, from
Strong's Exhaustive Concordance, 1890; JSON version CC-BY-SA, attribution:
Open Scriptures) and merges them into the format app/word_study.py reads:

    resources/strongs/strongs.json
    { "H430": {"lemma": "אֱלֹהִים", "translit": "ʼĕlôhîym", "pron": "el-o-heem'",
               "strongs_def": "gods in the ordinary sense...",
               "kjv_def": "angels, exceeding, God (gods)..."},
      "G26":  {...}, ... }

The kjv_def field lists how the KJV translates each original word — it's
what lets a KJV English word ("charity") map back to its Greek or Hebrew
entries (agape, G26).
"""
import json
import os

import requests

URLS = {
    "Hebrew": "https://raw.githubusercontent.com/openscriptures/strongs/master/hebrew/strongs-hebrew-dictionary.js",
    "Greek": "https://raw.githubusercontent.com/openscriptures/strongs/master/greek/strongs-greek-dictionary.js",
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "strongs")
OUT_PATH = os.path.join(OUT_DIR, "strongs.json")

KEEP_FIELDS = ("lemma", "pron", "derivation", "strongs_def", "kjv_def")


def parse_dictionary_js(text: str) -> dict:
    """The source is a JS file: comment header, `var x = {...};` and a
    module.exports line. Extract and parse the object literal."""
    start = text.index("= {") + 2
    end = text.rindex("}") + 1
    return json.loads(text[start:end])


def main():
    merged = {}
    for label, url in URLS.items():
        print(f"Downloading Strong's {label} dictionary ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        entries = parse_dictionary_js(resp.text)
        for number, entry in entries.items():
            slim = {k: entry[k] for k in KEEP_FIELDS if entry.get(k)}
            # Hebrew uses "xlit", Greek uses "translit" — normalize.
            translit = entry.get("translit") or entry.get("xlit")
            if translit:
                slim["translit"] = translit
            merged[number] = slim
        print(f"  {len(entries)} {label} entries")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"Wrote {len(merged)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
