"""
Retrieval evaluation against the golden set in scripts/golden_set.json.

For each question, runs the app's real retrieval pipeline and checks which
of the expected passages came back in the top k (a retrieved chunk counts
if it overlaps the expected verse range). Reports:

  recall@k   mean fraction of a question's expected passages retrieved
  hit@k      fraction of questions with at least one expected passage
  chapters@k mean distinct (book, chapter) pairs per result set (diversity)

Usage:
  .venv/bin/python scripts/eval_retrieval.py            # k=6
  .venv/bin/python scripts/eval_retrieval.py --top-k 8
  .venv/bin/python scripts/eval_retrieval.py -v         # per-question detail

Use it to A/B retrieval changes (embedding model, synonym map, fusion
tweaks): run once on main, once on your branch, compare the two summaries.
"""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag import _retrieval_query  # noqa: E402
from app.retrieval import canonical_book, retrieve  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")

REF_RE = re.compile(r"^(.+?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$")


def parse_ref(ref: str):
    """'Romans 8:1-4' -> ('Romans', 8, 1, 4); verse part optional."""
    m = REF_RE.match(ref.strip())
    if not m:
        raise ValueError(f"Unparseable reference in golden set: {ref!r}")
    book = canonical_book(m.group(1))
    chapter = int(m.group(2))
    v1 = int(m.group(3)) if m.group(3) else None
    v2 = int(m.group(4)) if m.group(4) else v1
    return book, chapter, v1, v2


def passage_matches(passage: dict, ref) -> bool:
    book, chapter, v1, v2 = ref
    if passage["book"] != book or passage["chapter"] != chapter:
        return False
    if v1 is None or passage.get("verse_start") is None:
        return True
    return not (passage["verse_end"] < v1 or v2 < passage["verse_start"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="per-question detail, including what was retrieved on misses")
    args = ap.parse_args()

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    recalls, hits, chapter_counts = [], 0, []
    failures = []

    for q in questions:
        expected = [parse_ref(r) for r in q["expected"]]
        # Follow-up cases carry conversation history; build the retrieval
        # query the same way the app does.
        history = [SimpleNamespace(**t) for t in q.get("history", [])]
        query = _retrieval_query(q["question"], history) if history else q["question"]
        results = retrieve(query, top_k=args.top_k)

        found = [any(passage_matches(p, ref) for p in results) for ref in expected]
        recall = sum(found) / len(expected)
        recalls.append(recall)
        hits += any(found)
        chapter_counts.append(len({(p["book"], p["chapter"]) for p in results}))

        missed = [r for r, ok in zip(q["expected"], found) if not ok]
        status = "PASS" if not missed else ("part" if any(found) else "MISS")
        if missed:
            failures.append((q["question"], missed, results))
        if args.verbose:
            print(f"[{status}] {recall:.2f}  {q['question']}")
            if missed:
                print(f"        missed: {', '.join(missed)}")
                print(f"        got:    {', '.join(p['reference'] for p in results)}")

    n = len(questions)
    print(f"\n{'=' * 56}")
    print(f"questions:   {n}   (top_k={args.top_k})")
    print(f"recall@{args.top_k}:    {sum(recalls) / n:.3f}")
    print(f"hit@{args.top_k}:       {hits / n:.3f}   ({hits}/{n} questions)")
    print(f"chapters@{args.top_k}:  {sum(chapter_counts) / n:.2f} distinct per result set")
    if failures and not args.verbose:
        print(f"\n{len(failures)} question(s) missing expected passages (rerun with -v for detail):")
        for question, missed, _ in failures:
            print(f"  - {question}  (missed {', '.join(missed)})")


if __name__ == "__main__":
    main()
