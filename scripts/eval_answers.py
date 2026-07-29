"""
End-to-end answer evaluation against the golden set in scripts/golden_set.json.

Where eval_retrieval.py scores what comes back from the index, this scores
what the LLM does with it. For each question it runs the full pipeline
(answer_question, including the quote-repair pass) and reports:

  citations    fraction of answers whose [n] markers all resolve to a
               retrieved passage, and fraction with at least one citation
  quotes       verified / mismatch / not_found counts across all quoted
               scripture spans (verify_quotes), and fraction of answers
               with no flagged quotes
  grounded     fraction of answers an LLM judge says are fully supported
               by the passages the model was shown (skip with --no-judge)

Usage:
  .venv/bin/python scripts/eval_answers.py              # first 15 questions
  .venv/bin/python scripts/eval_answers.py --all        # full set (slow)
  .venv/bin/python scripts/eval_answers.py -v --no-judge

Generation goes through the provider configured in .env, so a run with
Ollama is free but takes ~30s per question.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag import answer_question, build_context, _generate  # noqa: E402
from app.verify import verify_quotes  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")

CITATION_RE = re.compile(r"\[(\d+)\]")

JUDGE_PROMPT = """You are grading a Bible-study answer for groundedness.

Passages the answerer was given:
{context}

Question: {question}

Answer being graded:
{answer}

Grade ONLY groundedness: does the answer claim scripture says something the
passages above do not say (or say the opposite of)? General knowledge
framing (authorship, genre) is fine. Do NOT penalize the answer for leaving
passages out, being brief, or not covering everything — omission is not a
grounding failure.
Reply "yes" (grounded) or "no" (contains unsupported scripture claims) on
the first line, then one short reason."""


def check_citations(answer: str, passage_count: int):
    """(all_valid, has_any): every [n] must point at a retrieved passage."""
    nums = [int(n) for n in CITATION_RE.findall(answer)]
    return all(1 <= n <= passage_count for n in nums), bool(nums)


def judge_grounded(question: str, answer: str, passages):
    """Ask the configured LLM whether the answer sticks to its passages.
    Judged against the same expanded context build_context gives the
    answerer, so chapter-widened claims aren't false negatives."""
    prompt = JUDGE_PROMPT.format(
        context=build_context(passages), question=question, answer=answer
    )
    reply = _generate([{"role": "user", "content": prompt}])
    verdict = reply.strip().lower()
    return verdict.startswith("yes"), reply.strip().splitlines()[0][:120]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--limit", type=int, default=15,
                    help="how many golden questions to run (default 15)")
    ap.add_argument("--all", action="store_true", help="run the full set")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM groundedness judge")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if not args.all:
        questions = questions[:args.limit]

    cite_valid = cite_any = clean_quotes = grounded = judged = 0
    quote_counts = {"verified": 0, "mismatch": 0, "not_found": 0}
    problems = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        answer, passages = answer_question(question, top_k=args.top_k)

        ok_cites, has_cites = check_citations(answer, len(passages))
        cite_valid += ok_cites
        cite_any += has_cites

        checks = verify_quotes(answer, passages)
        for c in checks:
            quote_counts[c["status"]] += 1
        flagged = [c for c in checks if c["status"] != "verified"]
        clean_quotes += not flagged

        verdict_note = ""
        if not args.no_judge:
            is_grounded, reason = judge_grounded(question, answer, passages)
            judged += 1
            grounded += is_grounded
            if not is_grounded:
                verdict_note = reason

        issues = []
        if not ok_cites:
            issues.append("bad citation number")
        if flagged:
            issues.append(f"{len(flagged)} flagged quote(s)")
        if verdict_note:
            issues.append(f"judge: {verdict_note}")
        if issues:
            problems.append((question, issues))
        if args.verbose:
            status = "PASS" if not issues else "FAIL"
            print(f"[{status}] ({i}/{len(questions)}) {question}")
            for issue in issues:
                print(f"        {issue}")
        else:
            print(f"  ({i}/{len(questions)}) {question[:60]}", flush=True)

    n = len(questions)
    total_quotes = sum(quote_counts.values())
    print(f"\n{'=' * 56}")
    print(f"questions:        {n}   (top_k={args.top_k})")
    print(f"citations valid:  {cite_valid / n:.3f}   ({cite_valid}/{n} answers)")
    print(f"has citations:    {cite_any / n:.3f}   ({cite_any}/{n} answers)")
    print(f"clean quotes:     {clean_quotes / n:.3f}   ({clean_quotes}/{n} answers)")
    if total_quotes:
        print(f"quoted spans:     {total_quotes}   "
              f"(verified {quote_counts['verified']}, "
              f"mismatch {quote_counts['mismatch']}, "
              f"not_found {quote_counts['not_found']})")
    if judged:
        print(f"grounded (judge): {grounded / judged:.3f}   ({grounded}/{judged} answers)")
    if problems and not args.verbose:
        print(f"\n{len(problems)} answer(s) with issues:")
        for question, issues in problems:
            print(f"  - {question}  ({'; '.join(issues)})")


if __name__ == "__main__":
    main()
