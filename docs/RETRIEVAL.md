# Retrieval: how it's measured, and what has been tried

Everything Shiloh answers is capped by what retrieval finds. A passage that
never reaches the model can't be cited, can't be quoted, and can't appear in a
reading plan. So retrieval changes are measured, not eyeballed.

**Run the eval before and after any change to `app/retrieval.py`,
`scripts/build_index.py`, or the embedding model.**

```bash
python scripts/eval_retrieval.py          # recall@k, hit@k, chapter diversity
python scripts/eval_answers.py            # first 15 questions, end to end
python scripts/eval_answers.py --all      # full set (slow)
python scripts/eval_answers.py --no-judge # skip the LLM grader
```

Both score against `scripts/golden_set.json` — 66 thematic questions, mostly
without explicit references so they exercise semantic and lexical search rather
than the reference parser. A retrieved chunk counts as a hit if it overlaps the
expected verse range.

`eval_retrieval.py` scores what comes out of the index. `eval_answers.py` scores
what the model does with it: whether every `[n]` resolves to a real passage,
whether quoted scripture survives word-for-word verification, and whether an LLM
judge thinks the answer is supported by the passages the model was shown.

---

## Current baseline

`all-MiniLM-L6-v2` + BM25 fused with reciprocal-rank fusion, synonym-map query
expansion, `top_k=6`:

```
recall@6:    0.780
hit@6:       0.985   (65/66 questions)
chapters@6:  4.89 distinct per result set
```

Roughly one expected passage in five never reaches the model. That's the number
to beat.

---

## What has been tried and lost

Keeping the failures here is the point — it stops the same experiment being run
three times.

### Stronger embedding models (2026-07)

All at or below the `all-MiniLM-L6-v2` + RRF baseline:

- **`bge-small-en-v1.5`**, with and without the query prefix. Re-tested against
  the expanded question set and the larger synonym map and still lost:
  recall@6 0.655 vs 0.665, hit@6 0.931 vs 0.966.
- **Cross-encoder reranking** with `ms-marco-MiniLM-L-6-v2` and
  `bge-reranker-base`.

The pattern: models trained on modern English misjudge KJV text. A candidate
worth trying would need to be fine-tuned on archaic English, or chosen for it.
`EMBED_MODEL` and `RERANK_MODEL` are env-configurable so repeating this is cheap.

### LLM query rewriting (2026-07)

The synonym map only covers gaps someone thought to add, so asking a model for
the KJV's own wording ought to generalise better. It doesn't.

| Variant | recall@6 | hit@6 |
|---|---|---|
| Baseline (synonym map only) | **0.780** | **0.985** |
| Rewrite replaces the map — `llama3.2` | 0.492 | 0.742 |
| Rewrite replaces the map — `gemini-2.5-flash-lite` | 0.525 | 0.727 |
| Map **+** rewrite (additive) — `gemini-2.5-flash-lite` | 0.703 | 0.924 |

Two failure modes. A small local model invents archaic-*sounding* prose instead
of recalling real KJV phrases — "what afflictions of the mind are to be borne
with patience" for anxiety, where the KJV says "take no thought" — which poisons
both retrievers. Gemini does produce genuine KJV wording ("Go ye therefore,
teach all nations"), and additive expansion recovers most of the gap, but the
generated terms still dilute BM25 and pull the query embedding off the curated
centroid.

`QUERY_REWRITE` stays off by default. The code path is kept and tested so the
experiment is cheap to repeat with a better prompt or a KJV-tuned model.

---

## What has worked

**Growing the synonym map.** `app/retrieval.py:85-139` maps modern vocabulary
onto the KJV's own words ("anxiety" → "take no thought", "Holy Spirit" → "Holy
Ghost"). Expanding it moved recall@6 from 0.677 to 0.750 — the single largest
gain measured so far, and it's a hand-edited dict anyone can extend.

**Indexing the Berean Standard Bible as a search-only mirror.** Retrieval runs
over both translations; the app always displays KJV. This closes the
archaic-vocabulary gap for phrasings nobody thought to add to the map.

**A per-chapter diversity cap.** Without it, thematic questions returned six
chunks from one chapter. `MAX_PER_CHAPTER` keeps results spread across the canon.

---

## Not yet measured

**`CROSSREF_EXPAND`** (`app/retrieval.py:354`). Pulls each top passage's
cross-referenced verses in as extra candidates, weighted at 0.5. The
cross-references are human-curated for exactly the question "what else speaks to
this," so the idea is sound — but it has never been run against the golden set.
It stays off until it wins:

```bash
CROSSREF_EXPAND=1 python scripts/eval_retrieval.py
```

If it beats 0.780 recall@6, flip the default and add a row above. If it loses,
add a row anyway.

---

## Ideas not yet tried

- Indexing at two granularities (5-verse windows *and* whole chapters) so
  narrative questions can match a story arc rather than a fragment.
- Fine-tuning a small embedding model on KJV text.
- Query-side ensembling: run the raw question and the expanded question as
  separate queries and fuse both result sets.
