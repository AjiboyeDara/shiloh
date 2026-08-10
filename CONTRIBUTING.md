# Contributing to Shiloh

Thanks for looking. Shiloh is a small project with a specific goal: help people
study scripture with a tool that shows its work. Changes that make it more
honest about what it retrieved and what it quoted are always welcome.

## Getting set up

Follow the Quickstart in the README. You need `python scripts/setup.py` to have
finished — without the index, most of the test suite skips rather than fails.

```bash
python -m pytest        # should be all green, no skips, once setup has run
```

## The frontend has no build step

`frontend/index.html` is one file containing the markup, the CSS, and the
JavaScript. There is no bundler, no `package.json`, no compile step. Edit it and
refresh the page. Please keep it that way — being readable and hackable by one
person in one file is a feature, not an accident.

There's no JavaScript test runner either. Changes to the frontend need driving in
a browser by hand. Worth checking after any change to the chat flow:

- ask a question, watch it stream, confirm the passages pane fills and `[n]`
  citations resolve
- scroll up mid-answer — the view should stay where you put it
- reload — the conversation should come back intact
- star a passage, check the Saved tab, reload again
- narrow the window under 860px and confirm the passages pane becomes a sheet

## Changing retrieval

**Measure it.** Retrieval sets the ceiling on every answer, so it doesn't change
on vibes.

```bash
python scripts/eval_retrieval.py    # recall@k, hit@k, chapter diversity
python scripts/eval_answers.py      # citations, quote verification, groundedness
```

Run both before and after. Put the numbers in the PR. If your change loses, say
so and add it to the log in [docs/RETRIEVAL.md](docs/RETRIEVAL.md) — recording
failures is genuinely useful, and it's why that file exists.

## The API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat/stream` | SSE: a `passages` event, then `delta` events, then `done`. Errors mid-stream arrive as an `error` event, since the 200 is already committed. |
| `POST /api/chat` | Same, blocking. |
| `POST /api/search` | Retrieval only, no AI call. Useful for debugging search. |
| `POST /api/plan` | Builds a themed reading plan. |
| `GET /api/chapter` | A whole KJV chapter. |
| `GET /api/passage-text` | A verse range, KJV or BSB. |
| `GET /api/word-study` | Strong's entries and concordance for a word. |
| `GET /api/models` | Which providers are usable right now. |

Interactive docs at `/docs` while the server runs.

## Style

Match what's around your change. A few things the codebase already does that are
worth keeping:

- **Comments explain why, not what.** The valuable ones here record a decision
  and the trap it avoids — the safety filter's list of deliberately excluded
  words, the reranker's measured failure, the reason the citation few-shot can't
  be deleted. Add that kind.
- **Degrade rather than break.** A missing resource file, an unreachable
  provider, or full browser storage should cost a feature, not the app.
- **Small diffs.** The smallest change that actually fixes the problem, in the
  place all the callers route through.

Commit messages are short and lowercase, split by theme.

## Things that need doing

- A wider synonym map in `app/retrieval.py` — measurably the highest-leverage
  retrieval improvement, and it's a plain dict anyone can extend.
- Embedding models that handle archaic English. Several have been tried and
  lost; see [docs/RETRIEVAL.md](docs/RETRIEVAL.md).
- Commentary sources beyond Matthew Henry.
- Localising `CRISIS_MESSAGE` in `app/rag.py` — it currently gives a US crisis
  line.
- Making `frontend/landing.html` use the same CSS custom properties as the app,
  so a rebrand is one place instead of two.

## Data licences

If you add or swap a data source, add it to [NOTICE](NOTICE) with its URL and
licence. The Strong's dictionaries are CC BY-SA, which has real obligations for
anyone redistributing them.
