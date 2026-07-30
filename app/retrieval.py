"""
Retrieval layer: embeds a query and pulls the most relevant Bible passages
from the local Chroma index. Also exposes hooks for cross-references,
commentary, and Strong's data if the user has populated resources/.
"""
import json
import os
import re
import sys
from functools import lru_cache

import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.join(DATA_DIR, "chroma_index")
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "..", "resources")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# Some embedding models score short queries better with an instruction
# prefix on the query side only (documents are embedded bare). Matched by
# substring against EMBED_MODEL.
_QUERY_PREFIXES = {
    "bge-": "Represent this sentence for searching relevant passages: ",
}


def _query_prefix():
    override = os.environ.get("EMBED_QUERY_PREFIX")
    if override is not None:
        return override
    return next((p for name, p in _QUERY_PREFIXES.items() if name in EMBED_MODEL), "")


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


# Optional cross-encoder that re-scores the fused candidate pool by reading
# query and passage together. Off by default: on the golden set, every
# candidate tried (ms-marco-MiniLM-L-6-v2, bge-reranker-base) scored at or
# below the plain RRF fusion — modern-English rerankers misjudge KJV text.
# Scoring the BSB (modern) rendering of each chunk instead (2026-07, see
# _modern_chunk_text) also lost clearly: recall@6 0.557 vs 0.677 without.
# Set RERANK_MODEL to experiment.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")


@lru_cache(maxsize=1)
def get_reranker():
    if not RERANK_MODEL:
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def _chroma_client():
    return chromadb.PersistentClient(path=INDEX_DIR)


@lru_cache(maxsize=1)
def get_collection():
    return _chroma_client().get_collection("bible_kjv")


@lru_cache(maxsize=1)
def get_modern_collection():
    """Modern-English (BSB) mirror of the index — retrieval only, the app
    always displays KJV. None when the index was built without it."""
    try:
        return _chroma_client().get_collection("bible_modern")
    except Exception:
        return None


# ── Hybrid search: dense vectors + BM25, fused with RRF ─────────────────
# The embedding model was trained on modern English while the KJV says
# "charity" and "Holy Ghost"; BM25 catches exact archaic terms the
# embeddings fuzz over, and a small synonym map bridges the most common
# modern-vocabulary gaps in both directions.

_KJV_SYNONYMS = {
    "holy spirit": "holy ghost comforter",
    "love": "charity",
    "anxiety": "take no thought careful for nothing casting your care",
    "anxious": "take no thought careful",
    "worry": "take no thought careful for nothing casting your care",
    "worrying": "take no thought careful",
    "forgiveness": "forgiving one another trespasses tenderhearted",
    "pride": "pride goeth before destruction haughty spirit",
    "humility": "resisteth the proud grace unto the humble",
    "greed": "mammon treasures upon earth covetousness content",
    "created": "in the beginning was the word all things made",
    "resurrection of the dead": "i am the resurrection dead in christ shall rise",
    "gifts of the spirit": "gifts differing according to grace prophecy ministry",
    # Named concepts whose KJV wording never uses the modern name.
    "fruits of the spirit": "fruit of the spirit love joy peace",
    "beatitudes": "blessed are the poor in spirit",
    "prodigal": "younger son wasted his substance riotous living",
    "ten commandments": "thou shalt have no other gods graven image",
    "goliath": "sling stone philistine smote",
    "great commission": "go ye therefore teach all nations into all the world preach the gospel",
    "red sea": "waters divided dry ground",
    "resurrection of jesus": "he is risen sepulchre first day of the week mary magdalene",
    "second coming": "clouds trumpet archangel descend",
    "disciples to pray": "our father which art in heaven hallowed",
    "lord's prayer": "our father which art in heaven hallowed",
    "when i sin": "confess our sins faithful to forgive blot out my transgressions",
    "when we die": "spirit return absent from the body",
    "resist temptation": "resist the devil way to escape temptation taken you",
    "faith versus works": "justified by faith without works boast by grace are ye saved",
    "lord's supper": "take eat this is my body",
    "communion": "take eat this is my body",
    "divorce": "putting away writing of divorcement hateth",
    "waiting on god": "wait upon the lord renew their strength good courage",
    "tithing": "tithes storehouse cheerful giver give and it shall be given",
    # Broader modern terms the KJV renders differently.
    "fear": "fear not dismayed spirit of fear perfect love casteth out",
    "poor": "hath pity upon the poor lendeth unto the lord",
    "tongue": "death and life are in the power of the tongue corrupt communication edifying",
    "wisdom": "if any lack wisdom ask fear of the lord happy is the man that findeth",
    "saved": "believe on the lord jesus christ confess with thy mouth",
    "flood": "waters prevailed ark gopher wood",
    "sabbath": "remember the sabbath day rested seventh day made for man",
    "gossip": "talebearer whisperer separateth chief friends",
    "enemies": "love your enemies bless them that curse avenge not overcome evil with good",
    "marriage": "cleave unto his wife one flesh rib made he a woman",
    "stress": "heavy laden cast thy burden rest casting all your care careful for nothing",
    "depression": "cast down disquieted hope thou in god",
    "despair": "cast down disquieted",
    "suffering": "fiery trial count it all joy tribulation worketh patience light affliction",
    "trials": "temptations fiery trial",
    "baptism": "baptized buried with him repent and be baptized baptizing in the name",
    "hear our prayers": "ask and it shall be given heareth us",
}

_RRF_K = 60          # standard reciprocal-rank-fusion constant
# Candidates pulled from each retriever before fusing. Env-overridable for
# eval experiments.
_CANDIDATES = int(os.environ.get("RETRIEVAL_CANDIDATES", 50))

# Diversity cap: at most this many fused results per (book, chapter), so a
# thematic question spans the canon instead of one strong chapter filling
# every slot. Env-overridable for eval experiments.
MAX_PER_CHAPTER = int(os.environ.get("MAX_PER_CHAPTER", 2))

# Relevance floor: drop fused results scoring below this fraction of the
# leader. Steep score curves (precise questions) lose their noise tail —
# junk passages mislead small models more than they help. Flat curves
# (vague queries) drop nothing. At least MIN_KEEP always survive.
MIN_SCORE_RATIO = float(os.environ.get("MIN_SCORE_RATIO", 0.45))
MIN_KEEP = 3


def _expand_query(query: str):
    low = query.lower()
    extra = [kjv for term, kjv in _KJV_SYNONYMS.items()
             if re.search(rf"\b{re.escape(term)}\b", low) and kjv not in low]
    if extra:
        query = f"{query} ({' '.join(dict.fromkeys(extra))})"
    return query


# The synonym map above only covers gaps someone thought to add. Asking the
# LLM for the KJV's own wording generalizes to questions nobody anticipated,
# at the cost of one generation call before retrieval. Off until it beats the
# golden-set baseline; QUERY_REWRITE=1 to try it.
QUERY_REWRITE = os.environ.get("QUERY_REWRITE", "") == "1"
# The rewrite is a small utility call, so it can run on a different (cheaper
# or stronger) model than the answer. Defaults to the configured provider.
REWRITE_PROVIDER = os.environ.get("REWRITE_PROVIDER") or None
REWRITE_MODEL = os.environ.get("REWRITE_MODEL") or None

_REWRITE_SYSTEM = (
    "You translate modern questions into King James Version vocabulary for a "
    "search index. Reply with search terms only — the archaic words and "
    "phrases the KJV itself uses for the ideas in the question. No "
    "explanation, no numbering, no references, one line."
)


@lru_cache(maxsize=512)
def _rewrite_query(query: str):
    """Synonym-map expansion plus LLM-supplied KJV wording — additive, so the
    curated terms stay as a floor and the model only adds. Falls back to the
    map alone if the provider is unreachable or answers with nothing usable,
    so a dead provider degrades retrieval instead of breaking it."""
    from app.rag import _generate  # local: app.rag imports this module
    expanded = _expand_query(query)
    try:
        reply = _generate(
            [{"role": "user", "content": f"Question: {query}"}],
            provider=REWRITE_PROVIDER, model=REWRITE_MODEL,
            system=_REWRITE_SYSTEM,
        )
        lines = [ln.strip(" -*\t") for ln in reply.strip().splitlines() if ln.strip()]
        terms = " ".join(ln for ln in lines if not ln.endswith(":"))[:300]
        return f"{expanded} ({terms})" if terms else expanded
    except Exception as e:
        # Loud on purpose: a silently degraded rewrite looks like a bad
        # retrieval change when you're measuring one.
        print(f"query rewrite failed ({type(e).__name__}: {e}); "
              f"falling back to the synonym map", file=sys.stderr)
        return expanded


def _tokenize(text: str):
    return re.findall(r"[a-z']+", text.lower())


@lru_cache(maxsize=1)
def _bm25_index():
    """Lexical index over the same chunks as the vector indexes (KJV plus
    the modern mirror when present, so both archaic and modern wording get
    exact-term matches). Built lazily on first search, then cached."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None
    data = get_collection().get()  # ids, documents, metadatas
    modern = get_modern_collection()
    if modern is not None:
        m = modern.get()
        data = {
            "ids": data["ids"] + m["ids"],
            "documents": data["documents"] + m["documents"],
            "metadatas": data["metadatas"] + m["metadatas"],
        }
    return BM25Okapi([_tokenize(d) for d in data["documents"]]), data


def _kjv_text(meta):
    """Rebuild a chunk's display text from the KJV verses it spans, so a
    chunk found via the modern index still renders as KJV."""
    vs, ve = meta.get("verse_start"), meta.get("verse_end")
    if vs is None:
        return None
    verses = _load_all_verses().get((meta["book"], meta["chapter"]), [])
    window = [v for v in verses if vs <= v["verse"] <= ve]
    if not window:
        return None
    return " ".join(f"{v['verse']}. {v['text']}" for v in window)


def _as_passage(doc, meta):
    return {
        "reference": meta["reference"],
        "text": _kjv_text(meta) or doc,
        "book": meta["book"],
        "chapter": meta["chapter"],
        "verse_start": meta.get("verse_start"),
        "verse_end": meta.get("verse_end"),
    }


def search_passages(query: str, top_k: int = 6):
    """Hybrid semantic + lexical search. Returns a list of dicts:
    reference, text, book, chapter, verse_start, verse_end."""
    raw_query = query  # the reranker reads the query without synonym padding
    query = _rewrite_query(query) if QUERY_REWRITE else _expand_query(query)
    model = get_embedder()
    collection = get_collection()

    query_embedding = model.encode(
        [_query_prefix() + query], normalize_embeddings=True
    )[0].tolist()

    # id -> [rrf_score, doc, meta]; the same chunk found by several
    # retrievers accumulates contributions.
    fused = {}

    def fuse_ranking(ids, docs, metas):
        for rank, (id_, doc, meta) in enumerate(zip(ids, docs, metas)):
            contribution = 1.0 / (_RRF_K + rank + 1)
            if id_ in fused:
                fused[id_][0] += contribution
            else:
                fused[id_] = [contribution, doc, meta]

    dense_collections = [collection]
    modern = get_modern_collection()
    if modern is not None:
        dense_collections.append(modern)
    for coll in dense_collections:
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, _CANDIDATES),
        )
        fuse_ranking(results["ids"][0], results["documents"][0], results["metadatas"][0])

    bm = _bm25_index()
    if bm is not None:
        bm25, data = bm
        scores = bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(scores)), key=scores.__getitem__,
                         reverse=True)[:_CANDIDATES]
        for rank, i in enumerate(top_idx):
            if scores[i] <= 0:
                break
            contribution = 1.0 / (_RRF_K + rank + 1)
            id_ = data["ids"][i]
            if id_ in fused:
                fused[id_][0] += contribution
            else:
                fused[id_] = [contribution, data["documents"][i], data["metadatas"][i]]

    ranked = sorted(fused.values(), key=lambda x: -x[0])
    ranked = _crossref_expand(ranked)
    ranked = _rerank(raw_query, _apply_floor(ranked)[:_CANDIDATES])
    return [_as_passage(doc, meta) for _, doc, meta in _diversify(ranked, top_k)]


def _apply_floor(ranked):
    """Drop results scoring below MIN_SCORE_RATIO of the leader, keeping at
    least MIN_KEEP. Eval-verified recall-neutral: it only sheds the tail."""
    if not ranked:
        return ranked
    floor = ranked[0][0] * MIN_SCORE_RATIO
    kept = [r for r in ranked if r[0] >= floor]
    return kept if len(kept) >= MIN_KEEP else ranked[:MIN_KEEP]


def _modern_chunk_text(meta):
    """BSB wording for a chunk's verse range. Rerankers are trained on
    modern English and misjudge KJV text, so when a reranker is enabled it
    scores the modern rendering instead. None when BSB isn't on disk."""
    vs, ve = meta.get("verse_start"), meta.get("verse_end")
    if vs is None:
        return None
    verses = _load_bsb_verses().get((meta["book"], meta["chapter"]), [])
    window = [v["text"] for v in verses if vs <= v["verse"] <= ve]
    return " ".join(window) or None


def _rerank(query, candidates):
    """Re-order the fused pool by cross-encoder relevance; RRF picks the
    pool, the reranker orders it. No-op when reranking is disabled."""
    reranker = get_reranker()
    if reranker is None or not candidates:
        return candidates
    pairs = [(query, _modern_chunk_text(meta) or doc)
             for _, doc, meta in candidates]
    scores = reranker.predict(pairs)
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order]


# Cross-reference expansion: verses cross-referenced from the top fused
# hits join the candidate pool at a fraction of their parent's score.
# Off by default until it wins the golden-set eval; CROSSREF_EXPAND=1 to try.
CROSSREF_EXPAND = os.environ.get("CROSSREF_EXPAND", "") == "1"
_CROSSREF_WEIGHT = 0.5   # child score = parent score × this
_CROSSREF_PARENTS = 3    # expand only this many top hits

_REF_STRING_RE = re.compile(r"^(.+?)\s+(\d+):(\d+)(?:-(\d+))?$")


def _ref_to_candidate(ref: str):
    """'Romans 8:28' or 'Luke 15:20-24' -> (doc, meta) chunk, or None."""
    m = _REF_STRING_RE.match(ref.strip())
    if not m:
        return None
    book = canonical_book(m.group(1))
    chapter, v1 = int(m.group(2)), int(m.group(3))
    v2 = int(m.group(4) or v1)
    verses = [v for v in _load_all_verses().get((book, chapter), [])
              if v1 <= v["verse"] <= v2]
    if not verses:
        return None
    meta = {
        "reference": f"{book} {chapter}:{v1}" + (f"-{v2}" if v2 != v1 else ""),
        "book": book, "chapter": chapter, "verse_start": v1, "verse_end": v2,
    }
    return " ".join(f"{v['verse']}. {v['text']}" for v in verses), meta


def _covered(cmeta, metas):
    for m in metas:
        if m["book"] != cmeta["book"] or m["chapter"] != cmeta["chapter"]:
            continue
        vs, ve = m.get("verse_start"), m.get("verse_end")
        if vs is None or not (ve < cmeta["verse_start"] or cmeta["verse_end"] < vs):
            return True
    return False


def _crossref_expand(ranked):
    """No-op unless CROSSREF_EXPAND=1 and cross-reference data is on disk."""
    if not CROSSREF_EXPAND or not ranked:
        return ranked
    existing = [meta for _, _, meta in ranked]
    added = []
    for score, _, meta in ranked[:_CROSSREF_PARENTS]:
        for ref in passage_cross_references(meta):
            cand = _ref_to_candidate(ref)
            if cand is None or _covered(cand[1], existing):
                continue
            existing.append(cand[1])
            added.append([score * _CROSSREF_WEIGHT, cand[0], cand[1]])
    return sorted(ranked + added, key=lambda x: -x[0]) if added else ranked


def _diversify(ranked, top_k):
    """Greedy pick over the fused ranking with a per-chapter cap. Backfills
    from the skipped chunks (in score order) if there aren't enough distinct
    chapters, so a narrow question about one chapter still fills top_k."""
    picked, skipped, counts = [], [], {}
    for item in ranked:
        meta = item[2]
        key = (meta["book"], meta["chapter"])
        if counts.get(key, 0) >= MAX_PER_CHAPTER:
            skipped.append(item)
            continue
        counts[key] = counts.get(key, 0) + 1
        picked.append(item)
        if len(picked) == top_k:
            return picked
    return picked + skipped[:top_k - len(picked)]


VERSES_PATH = os.path.join(DATA_DIR, "kjv_verses.json")


@lru_cache(maxsize=1)
def _load_all_verses():
    """Load the full KJV verse list once and index it by (book, chapter).
    Empty on a fresh clone (before scripts/download_bible.py), so callers
    get clean empty results instead of a crash."""
    if not os.path.exists(VERSES_PATH):
        return {}
    with open(VERSES_PATH, encoding="utf-8") as f:
        verses = json.load(f)
    by_chapter = {}
    for v in verses:
        by_chapter.setdefault((v["book"], v["chapter"]), []).append(v)
    for vlist in by_chapter.values():
        vlist.sort(key=lambda v: v["verse"])
    return by_chapter


def canonical_book(book: str) -> str:
    """'psalms', 'PSALMS', or 'psalm' -> 'Psalms'; unknown names pass through."""
    _, names = _reference_pattern()
    return names.get(book.strip().lower(), book)


# A 5-verse chunk can cut a narrative off mid-story (1 Kings 3:16-20 ends
# before Solomon's ruling), leaving the LLM to invent the ending. Top hits
# are widened to their surrounding chapter before they reach the prompt.
EXPAND_MAX_VERSES = int(os.environ.get("EXPAND_MAX_VERSES", 40))


def expanded_text(passage: dict, max_verses: int = None):
    """(text, reference, verse_count) for the passage widened to its
    surrounding chapter, capped at max_verses centered on the hit. Falls
    back to the passage's own text when it can't be located."""
    if max_verses is None:
        max_verses = EXPAND_MAX_VERSES
    verses = _load_all_verses().get((passage["book"], passage["chapter"]), [])
    vs, ve = passage.get("verse_start"), passage.get("verse_end")
    if not verses or vs is None:
        return passage["text"], passage["reference"], 0
    if len(verses) <= max_verses:
        window = verses
    else:
        span = ve - vs + 1
        pad = max(0, (max_verses - span) // 2)
        start = next((i for i, v in enumerate(verses) if v["verse"] >= vs), 0)
        start = min(max(0, start - pad), len(verses) - max_verses)
        window = verses[start:start + max_verses]
    text = " ".join(f"{v['verse']}. {v['text']}" for v in window)
    if len(window) == len(verses):
        ref = f"{passage['book']} {passage['chapter']}"
    else:
        ref = f"{passage['book']} {passage['chapter']}:{window[0]['verse']}-{window[-1]['verse']}"
    return text, ref, len(window)


def get_chapter(book: str, chapter: int):
    """Return the ordered verses (verse, text) for a given book + chapter.
    Book name matching is case-insensitive."""
    vlist = _load_all_verses().get((canonical_book(book), chapter), [])
    return [{"verse": v["verse"], "text": v["text"]} for v in vlist]


BSB_VERSES_PATH = os.path.join(DATA_DIR, "bsb_verses.json")


@lru_cache(maxsize=1)
def _load_bsb_verses():
    """BSB verses indexed by (book, chapter), like _load_all_verses. The BSB
    file is optional on disk; empty dict when it hasn't been downloaded."""
    if not os.path.exists(BSB_VERSES_PATH):
        return {}
    with open(BSB_VERSES_PATH, encoding="utf-8") as f:
        verses = json.load(f)
    by_chapter = {}
    for v in verses:
        by_chapter.setdefault((v["book"], v["chapter"]), []).append(v)
    for vlist in by_chapter.values():
        vlist.sort(key=lambda v: v["verse"])
    return by_chapter


def has_bsb() -> bool:
    return bool(_load_bsb_verses())


def get_passage_text(translation: str, book: str, chapter: int,
                     verse_start: int = None, verse_end: int = None):
    """Ordered verses for one passage in the given translation ("kjv" or
    "bsb"), whole chapter when the range is omitted. [] when unknown."""
    verse_map = _load_bsb_verses() if translation == "bsb" else _load_all_verses()
    vlist = verse_map.get((canonical_book(book), chapter), [])
    if verse_start is not None:
        end = verse_end if verse_end is not None else verse_start
        vlist = [v for v in vlist if verse_start <= v["verse"] <= end]
    return [{"verse": v["verse"], "text": v["text"]} for v in vlist]


# ── Reference-aware retrieval ────────────────────────────────────────────
# Semantic search alone can miss an explicitly named passage ("What does
# Romans 8 say about the Spirit?" isn't guaranteed to retrieve Romans 8).
# So references written in the question are parsed out and their text is
# always included, with vector search filling the remaining slots.

MAX_REF_PASSAGES = 2       # at most this many direct references per query
MAX_CHAPTER_VERSES = 60    # cap whole-chapter references (Psalm 119 is 176)


@lru_cache(maxsize=1)
def _reference_pattern():
    """Regex matching 'Book Chapter', 'Book Chapter:Verse', and
    'Book Chapter:Start-End', built from the book names in the data."""
    canonical = {b.lower(): b for b, _ in _load_all_verses().keys()}
    if not canonical:  # no Bible data yet — a pattern that never matches
        return re.compile(r"(?!x)x"), {}
    canonical["psalm"] = "Psalms"  # people usually write "Psalm 23"
    # Longest first so "1 John" wins over "John", "Psalms" over "Psalm".
    alts = "|".join(re.escape(n) for n in sorted(canonical, key=len, reverse=True))
    pattern = re.compile(
        rf"\b({alts})\s+(\d{{1,3}})(?:\s*:\s*(\d{{1,3}})(?:\s*[-–—]\s*(\d{{1,3}}))?)?",
        re.IGNORECASE,
    )
    return pattern, canonical


def reference_passages(query: str):
    """Passages for scripture references written directly in the query."""
    pattern, canonical = _reference_pattern()
    passages = []
    seen = set()
    for m in pattern.finditer(query):
        book = canonical[m.group(1).lower()]
        chapter = int(m.group(2))
        verses = _load_all_verses().get((book, chapter))
        if not verses:
            continue
        if m.group(3):
            v_start = int(m.group(3))
            v_end = int(m.group(4)) if m.group(4) else v_start
            if v_end < v_start:
                v_end = v_start
            window = [v for v in verses if v_start <= v["verse"] <= v_end]
        else:
            window = verses[:MAX_CHAPTER_VERSES]
        if not window:
            continue
        key = (book, chapter, window[0]["verse"], window[-1]["verse"])
        if key in seen:
            continue
        seen.add(key)
        whole_chapter = len(window) == len(verses)
        if whole_chapter:
            ref = f"{book} {chapter}"
        elif len(window) > 1:
            ref = f"{book} {chapter}:{window[0]['verse']}-{window[-1]['verse']}"
        else:
            ref = f"{book} {chapter}:{window[0]['verse']}"
        passages.append({
            "reference": ref,
            "text": " ".join(f"{v['verse']}. {v['text']}" for v in window),
            "book": book,
            "chapter": chapter,
            "verse_start": window[0]["verse"],
            "verse_end": window[-1]["verse"],
        })
        if len(passages) >= MAX_REF_PASSAGES:
            break
    return passages


def _overlaps(a, b):
    if a["book"] != b["book"] or a["chapter"] != b["chapter"]:
        return False
    return not (a["verse_end"] < b["verse_start"] or b["verse_end"] < a["verse_start"])


def retrieve(query: str, top_k: int = 6):
    """Direct references from the query first, then semantic results
    (skipping any that overlap a direct reference) up to top_k total."""
    direct = reference_passages(query)
    results = list(direct)
    for p in search_passages(query, top_k=top_k):
        if len(results) >= top_k:
            break
        if any(_overlaps(p, d) for d in direct):
            continue
        results.append(p)
    for p in results:
        p["cross_references"] = passage_cross_references(p)
    return results


@lru_cache(maxsize=1)
def _cross_reference_data():
    """Cross-references from resources/cross_references/cross_references.json
    in the format { "Book Chapter:Verse": ["Book Chapter:Verse", ...], ... }.
    Generate it with scripts/fetch_cross_references.py, or supply your own.
    Loaded once; empty dict if the file isn't there."""
    path = os.path.join(RESOURCES_DIR, "cross_references", "cross_references.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cross_references(book: str, chapter: int, verse: int):
    return _cross_reference_data().get(f"{book} {chapter}:{verse}", [])


def passage_cross_references(passage: dict, limit: int = 6):
    """Deduped cross-references for every verse a passage spans."""
    data = _cross_reference_data()
    if not data or passage.get("verse_start") is None:
        return []
    refs = []
    for verse in range(passage["verse_start"], passage["verse_end"] + 1):
        for ref in data.get(f"{passage['book']} {passage['chapter']}:{verse}", []):
            if ref not in refs:
                refs.append(ref)
            if len(refs) >= limit:
                return refs
    return refs


def load_commentary(book: str, chapter: int):
    """
    Optional: looks up commentary if the user has placed files at
    resources/commentary/<Book>.json in the format:
    { "1": "commentary text for chapter 1", "2": "...", ... }

    Source suggestion (public domain): Matthew Henry's Concise Commentary,
    available via ccel.org or bundled in the scrollmapper repo's commentary
    formats.
    """
    path = os.path.join(RESOURCES_DIR, "commentary", f"{book}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(str(chapter))


# Strong's dictionary lookups live in app/word_study.py, which reads
# resources/strongs/strongs.json (fetched by scripts/fetch_strongs.py).
