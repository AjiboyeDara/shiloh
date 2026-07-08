"""
Builds a local vector index over the Bible text for retrieval-augmented
generation (RAG). Uses a small local sentence-transformer model (no API key
or internet call needed at query time) and a persistent Chroma collection.

Chunking strategy: group verses into overlapping windows of CHUNK_SIZE
verses within the same chapter, so retrieval returns coherent passages
rather than single disconnected verses.
"""
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VERSES_PATH = os.path.join(DATA_DIR, "kjv_verses.json")
INDEX_DIR = os.path.join(DATA_DIR, "chroma_index")

CHUNK_SIZE = 5      # verses per chunk
CHUNK_STRIDE = 3    # overlap between consecutive chunks
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_verses():
    with open(VERSES_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_chunks(verses):
    """Group consecutive verses (same book+chapter) into overlapping chunks."""
    chunks = []
    by_chapter = {}
    for v in verses:
        key = (v["book"], v["chapter"])
        by_chapter.setdefault(key, []).append(v)

    for (book, chapter), vlist in by_chapter.items():
        vlist.sort(key=lambda v: v["verse"])
        i = 0
        while i < len(vlist):
            window = vlist[i:i + CHUNK_SIZE]
            if not window:
                break
            text = " ".join(f"{v['verse']}. {v['text']}" for v in window)
            ref = f"{book} {chapter}:{window[0]['verse']}-{window[-1]['verse']}" \
                if len(window) > 1 else f"{book} {chapter}:{window[0]['verse']}"
            chunks.append({
                "id": f"{book}-{chapter}-{window[0]['verse']}-{window[-1]['verse']}",
                "text": text,
                "book": book,
                "chapter": chapter,
                "verse_start": window[0]["verse"],
                "verse_end": window[-1]["verse"],
                "reference": ref,
            })
            if len(window) < CHUNK_SIZE:
                break
            i += CHUNK_STRIDE
    return chunks


def build():
    print("Loading verses...")
    verses = load_verses()
    print(f"Loaded {len(verses)} verses.")

    print("Chunking...")
    chunks = make_chunks(verses)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBED_MODEL})... this downloads once.")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding chunks (this can take a few minutes)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    print("Writing to Chroma index...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=INDEX_DIR)
    # Fresh collection each build
    try:
        client.delete_collection("bible_kjv")
    except Exception:
        pass
    collection = client.create_collection("bible_kjv")

    batch = 500
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in batch_chunks],
            embeddings=embeddings[i:i + batch].tolist(),
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{
                "book": c["book"],
                "chapter": c["chapter"],
                "verse_start": c["verse_start"],
                "verse_end": c["verse_end"],
                "reference": c["reference"],
            } for c in batch_chunks],
        )
        print(f"  indexed {min(i + batch, len(chunks))}/{len(chunks)}")

    print(f"Done. Index stored at {INDEX_DIR}")


if __name__ == "__main__":
    build()
