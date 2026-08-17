import json
import pickle
import os
from xmlrpc import client
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FilterSelector
from rank_bm25 import BM25Okapi

load_dotenv()

QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE     = 384


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_chunks():
    with open("../data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from chunks.json")
    return chunks


# ─────────────────────────────────────────────
# BM25 INDEX
# ─────────────────────────────────────────────
def build_bm25(chunks):
    print("\n── Building BM25 index ──")

    # Index the full text of each chunk
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    with open("../data/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print(f"✓ BM25 index saved → ../data/bm25_index.pkl")
    return bm25


# ─────────────────────────────────────────────
# DENSE INDEX
# ─────────────────────────────────────────────
def build_dense(chunks):
    print("\n── Building dense index ──")

    # ── Embed ──
    print(f"Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # ── Connect to Qdrant ──
    print(f"\nConnecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60, check_compatibility=False)

    # ── Create collection ──
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' exists — deleting and recreating")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"✓ Created collection '{COLLECTION_NAME}'")

    # Create payload indexes so we can filter by section and ticker at query time
    from qdrant_client.models import PayloadSchemaType
    client.create_payload_index(COLLECTION_NAME, "section", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "ticker",  PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "company", PayloadSchemaType.KEYWORD)
    print("✓ Payload indexes created for section, ticker, company")   
    # ── Upload in batches ──
    # KEY DIFFERENCE from previous project:
    # We store section + company + ticker in the payload.
    # This lets the router do filtered retrieval:
    # "only search risk_factors chunks for Apple"
    batch_size = 50
    total = len(chunks)

    print(f"\nUploading {total} vectors...")

    for i in range(0, total, batch_size):
        batch_chunks     = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        points = [
            PointStruct(
                id=i + idx,
                vector=emb.tolist(),
                payload={
                    "chunk_id":   chunk["id"],
                    "company":    chunk["company"],
                    "ticker":     chunk["ticker"],
                    "section":    chunk["section"],
                    "text":       chunk["text"],
                    "position":   chunk["position"],
                    "word_count": chunk["word_count"]
                }
            )
            for idx, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings))
        ]

        for attempt in range(3):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt + 1}/3 on batch {i}...")
                    import time; time.sleep(5)
                else:
                    raise

        uploaded = min(i + batch_size, total)
        if uploaded % 500 == 0 or uploaded == total:
            print(f"  {uploaded}/{total} uploaded")

    print(f"✓ All vectors uploaded to '{COLLECTION_NAME}'")
    return client


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
def sanity_check(chunks, bm25, client):
    print("\n── Sanity check ──")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Test 1: BM25 unfiltered ──
    query = "Apple revenue net sales"
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = int(np.argsort(scores)[::-1][0])
    print(f"\nBM25 top result for '{query}':")
    print(f"  [{chunks[top_idx]['ticker']}][{chunks[top_idx]['section']}] "
          f"{chunks[top_idx]['text'][:120]}...")

    # ── Test 2: Dense filtered by section ──
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    query2 = "cybersecurity risks data breach"
    vec = model.encode(query2).tolist()

    # Filter to only risk_factors chunks — this is what the router will do
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        query_filter=Filter(
            must=[FieldCondition(key="section", match=MatchValue(value="risk_factors"))]
        ),
        limit=3
    ).points

    print(f"\nDense (risk_factors only) top result for '{query2}':")
    if results:
        r = results[0]
        print(f"  [{r.payload['ticker']}][{r.payload['section']}] "
              f"{r.payload['text'][:120]}...")
    else:
        print("  No results — section filter may be too narrow")

    # ── Test 3: Filter by company ──
    query3 = "revenue growth operating income"
    vec3 = model.encode(query3).tolist()

    results3 = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec3,
        query_filter=Filter(
            must=[
                FieldCondition(key="ticker",  match=MatchValue(value="NVDA")),
                FieldCondition(key="section", match=MatchValue(value="mda"))
            ]
        ),
        limit=3
    ).points

    print(f"\nDense (NVDA mda only) top result for '{query3}':")
    if results3:
        r = results3[0]
        print(f"  [{r.payload['ticker']}][{r.payload['section']}] "
              f"{r.payload['text'][:120]}...")
    else:
        print("  No results — try a different section filter")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    chunks = load_chunks()
    bm25   = build_bm25(chunks)
    client = build_dense(chunks)
    sanity_check(chunks, bm25, client)

    print("\n✓ Phase 2 complete.")
    print("  BM25 index  → data/bm25_index.pkl")
    print(f"  Dense index → Qdrant (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    main()