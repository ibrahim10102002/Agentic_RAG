import json
import pickle
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

load_dotenv()

QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_TOP_K  = 20
BM25_TOP_K   = 20
RERANK_TOP_N = 20
FINAL_TOP_K  = 5


# ─────────────────────────────────────────────
# LOAD COMPONENTS (once at startup)
# ─────────────────────────────────────────────
def load_retriever_components():
    print("Loading retriever components...")

    with open("../data/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open("../data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    bi_encoder    = SentenceTransformer(EMBEDDING_MODEL)
    cross_encoder = CrossEncoder(RERANKER_MODEL)
    qdrant        = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        check_compatibility=False
    )

    print("✓ Retriever components loaded\n")
    return {
        "bm25":          bm25,
        "chunks":        chunks,
        "bi_encoder":    bi_encoder,
        "cross_encoder": cross_encoder,
        "qdrant":        qdrant
    }


# ─────────────────────────────────────────────
# BUILD QDRANT FILTER FROM ROUTE RESULT
# ─────────────────────────────────────────────
def build_filter(route_result: dict):
    """
    Converts the router's output into a Qdrant filter.

    The router tells us:
      sections:  ["risk_factors"] or ["mda", "financials"] etc.
      companies: ["AAPL"] or ["AAPL", "MSFT"] or [] (all companies)

    Qdrant filter logic:
      - sections  → MatchAny (chunk must be in ONE of the listed sections)
      - companies → MatchAny (chunk must be from ONE of the listed companies)
      - Both conditions must be satisfied simultaneously (must=[...])
      - If companies is empty → no company filter, search all companies

    This is the key difference from the previous project.
    Instead of searching 808 chunks blindly, we might search
    only the 36 risk_factors chunks for Apple — dramatically
    improving precision.
    """
    must_conditions = []

    sections  = route_result.get("sections",  [])
    companies = route_result.get("companies", [])

    if sections:
        must_conditions.append(
            FieldCondition(key="section", match=MatchAny(any=sections))
        )

    if companies:
        must_conditions.append(
            FieldCondition(key="ticker", match=MatchAny(any=companies))
        )

    if must_conditions:
        return Filter(must=must_conditions)

    return None  # No filter — search everything


# ─────────────────────────────────────────────
# DENSE RETRIEVAL (with filter)
# ─────────────────────────────────────────────
def dense_search(query, bi_encoder, qdrant, qdrant_filter=None, top_k=DENSE_TOP_K):
    query_vec = bi_encoder.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=qdrant_filter,
        limit=top_k
    ).points

    chunks_found = []
    for r in results:
        chunk = dict(r.payload)
        chunk["dense_score"] = r.score
        chunks_found.append(chunk)

    return chunks_found


# ─────────────────────────────────────────────
# BM25 RETRIEVAL (with filter)
# ─────────────────────────────────────────────
def bm25_search(query, bm25, chunks, route_result, top_k=BM25_TOP_K):
    """
    BM25 doesn't natively support metadata filtering the way Qdrant does.
    So we filter the chunk list before scoring:
      1. Filter chunks by section and company
      2. Score only the filtered subset
      3. Return top_k by score

    This mirrors what the dense filter does — both retrievers
    search the same logical subset of chunks.
    """
    sections  = route_result.get("sections",  [])
    companies = route_result.get("companies", [])

    # Filter chunks list
    filtered_chunks   = []
    filtered_indices  = []

    for i, chunk in enumerate(chunks):
        section_ok  = (not sections)  or (chunk["section"] in sections)
        company_ok  = (not companies) or (chunk["ticker"]  in companies)
        if section_ok and company_ok:
            filtered_chunks.append(chunk)
            filtered_indices.append(i)

    if not filtered_chunks:
        # Fallback — no chunks match the filter, search everything
        filtered_chunks  = chunks
        filtered_indices = list(range(len(chunks)))

    # Score filtered chunks using their original BM25 positions
    tokens = query.lower().split()
    all_scores = bm25.get_scores(tokens)
    filtered_scores = np.array([all_scores[i] for i in filtered_indices])

    top_local_indices = np.argsort(filtered_scores)[::-1][:top_k]

    results = []
    for local_idx in top_local_indices:
        chunk = dict(filtered_chunks[local_idx])
        chunk["bm25_score"] = float(filtered_scores[local_idx])
        results.append(chunk)

    return results


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    rrf_scores = {}
    chunk_map  = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid]  = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk.get("chunk_id") or chunk.get("id")
        if not cid:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid]  = chunk

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    fused = []
    for cid in sorted_ids[:RERANK_TOP_N]:
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        fused.append(chunk)

    return fused


# ─────────────────────────────────────────────
# CROSS-ENCODER RERANKING
# ─────────────────────────────────────────────
def rerank(query, candidates, cross_encoder, top_k=FINAL_TOP_K):
    if not candidates:
        return []

    pairs  = [(query, chunk["text"]) for chunk in candidates]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


# ─────────────────────────────────────────────
# MAIN RETRIEVE FUNCTION
# ─────────────────────────────────────────────
def retrieve(query: str, route_result: dict, components: dict) -> dict:
    """
    Full hybrid retrieval pipeline with metadata filtering.

    Returns a dict with:
      chunks:       top-5 reranked chunks
      dense_count:  how many dense results came back
      bm25_count:   how many BM25 results came back
      fused_count:  unique chunks after RRF
      filter_used:  what filter was applied
    """
    qdrant_filter = build_filter(route_result)

    sections  = route_result.get("sections",  [])
    companies = route_result.get("companies", [])

    filter_desc = []
    if sections:  filter_desc.append(f"sections={sections}")
    if companies: filter_desc.append(f"companies={companies}")
    filter_str = ", ".join(filter_desc) if filter_desc else "none (all chunks)"

    # Run both retrievers
    dense_results = dense_search(
        query,
        components["bi_encoder"],
        components["qdrant"],
        qdrant_filter=qdrant_filter
    )

    bm25_results = bm25_search(
        query,
        components["bm25"],
        components["chunks"],
        route_result
    )

    # Fuse
    fused = reciprocal_rank_fusion(dense_results, bm25_results)

    # Rerank
    final = rerank(query, fused, components["cross_encoder"])

    return {
        "chunks":       final,
        "dense_count":  len(dense_results),
        "bm25_count":   len(bm25_results),
        "fused_count":  len(fused),
        "filter_used":  filter_str,
        "query_used":   query
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from router import route

    components = load_retriever_components()

    test_cases = [
        ("What are Apple's cybersecurity risks?",     {"sections": ["risk_factors"], "companies": ["AAPL"]}),
        ("How did Nvidia's revenue grow?",            {"sections": ["mda", "financials"], "companies": ["NVDA"]}),
        ("What products does Microsoft sell?",        {"sections": ["business"], "companies": ["MSFT"]}),
        ("Compare risks across semiconductor firms",  {"sections": ["risk_factors"], "companies": []}),
    ]

    for query, mock_route in test_cases:
        print(f"\nQuery: '{query}'")
        print(f"Filter: sections={mock_route['sections']}, companies={mock_route['companies']}")

        result = retrieve(query, mock_route, components)

        print(f"  Dense: {result['dense_count']} | "
              f"BM25: {result['bm25_count']} | "
              f"Fused: {result['fused_count']} | "
              f"Final: {len(result['chunks'])}")

        if result["chunks"]:
            top = result["chunks"][0]
            print(f"  Top chunk: [{top['ticker']}][{top['section']}] "
                  f"(rerank={top['rerank_score']:+.2f})")
            print(f"  Text: {top['text'][:120]}...")