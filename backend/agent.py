import time
from router       import route
from reformulator import reformulate, generate_retry_queries
from retriever    import retrieve
from confidence   import score_confidence, loosen_filter
from generator    import generate

MAX_RETRIES = 2


def run(query: str, components: dict) -> dict:
    """
    Full agentic RAG pipeline.

    The agent trace records every decision made so the
    frontend can show the user exactly what happened:
      - what the query was classified as
      - whether it was reformulated and why
      - how many retries were needed
      - what the confidence score was at each attempt
      - what filter was used at each attempt

    This is the core loop:
      route → (reformulate?) → retrieve → confidence check
        → if insufficient: loosen filter + new query → retry
        → if sufficient:   generate answer
    """

    start_time = time.time()

    # The trace is a list of step dicts — one per action taken
    trace = []

    # ── STEP 1: ROUTE ──
    print(f"\n{'═'*60}")
    print(f"AGENT: '{query}'")
    print(f"{'═'*60}")
    print("Step 1 — Routing...")

    route_result = route(query)

    trace.append({
        "step":     "route",
        "decision": {
            "query_type":          route_result["query_type"],
            "sections":            route_result["sections"],
            "companies":           route_result["companies"],
            "needs_reformulation": route_result["needs_reformulation"],
            "reasoning":           route_result["reasoning"]
        }
    })

    print(f"  Type:     {route_result['query_type']}")
    print(f"  Sections: {route_result['sections']}")
    print(f"  Companies:{route_result['companies']}")
    print(f"  Reformat: {route_result['needs_reformulation']}")
    print(f"  Reason:   {route_result['reasoning']}")

    # ── STEP 2: REFORMULATE (if needed) ──
    active_query = query

    if route_result["needs_reformulation"]:
        print("\nStep 2 — Reformulating weak query...")
        active_query = reformulate(query, route_result)
        print(f"  Original:  '{query}'")
        print(f"  Rewritten: '{active_query}'")

        trace.append({
            "step":     "reformulate",
            "decision": {
                "original_query":    query,
                "reformulated_query": active_query,
                "reason": "Query flagged as too vague for direct retrieval"
            }
        })
    else:
        print("\nStep 2 — Query is precise, no reformulation needed")
        trace.append({
            "step":     "reformulate",
            "decision": {
                "original_query":     query,
                "reformulated_query": active_query,
                "reason":             "Query already precise"
            }
        })

    # ── STEP 3: RETRIEVE + CONFIDENCE LOOP ──
    active_route    = route_result
    attempt         = 0
    retrieve_result = None
    confidence      = None

    while attempt <= MAX_RETRIES:
        print(f"\nStep 3.{attempt + 1} — Retrieving (attempt {attempt + 1}/{MAX_RETRIES + 1})...")
        print(f"  Query:   '{active_query}'")
        print(f"  Filter:  sections={active_route.get('sections', [])}, "
              f"companies={active_route.get('companies', [])}")

        retrieve_result = retrieve(active_query, active_route, components)
        confidence      = score_confidence(retrieve_result)

        print(f"  Results: {len(retrieve_result['chunks'])} chunks")
        print(f"  Confidence: {confidence['confidence']} "
              f"({'✓ sufficient' if confidence['is_sufficient'] else '✗ insufficient'})")

        if confidence["failures"]:
            for failure in confidence["failures"]:
                print(f"    ✗ {failure}")

        trace.append({
            "step":    f"retrieve_attempt_{attempt + 1}",
            "decision": {
                "query_used":      active_query,
                "filter_used":     retrieve_result["filter_used"],
                "chunks_returned": len(retrieve_result["chunks"]),
                "confidence":      confidence["confidence"],
                "is_sufficient":   confidence["is_sufficient"],
                "failures":        confidence["failures"],
                "recommendation":  confidence["recommendation"],
                "top_score": (
                    retrieve_result["chunks"][0]["rerank_score"]
                    if retrieve_result["chunks"] else None
                )
            }
        })

        # If sufficient — break out and generate
        if confidence["is_sufficient"]:
            print(f"  → Proceeding to generation")
            break

        # If we've used all retries — generate anyway with what we have
        if attempt >= MAX_RETRIES:
            print(f"  → Max retries reached — generating with available chunks")
            break

        # ── RETRY: new query + loosened filter ──
        attempt += 1
        print(f"\n  Retrying with attempt {attempt}...")

        # Loosen the filter
        active_route = loosen_filter(active_route, attempt)
        loosened_desc = active_route.get("_filter_loosened", "filter loosened")
        print(f"  Filter loosened: {loosened_desc}")

        # Generate a new query angle
        active_query = generate_retry_queries(
            query, active_query, active_route, attempt
        )
        print(f"  New query: '{active_query}'")

        trace.append({
            "step":     f"retry_{attempt}",
            "decision": {
                "reason":          confidence["recommendation"],
                "filter_loosened": loosened_desc,
                "new_query":       active_query,
                "new_sections":    active_route.get("sections",  []),
                "new_companies":   active_route.get("companies", [])
            }
        })

    # ── STEP 4: GENERATE ──
    print(f"\nStep 4 — Generating answer...")

    chunks = retrieve_result["chunks"] if retrieve_result else []

    if not chunks:
        # Complete retrieval failure — return honest message
        answer_result = {
            "answer": (
                "I was unable to find relevant information in the SEC filings "
                f"to answer your question: '{query}'. "
                "Please try rephrasing your question or asking about a specific "
                "company or financial metric."
            ),
            "cited_chunks": []
        }
    else:
        answer_result = generate(query, chunks)

    elapsed = round(time.time() - start_time, 2)

    trace.append({
        "step":     "generate",
        "decision": {
            "chunks_used":    len(chunks),
            "citations_used": len(answer_result["cited_chunks"]),
            "elapsed_sec":    elapsed
        }
    })

    print(f"\n{'═'*60}")
    print(f"DONE in {elapsed}s")
    print(f"{'═'*60}")

    return {
        "query":        query,
        "answer":       answer_result["answer"],
        "cited_chunks": answer_result["cited_chunks"],
        "sources":      chunks,
        "confidence":   confidence,
        "trace":        trace,
        "elapsed_sec":  elapsed,
        "attempts":     attempt + 1
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from retriever import load_retriever_components

    components = load_retriever_components()

    test_queries = [
        "What are Apple's main cybersecurity risks?",   # precise → no reformulation
        "How is Nvidia doing?",                         # vague → reformulation
        "What are the risks facing semiconductor companies like AMD and Nvidia?",
    ]

    for query in test_queries:
        result = run(query, components)

        print(f"\nANSWER:")
        print(result["answer"])

        print(f"\nSOURCES USED:")
        for c in result["cited_chunks"]:
            print(f"  [{c['citation_number']}] {c['company']} — {c['section']}")

        print(f"\nAGENT TRACE ({len(result['trace'])} steps):")
        for step in result["trace"]:
            print(f"  {step['step']}: {list(step['decision'].keys())}")

        print(f"\nAttempts: {result['attempts']} | "
              f"Confidence: {result['confidence']['confidence']} | "
              f"Time: {result['elapsed_sec']}s")
        print(f"\n{'─'*60}\n")