import numpy as np


# ─────────────────────────────────────────────
# CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────────
# These control when the agent decides to retry.
# Tuned for SEC 10-K chunks — financial text scores
# lower on the ms-marco cross-encoder (trained on web
# queries) than academic text, so thresholds are lower
# than you'd use for a general-purpose RAG system.

MIN_CHUNKS_REQUIRED  = 3     # fewer than this → definitely retry
MIN_RERANK_SCORE     = -8.0  # top chunk below this → likely off-topic
MIN_AVERAGE_SCORE    = -10.0  # average below this → broad retrieval failure
MIN_SCORE_SPREAD     = 0.5   # top and bottom too close → no clear winner


# ─────────────────────────────────────────────
# SCORE RETRIEVAL CONFIDENCE
# ─────────────────────────────────────────────
def score_confidence(retrieve_result: dict) -> dict:
    """
    Examines the retrieval result and produces a confidence assessment.

    Four signals are checked independently:
      1. chunk_count   — did we get enough results?
      2. top_score     — is the best chunk actually relevant?
      3. average_score — are results broadly relevant or just one lucky hit?
      4. score_spread  — is there a clear winner or are all chunks equally weak?

    Returns a confidence dict with:
      is_sufficient:   bool — whether to proceed to generation or retry
      confidence:      float 0.0–1.0 — overall confidence score
      signals:         dict of individual signal values
      failures:        list of which signals failed (shown in agent trace)
      recommendation:  what the agent should do next
    """
    chunks = retrieve_result.get("chunks", [])

    # ── Signal 1: chunk count ──
    chunk_count = len(chunks)
    count_ok    = chunk_count >= MIN_CHUNKS_REQUIRED

    # ── Signal 2: top rerank score ──
    if chunks:
        top_score    = chunks[0].get("rerank_score", -999)
        all_scores   = [c.get("rerank_score", -999) for c in chunks]
        avg_score    = float(np.mean(all_scores))
        score_spread = top_score - min(all_scores) if len(all_scores) > 1 else 0.0
    else:
        top_score    = -999
        avg_score    = -999
        score_spread = 0.0

    top_ok     = top_score  >= MIN_RERANK_SCORE
    avg_ok     = avg_score  >= MIN_AVERAGE_SCORE
    spread_ok  = score_spread >= MIN_SCORE_SPREAD

    # ── Collect failures ──
    failures = []
    if not count_ok:
        failures.append(
            f"too few chunks ({chunk_count} < {MIN_CHUNKS_REQUIRED} required)"
        )
    if not top_ok:
        failures.append(
            f"top score too low ({top_score:.2f} < {MIN_RERANK_SCORE} threshold)"
        )
    if not avg_ok:
        failures.append(
            f"average score too low ({avg_score:.2f} < {MIN_AVERAGE_SCORE} threshold)"
        )
    if not spread_ok and chunk_count >= MIN_CHUNKS_REQUIRED:
        # Only flag spread if we have enough chunks to judge
        failures.append(
            f"score spread too narrow ({score_spread:.2f} — no clear winner)"
        )

    # ── Overall confidence score (0.0 – 1.0) ──
    # Each signal contributes 25%. Partial credit for near-misses.
    signals_passed = sum([count_ok, top_ok, avg_ok, spread_ok])
    confidence     = round(signals_passed / 4, 2)

    # Must pass count + top_score at minimum to be sufficient
    # (the two most critical signals)
    is_sufficient = count_ok and top_ok

    # ── Recommendation ──
    if is_sufficient:
        recommendation = "proceed_to_generation"
    elif chunk_count == 0:
        recommendation = "retry_with_no_filter"
    elif not count_ok:
        recommendation = "retry_with_broader_sections"
    else:
        recommendation = "retry_with_reformulated_query"

    return {
        "is_sufficient":  is_sufficient,
        "confidence":     confidence,
        "signals": {
            "chunk_count":   chunk_count,
            "top_score":     round(top_score, 4),
            "average_score": round(avg_score, 4),
            "score_spread":  round(score_spread, 4),
        },
        "failures":        failures,
        "recommendation":  recommendation,
    }


# ─────────────────────────────────────────────
# LOOSEN FILTER FOR RETRY
# ─────────────────────────────────────────────
def loosen_filter(route_result: dict, attempt: int) -> dict:
    """
    When retrieval is thin, the agent needs to search a broader space.
    This progressively relaxes the filters on each retry.

    attempt=1 → keep company filter, broaden sections to include 'general'
    attempt=2 → drop all filters, search everything

    This mirrors how a human researcher would approach a failed search:
    first try adjacent sections, then give up on filtering entirely.
    """
    loosened = dict(route_result)

    if attempt == 1:
        # Add 'general' section to capture chunks that weren't labeled precisely
        current_sections = loosened.get("sections", [])
        if "general" not in current_sections:
            loosened["sections"] = current_sections + ["general"]
        loosened["_filter_loosened"] = "added general section"

    elif attempt >= 2:
        # Drop all filters — search the full 808 chunks
        loosened["sections"]  = []
        loosened["companies"] = []
        loosened["_filter_loosened"] = "removed all filters"

    return loosened


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Simulate different retrieval outcomes
    test_cases = [
        {
            "label": "Good retrieval — should proceed",
            "retrieve_result": {
                "chunks": [
                    {"rerank_score": 6.2, "ticker": "AAPL", "section": "risk_factors"},
                    {"rerank_score": 5.1, "ticker": "AAPL", "section": "risk_factors"},
                    {"rerank_score": 4.8, "ticker": "AAPL", "section": "risk_factors"},
                    {"rerank_score": 3.2, "ticker": "AAPL", "section": "risk_factors"},
                    {"rerank_score": 2.1, "ticker": "AAPL", "section": "risk_factors"},
                ]
            }
        },
        {
            "label": "Too few chunks — should retry",
            "retrieve_result": {
                "chunks": [
                    {"rerank_score": -1.8, "ticker": "AAPL", "section": "risk_factors"},
                    {"rerank_score": -2.1, "ticker": "AAPL", "section": "risk_factors"},
                ]
            }
        },
        {
            "label": "Low scores — should retry",
            "retrieve_result": {
                "chunks": [
                    {"rerank_score": -4.2, "ticker": "NVDA", "section": "mda"},
                    {"rerank_score": -5.1, "ticker": "NVDA", "section": "mda"},
                    {"rerank_score": -5.8, "ticker": "NVDA", "section": "mda"},
                    {"rerank_score": -6.1, "ticker": "NVDA", "section": "mda"},
                    {"rerank_score": -6.9, "ticker": "NVDA", "section": "mda"},
                ]
            }
        },
        {
            "label": "Empty — should retry with no filter",
            "retrieve_result": {"chunks": []}
        },
    ]

    mock_route = {
        "sections":  ["risk_factors"],
        "companies": ["AAPL"],
        "query_type": "risk_analysis"
    }

    print("── Confidence checker test ──\n")

    for case in test_cases:
        print(f"Case: {case['label']}")
        result = score_confidence(case["retrieve_result"])

        status = "✓ SUFFICIENT" if result["is_sufficient"] else "✗ INSUFFICIENT"
        print(f"  {status} (confidence={result['confidence']})")
        print(f"  Signals: chunks={result['signals']['chunk_count']} | "
              f"top={result['signals']['top_score']} | "
              f"avg={result['signals']['average_score']:.2f} | "
              f"spread={result['signals']['score_spread']:.2f}")

        if result["failures"]:
            for f in result["failures"]:
                print(f"  ✗ {f}")

        print(f"  → {result['recommendation']}")

        if not result["is_sufficient"]:
            loosened1 = loosen_filter(mock_route, attempt=1)
            loosened2 = loosen_filter(mock_route, attempt=2)
            print(f"  Retry 1 filter: sections={loosened1['sections']}, "
                  f"companies={loosened1['companies']}")
            print(f"  Retry 2 filter: sections={loosened2['sections']}, "
                  f"companies={loosened2['companies']}")
        print()