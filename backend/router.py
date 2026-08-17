import os
import json
import groq
from dotenv import load_dotenv

load_dotenv()
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Known tickers for company detection
KNOWN_TICKERS = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "nvda": "NVDA",
    "amazon": "AMZN", "amzn": "AMZN",
    "tesla": "TSLA", "tsla": "TSLA",
    "netflix": "NFLX", "nflx": "NFLX",
    "salesforce": "CRM", "crm": "CRM",
    "amd": "AMD",
}

VALID_SECTIONS  = ["business", "risk_factors", "mda", "financials", "general"]
VALID_COMPANIES = ["AAPL","MSFT","GOOGL","META","NVDA","AMZN","TSLA","NFLX","CRM","AMD"]


# ─────────────────────────────────────────────
# ROUTER — classifies the query
# ─────────────────────────────────────────────
def route(query: str) -> dict:
    """
    Classifies the query into:
      - query_type: what kind of question this is
      - sections:   which 10-K sections to search (router's core decision)
      - companies:  which companies to filter to (empty = search all)
      - needs_reformulation: whether the query is too vague to retrieve well
      - reasoning:  why the router made these choices (shown in agent trace)

    This runs before any retrieval happens. It tells the retriever
    exactly where to look instead of searching everything blindly.
    """

    system_prompt = """You are a financial document router. Your job is to classify a user query about SEC 10-K filings and decide exactly where to search.

You must respond with valid JSON only. No preamble, no explanation outside the JSON.

The 10-K sections available are:
- "business": company overview, products, services, strategy, markets, competition
- "risk_factors": risks, threats, uncertainties, regulatory concerns, cybersecurity risks
- "mda": revenue, growth, financial performance, operating results, management outlook, guidance
- "financials": specific financial figures, income statements, balance sheet numbers, EPS, margins
- "general": anything that doesn't clearly fit the above

Companies available: AAPL, MSFT, GOOGL, META, NVDA, AMZN, TSLA, NFLX, CRM, AMD

Respond with this exact JSON structure:
{
  "query_type": one of ["financial_metrics", "risk_analysis", "business_overview", "comparison", "general"],
  "sections": [list of 1-3 section names most relevant to this query],
  "companies": [list of ticker symbols mentioned, empty list if none or if comparing all],
  "needs_reformulation": true if the query is vague, ambiguous, or too short to retrieve well,
  "reasoning": "one sentence explaining the routing decision"
}

Examples:
Query: "What are Apple's cybersecurity risks?"
{"query_type":"risk_analysis","sections":["risk_factors"],"companies":["AAPL"],"needs_reformulation":false,"reasoning":"Explicit risk question about a named company — route directly to risk_factors."}

Query: "How is Nvidia doing?"
{"query_type":"financial_metrics","sections":["mda","financials"],"companies":["NVDA"],"needs_reformulation":true,"reasoning":"Query is too vague — needs reformulation to specify what aspect of Nvidia's performance to retrieve."}

Query: "Compare Tesla and Ford risks"
{"query_type":"risk_analysis","sections":["risk_factors"],"companies":["TSLA"],"needs_reformulation":false,"reasoning":"Ford is not in the index so only Tesla is searched; risk_factors is the clear section."}

Query: "revenue"
{"query_type":"financial_metrics","sections":["mda","financials"],"companies":[],"needs_reformulation":true,"reasoning":"Single word query — too vague to retrieve meaningfully without reformulation."}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=300,
        temperature=0.0,   # deterministic — routing must be consistent
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Route this query: {query}"}
        ]
    )

    raw = response.choices[0].message.content.strip()

    # ── Parse JSON response ──
    try:
        # Strip markdown code fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fallback if model doesn't follow format
        print(f"  Router JSON parse failed, using fallback. Raw: {raw[:100]}")
        result = {
            "query_type":          "general",
            "sections":            ["general", "mda"],
            "companies":           [],
            "needs_reformulation": True,
            "reasoning":           "Router parse failed — using broad fallback"
        }

    # ── Validate and sanitize ──
    # Ensure sections are valid
    result["sections"] = [
        s for s in result.get("sections", ["general"])
        if s in VALID_SECTIONS
    ] or ["general"]

    # Ensure companies are valid tickers
    result["companies"] = [
        c.upper() for c in result.get("companies", [])
        if c.upper() in VALID_COMPANIES
    ]

    # Ensure required fields exist
    result.setdefault("query_type",          "general")
    result.setdefault("needs_reformulation", False)
    result.setdefault("reasoning",           "")

    return result


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are Apple's main cybersecurity risks?",
        "How did Nvidia's revenue grow last year?",
        "How is Tesla doing?",
        "Compare the business models of Microsoft and Salesforce",
        "What products does Amazon sell?",
        "revenue",
        "What are the biggest risks facing semiconductor companies?",
        "What was Meta's net income?",
    ]

    print("── Router test ──\n")
    for query in test_queries:
        print(f"Query: '{query}'")
        result = route(query)
        print(f"  type:     {result['query_type']}")
        print(f"  sections: {result['sections']}")
        print(f"  companies:{result['companies']}")
        print(f"  reformat: {result['needs_reformulation']}")
        print(f"  reason:   {result['reasoning']}")
        print()