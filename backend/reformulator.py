import os
import json
import groq
from dotenv import load_dotenv

load_dotenv()
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "openai/gpt-oss-20b"


def _call(messages, max_tokens=150):
    """
    Wrapper around Groq chat completions.
    gpt-oss-20b sometimes returns content in a 'thinking' block
    and leaves the main content empty. This handles that by
    falling back to any non-empty content block found.
    """
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        temperature=0.1,
        messages=messages
    )

    # Primary path
    content = response.choices[0].message.content
    if content and content.strip():
        return content.strip().strip('"').strip("'")

    # Fallback — some models return structured blocks
    try:
        blocks = response.choices[0].message.model_extra or {}
        for val in blocks.values():
            if isinstance(val, str) and val.strip():
                return val.strip().strip('"').strip("'")
    except Exception:
        pass

    return None


def reformulate(query: str, route_result: dict) -> str:
    sections   = route_result.get("sections", [])
    companies  = route_result.get("companies", [])
    query_type = route_result.get("query_type", "general")

    company_hint = f"Companies: {', '.join(companies)}" if companies else "No specific company"
    section_hint = f"Sections to search: {', '.join(sections)}"

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite vague financial search queries into precise, "
                "information-dense queries for SEC 10-K filings. "
                "Output the rewritten query only. No explanation. No quotes. "
                "Use financial terminology. 1-2 sentences max."
            )
        },
        {
            "role": "user",
            "content": (
                f"Original query: {query}\n"
                f"{company_hint}\n"
                f"{section_hint}\n"
                f"Query type: {query_type}\n\n"
                f"Rewrite into a precise 10-K retrieval query:"
            )
        }
    ]

    result = _call(messages, max_tokens=120)

    if not result:
        # Build a fallback manually from what the router gave us
        company_str = " ".join(companies) if companies else ""
        section_map = {
            "mda": "revenue growth operating results",
            "financials": "net income earnings financial statements",
            "risk_factors": "risk factors uncertainties threats",
            "business": "business overview products services strategy",
        }
        section_str = " ".join(section_map.get(s, "") for s in sections)
        result = f"{company_str} {section_str} {query}".strip()

    return result


def generate_retry_queries(
    original_query: str,
    reformulated_query: str,
    route_result: dict,
    attempt: int
) -> str:
    companies = route_result.get("companies", [])
    sections  = route_result.get("sections",  [])
    company_hint = ", ".join(companies) if companies else "the company"

    if attempt >= 2:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate broad fallback search queries for SEC 10-K filings. "
                    "The previous query failed. Output only the new query. "
                    "No explanation. No quotes."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {original_query}\n"
                    f"Failed query: {reformulated_query}\n"
                    f"Company: {company_hint}\n\n"
                    f"Generate a broader, simpler fallback query:"
                )
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate alternative search queries for SEC 10-K filings when "
                    "the first retrieval attempt returned poor results. "
                    "Use synonyms and different financial concepts. "
                    "Output only the query. No explanation. No quotes."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {original_query}\n"
                    f"Failed query: {reformulated_query}\n"
                    f"Company: {company_hint}\n"
                    f"Sections: {', '.join(sections)}\n\n"
                    f"Generate a different-angled retry query:"
                )
            }
        ]

    result = _call(messages, max_tokens=100)

    if not result:
        # Fallback — use the original query with company prepended
        company_str = " ".join(companies) if companies else ""
        result = f"{company_str} {original_query} annual report".strip()

    return result


if __name__ == "__main__":
    from router import route

    test_cases = [
        "How is Nvidia doing?",
        "Are there any risks I should know about for Apple?",
        "revenue",
        "What does Microsoft do?",
    ]

    print("── Reformulator test ──\n")
    for query in test_cases:
        route_result = route(query)
        print(f"Original:  '{query}'")
        if route_result["needs_reformulation"]:
            rewritten = reformulate(query, route_result)
            print(f"Rewritten: '{rewritten}'")
            retry1 = generate_retry_queries(query, rewritten, route_result, attempt=1)
            print(f"Retry 1:   '{retry1}'")
        else:
            print("No reformulation needed")
        print()