import os
import re
import groq
from dotenv import load_dotenv

load_dotenv()
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_prompt(query: str, chunks: list) -> tuple:
    sources_block = ""
    for i, chunk in enumerate(chunks, 1):
        sources_block += f"""
[{i}] Company: {chunk['company']} ({chunk['ticker']})
     Section: {chunk['section']}
     Text: {chunk['text'][:600]}
"""

    system_prompt = """You are a financial research assistant analyzing SEC 10-K filings.

STRICT RULES:
1. Answer using ONLY the provided sources. Never use outside knowledge.
2. After every factual claim, add a citation tag [1], [2], etc.
3. Use specific numbers, percentages, and figures when they appear in the sources.
4. If sources lack sufficient information, say: "The provided sources do not contain enough information to fully answer this question."
5. Never speculate beyond what the sources state.
6. Structure your answer in clear paragraphs. No bullet points."""

    user_prompt = f"""Sources:
{sources_block}

Question: {query}

Answer using only the sources above. Cite every factual claim."""

    return system_prompt, user_prompt


def generate_answer(query: str, chunks: list) -> str:
    system_prompt, user_prompt = build_prompt(query, chunks)

    message = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )

    return message.choices[0].message.content


def parse_citations(answer: str, chunks: list) -> list:
    cited_numbers = set(re.findall(r'\[(\d+)\]', answer))
    cited = []
    for n in sorted(cited_numbers):
        idx = int(n) - 1
        if 0 <= idx < len(chunks):
            cited.append({
                "citation_number": int(n),
                "company":  chunks[idx]["company"],
                "ticker":   chunks[idx]["ticker"],
                "section":  chunks[idx]["section"],
                "text_snippet": chunks[idx]["text"][:200]
            })
    return cited


def generate(query: str, chunks: list) -> dict:
    print(f"  Generating answer over {len(chunks)} chunks...")
    answer       = generate_answer(query, chunks)
    cited        = parse_citations(answer, chunks)
    print(f"  Citations used: {[c['citation_number'] for c in cited]}")
    return {
        "answer":       answer,
        "cited_chunks": cited
    }