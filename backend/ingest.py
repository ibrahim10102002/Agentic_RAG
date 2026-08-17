import requests
import json
import os
import time
import re

# ── 10 companies with their SEC CIK numbers ──
# CIK is SEC's unique identifier for every public company
COMPANIES = [
    {"name": "Apple",     "ticker": "AAPL", "cik": "0000320193"},
    {"name": "Microsoft", "ticker": "MSFT", "cik": "0000789019"},
    {"name": "Alphabet",  "ticker": "GOOGL","cik": "0001652044"},
    {"name": "Meta",      "ticker": "META", "cik": "0001326801"},
    {"name": "Nvidia",    "ticker": "NVDA", "cik": "0001045810"},
    {"name": "Amazon",    "ticker": "AMZN", "cik": "0001018724"},
    {"name": "Tesla",     "ticker": "TSLA", "cik": "0001318605"},
    {"name": "Netflix",   "ticker": "NFLX", "cik": "0001065280"},
    {"name": "Salesforce","ticker": "CRM",  "cik": "0001108524"},
    {"name": "AMD",       "ticker": "AMD",  "cik": "0000002488"},
]

# SEC requires a descriptive User-Agent header — they block generic ones
HEADERS = {
    "User-Agent": "Agentic RAG Project researcher@example.com",
    "Accept": "application/json"
}

# ── Section patterns we care about in a 10-K ──
# We detect these by scanning for their standard headings
SECTION_PATTERNS = {
    "business": [
        r"item\s+1[\.\s]+business",
        r"our\s+business",
        r"company\s+overview",
    ],
    "risk_factors": [
        r"item\s+1a[\.\s]+risk\s+factors",
        r"risk\s+factors",
    ],
    "mda": [
        r"item\s+7[\.\s]+management",
        r"management.s\s+discussion",
        r"results\s+of\s+operations",
    ],
    "financials": [
        r"item\s+8[\.\s]+financial\s+statements",
        r"consolidated\s+statements?\s+of\s+(operations|income|earnings)",
        r"revenue\s+\$",
    ]
}


# ─────────────────────────────────────────────
# STEP A: Get the latest 10-K filing URL
# ─────────────────────────────────────────────
def get_latest_10k_url(cik):
    """
    SEC EDGAR has a submissions API that lists every filing
    a company has ever made, sorted newest first.
    We grab the most recent 10-K filing URL.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    forms       = filings.get("form", [])
    accessions  = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    for form, accession, doc in zip(forms, accessions, primary_docs):
        if form == "10-K":
            # Build the URL to the actual filing document
            accession_clean = accession.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_clean}/{doc}"
            )
            return filing_url

    return None


# ─────────────────────────────────────────────
# STEP B: Download and clean the filing text
# ─────────────────────────────────────────────
def download_filing(url):
    """
    Downloads the raw 10-K HTML/text from SEC EDGAR.
    Strips HTML tags to get clean text.
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    raw = resp.text

    # Strip HTML tags — 10-Ks are often filed as HTML
    clean = re.sub(r'<[^>]+>', ' ', raw)

    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()

    return clean


# ─────────────────────────────────────────────
# STEP C: Detect which section a text block belongs to
# ─────────────────────────────────────────────
def detect_section(text_lower):
    """
    Scans a chunk of text against our section patterns.
    Returns the section label if matched, else 'general'.
    """
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return section
    return "general"


# ─────────────────────────────────────────────
# STEP D: Chunk the filing text with section labels
# ─────────────────────────────────────────────
def chunk_filing(text, company, chunk_size=800, overlap=100):
    """
    Splits the filing into overlapping chunks of ~800 words.
    Each chunk gets:
      - company name + ticker
      - section label (business / risk_factors / mda / financials / general)
      - position in document (useful for understanding context)

    Why 800 words? Large enough to contain a complete idea,
    small enough to stay within LLM context windows and keep
    retrieval precise. Overlap of 100 words prevents cutting
    ideas at chunk boundaries.
    """
    words = text.split()
    total = len(words)
    chunks = []
    i = 0
    chunk_idx = 0

    while i < total:
        chunk_words = words[i : i + chunk_size]
        chunk_text  = " ".join(chunk_words)
        text_lower  = chunk_text.lower()

        section = detect_section(text_lower)

        # Skip chunks that are mostly boilerplate / table of contents
        if len(chunk_words) < 50:
            i += chunk_size - overlap
            continue

        chunks.append({
            "id":        f"{company['ticker']}_{chunk_idx}",
            "company":   company["name"],
            "ticker":    company["ticker"],
            "section":   section,
            "text":      chunk_text,
            "position":  round(i / total, 3),  # 0.0 = start, 1.0 = end of doc
            "word_count": len(chunk_words)
        })

        chunk_idx += 1
        i += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────
# STEP E: Run everything
# ─────────────────────────────────────────────
def main():
    all_chunks = []
    processed_companies = []

    os.makedirs("../data/filings", exist_ok=True)

    for company in COMPANIES:
        print(f"\nProcessing {company['name']} ({company['ticker']})...")

        # ── 1. Get filing URL ──
        try:
            filing_url = get_latest_10k_url(company["cik"])
            if not filing_url:
                print(f"  No 10-K found for {company['name']}, skipping")
                continue
            print(f"  Filing URL: {filing_url}")
        except Exception as e:
            print(f"  Error getting URL: {e}")
            continue

        # ── 2. Download filing ──
        try:
            raw_path = f"../data/filings/{company['ticker']}.txt"

            # Cache locally so we don't re-download on reruns
            if os.path.exists(raw_path):
                print(f"  Using cached filing")
                with open(raw_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                print(f"  Downloading filing...")
                text = download_filing(filing_url)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"  Saved to {raw_path}")

        except Exception as e:
            print(f"  Error downloading: {e}")
            continue

        # ── 3. Chunk ──
        chunks = chunk_filing(text, company)
        all_chunks.extend(chunks)

        # Count sections
        section_counts = {}
        for c in chunks:
            section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1

        print(f"  Generated {len(chunks)} chunks")
        for section, count in sorted(section_counts.items()):
            print(f"    {section}: {count} chunks")

        processed_companies.append({**company, "chunk_count": len(chunks)})

        # SEC rate limit — be polite
        time.sleep(1)

    # ── Save ──
    chunks_path = "../data/chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    companies_path = "../data/companies.json"
    with open(companies_path, "w", encoding="utf-8") as f:
        json.dump(processed_companies, f, indent=2)

    print(f"\n✓ Done!")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Companies processed: {len(processed_companies)}")
    print(f"  Chunks saved → {chunks_path}")

    # ── Section distribution across all companies ──
    print(f"\nSection distribution across all companies:")
    all_sections = {}
    for c in all_chunks:
        all_sections[c["section"]] = all_sections.get(c["section"], 0) + 1
    for section, count in sorted(all_sections.items(), key=lambda x: -x[1]):
        pct = round(count / len(all_chunks) * 100)
        print(f"  {section:15s}: {count:4d} chunks ({pct}%)")


if __name__ == "__main__":
    main()