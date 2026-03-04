#!/usr/bin/env python3
"""
Company Stock Market Finder
────────────────────────────
Accepts a company name, uses Gemini AI + Google Search to research it,
and returns the top 4 similar publicly listed companies as a dictionary.

Requirements:
    pip install google-genai

Usage:
    python company_finder.py
    python company_finder.py "Tesla"
"""

import json
import os
import re
import sys

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Missing dependency. Run:  pip install google-genai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; set GEMINI_API_KEY in environment directly

# ── Configuration ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL          = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not set. Add it to a .env file or environment.")
    sys.exit(1)
# ──────────────────────────────────────────────────────────────────────────────


def find_similar_companies(company_name: str) -> dict:
    """
    Uses Gemini AI with live Google Search to find the top 4
    publicly listed companies similar to the given company name.

    Returns a parsed dict with 'input_company' and 'similar_companies' keys.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
You are a financial research analyst with access to live web search.

Research the company "{company_name}" using web search and then identify
the TOP 4 most similar companies that are CURRENTLY PUBLICLY LISTED
ONLY on Indian stock exchanges — BSE (Bombay Stock Exchange) or NSE (National Stock Exchange of India).

Rules:
- ONLY include companies listed on BSE or NSE. Do NOT include any company from NYSE, NASDAQ, LSE, or any other exchange.
- Only include companies with ACTIVE stock listings (not delisted or private).
- Base similarity on: same sector, business model, products/services, customer base.
- Verify the ticker symbols are correct and currently traded on BSE/NSE.

Return ONLY a valid JSON object — no markdown, no code fences, no extra text:

{{
    "input_company": {{
        "name": "{company_name}",
        "sector": "broad sector",
        "industry": "specific industry",
        "description": "2-3 sentence overview"
    }},
    "similar_companies": [
        {{
            "rank": 1,
            "name": "Full Legal Company Name",
            "ticker": "TICK",
            "exchange": "NSE or BSE",
            "sector": "Technology",
            "industry": "Cloud Computing",
            "headquarters": "City, Country",
            "similarity_reason": "Concise reason why this company is similar",
            "market_cap": "$X billion"
        }},
        {{
            "rank": 2,
            "name": "...",
            "ticker": "...",
            "exchange": "...",
            "sector": "...",
            "industry": "...",
            "headquarters": "...",
            "similarity_reason": "...",
            "market_cap": "..."
        }},
        {{
            "rank": 3,
            "name": "...",
            "ticker": "...",
            "exchange": "...",
            "sector": "...",
            "industry": "...",
            "headquarters": "...",
            "similarity_reason": "...",
            "market_cap": "..."
        }},
        {{
            "rank": 4,
            "name": "...",
            "ticker": "...",
            "exchange": "...",
            "sector": "...",
            "industry": "...",
            "headquarters": "...",
            "similarity_reason": "...",
            "market_cap": "..."
        }}
    ]
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        ),
    )

    raw_text = response.text

    # Strip markdown code fences if the model added them
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Extract the first JSON object found in the response
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            return {
                "error": f"JSON parse error: {exc}",
                "raw_response": raw_text,
            }

    return {"error": "No JSON found in response.", "raw_response": raw_text}


def display_results(data: dict) -> None:
    """Pretty-print results and the final publicly-listed companies dictionary."""

    if "error" in data:
        print(f"\n[ERROR] {data['error']}")
        if "raw_response" in data:
            print("\nRaw Gemini response:\n")
            print(data["raw_response"])
        return

    input_co  = data.get("input_company", {})
    companies = data.get("similar_companies", [])

    # ── Searched Company ───────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  RESEARCH RESULT: {input_co.get('name', 'Unknown').upper()}")
    print("=" * 68)
    print(f"  Sector      : {input_co.get('sector', 'N/A')}")
    print(f"  Industry    : {input_co.get('industry', 'N/A')}")
    print(f"  Description : {input_co.get('description', 'N/A')}")

    # ── Top 4 Similar Listed Companies ────────────────────────────────────────
    print("\n" + "-" * 68)
    print("  TOP 4 SIMILAR COMPANIES LISTED ON BSE / NSE")
    print("-" * 68)

    for c in companies:
        print(f"\n  #{c.get('rank')}  {c.get('name', 'N/A')}")
        print(f"       Ticker      : {c.get('ticker', 'N/A')}  ({c.get('exchange', 'N/A')})")
        print(f"       Sector      : {c.get('sector', 'N/A')}  |  {c.get('industry', 'N/A')}")
        print(f"       HQ          : {c.get('headquarters', 'N/A')}")
        print(f"       Market Cap  : {c.get('market_cap', 'N/A')}")
        print(f"       Similarity  : {c.get('similarity_reason', 'N/A')}")

    # ── Clean Dictionary Output ───────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  DICTIONARY  —  INDIAN LISTED COMPANIES (BSE / NSE)")
    print("=" * 68)

    listed_dict: dict = {}
    for c in companies:
        ticker = c.get("ticker", f"RANK_{c.get('rank', '?')}")
        listed_dict[ticker] = {
            "name"            : c.get("name"),
            "exchange"        : c.get("exchange"),
            "sector"          : c.get("sector"),
            "industry"        : c.get("industry"),
            "headquarters"    : c.get("headquarters"),
            "market_cap"      : c.get("market_cap"),
            "similarity_reason": c.get("similarity_reason"),
        }

    print(json.dumps(listed_dict, indent=4))
    print()


def main() -> None:
    print("\n" + "=" * 68)
    print("     STOCK MARKET COMPANY FINDER  |  Powered by Gemini AI")
    print("=" * 68)

    # Accept company name from CLI args or interactive prompt
    if len(sys.argv) > 1:
        company_name = " ".join(sys.argv[1:]).strip()
    else:
        company_name = input("\n  Enter company name: ").strip()

    if not company_name:
        print("  Error: Company name cannot be empty.")
        sys.exit(1)

    print(f"\n  Searching for companies similar to: \"{company_name}\"")
    print("  Gemini is searching the web, please wait...\n")

    result = find_similar_companies(company_name)
    display_results(result)


if __name__ == "__main__":
    main()
