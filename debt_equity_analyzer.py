#!/usr/bin/env python3
"""
Debt-Equity Analyzer
─────────────────────
1. Calls run_pipeline() from stock_price_tracker.py — this is the SINGLE
   source of company names. No re-searching. Companies are fixed from that run.
2. Filters to companies with a VALID Beta only.
3. Uses Gemini API + Google Search to fetch live Balance Sheet / D/E data
   from the web (Screener, Moneycontrol, BSE, NSE filings) for those names.

Requirements:
    pip install google-genai

Usage:
    python debt_equity_analyzer.py
    python debt_equity_analyzer.py "Digantara"
"""

import json
import re
import sys

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Missing dependency. Run:  pip install google-genai")
    sys.exit(1)

try:
    from stock_price_tracker import run_pipeline
    from company_finder import GEMINI_API_KEY
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

MODEL = "gemini-2.0-flash"

# ──────────────────────────────────────────────────────────────────────────────

def fetch_balance_sheet(company_name: str, ticker: str) -> dict:
    """
    Use Gemini API + Google Search to fetch the most recent balance sheet
    and debt-equity figures for a given company from the web.
    Returns a parsed dict with financial fields in INR Crores.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""Search the web for the most recent annual balance sheet data for:

Company : {company_name}
Ticker  : {ticker} (NSE/BSE listed, India)

Look at sources like Screener.in, Moneycontrol, BSE filings, NSE filings,
or the company's latest annual report.

Extract these figures (all in Indian Rupees, Crores):
- Total Debt (all borrowings combined)
- Long-Term Debt / Long-Term Borrowings
- Short-Term Debt / Short-Term Borrowings / Current Portion of Debt
- Total Shareholders' Equity (Net Worth)
- Debt-to-Equity Ratio (D/E)
- Total Assets
- Cash and Cash Equivalents
- Current Liabilities
- Financial Year or quarter the data belongs to

Return ONLY a valid JSON object — no markdown, no code fences, no extra text:
{{
    "company"                : "{company_name}",
    "ticker"                 : "{ticker}",
    "data_source"            : "screener / moneycontrol / bse / nse / annual report",
    "as_of"                  : "FY2024 / Q3FY25 / etc",
    "total_debt_cr"          : <number or null>,
    "long_term_debt_cr"      : <number or null>,
    "short_term_debt_cr"     : <number or null>,
    "total_equity_cr"        : <number or null>,
    "de_ratio"               : <number or null>,
    "total_assets_cr"        : <number or null>,
    "cash_cr"                : <number or null>,
    "current_liabilities_cr" : <number or null>
}}"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        ),
    )

    raw     = response.text
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "",     cleaned).strip()

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"company": company_name, "ticker": ticker,
            "error": "Could not parse Gemini response", "raw": raw}


def fmt(value, suffix=" Cr") -> str:
    """Format a number nicely, return N/A if None."""
    if value is None:
        return "N/A"
    try:
        return f"₹ {float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def leverage_label(de) -> str:
    if de is None:   return "N/A"
    if de == 0:      return "Debt-Free"
    if de < 0.5:     return "Low Leverage"
    if de < 1.0:     return "Moderate Leverage"
    if de < 2.0:     return "High Leverage"
    return "Very High Leverage"


def display_balance_sheet(beta: float, bs: dict) -> None:
    """Pretty-print balance sheet data for one company."""
    if "error" in bs:
        print(f"\n  [!] {bs['company']} ({bs['ticker']}) — {bs['error']}")
        return

    de = bs.get("de_ratio")
    print(f"\n  Company          : {bs.get('company')}")
    print(f"  Ticker           : {bs.get('ticker')}")
    print(f"  Beta (3M)        : {beta:.4f}")
    print(f"  Source           : {bs.get('data_source', 'N/A')}  |  As of: {bs.get('as_of', 'N/A')}")
    print(f"  ── Debt-Equity Structure ──────────────────────────────────")
    print(f"  D/E Ratio        : {f'{de:.4f}  ({leverage_label(de)})' if de is not None else 'N/A'}")
    print(f"  Total Debt       : {fmt(bs.get('total_debt_cr'))}")
    print(f"  Long-Term Debt   : {fmt(bs.get('long_term_debt_cr'))}")
    print(f"  Short-Term Debt  : {fmt(bs.get('short_term_debt_cr'))}")
    print(f"  Total Equity     : {fmt(bs.get('total_equity_cr'))}")
    print(f"  ── Balance Sheet Snapshot ─────────────────────────────────")
    print(f"  Total Assets     : {fmt(bs.get('total_assets_cr'))}")
    print(f"  Cash & Equiv     : {fmt(bs.get('cash_cr'))}")
    print(f"  Current Liab.    : {fmt(bs.get('current_liabilities_cr'))}")
    print("  " + "=" * 58)


def display_final_summary(results: list[dict]) -> None:
    """Compact comparison table."""
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY  —  Beta + Debt-Equity (Gemini Web Search)")
    print("=" * 72)
    print(f"\n  {'Company':<32} {'Ticker':<16} {'Beta':>7}  {'D/E':>7}  Leverage")
    print("  " + "-" * 70)
    for r in results:
        de  = r["bs"].get("de_ratio")
        de_str = f"{de:.4f}" if de is not None else "   N/A"
        print(f"  {r['name']:<32} {r['ticker']:<16} {r['beta']:>7.4f}  "
              f"{de_str:>7}  {leverage_label(de)}")
    print()


# ── Pipeline (importable) ──────────────────────────────────────────────────────

def run_full_pipeline(company_name: str) -> list[dict]:
    """
    Runs stock price + beta pipeline, then fetches D/E via Gemini for
    companies with valid Beta. Returns a list of dicts:
        [{"name": ..., "ticker": ..., "beta": ..., "de_ratio": ...}, ...]
    de_ratio is None if D/E data was unavailable.
    """
    beta_results = run_pipeline(company_name)
    if not beta_results:
        return []

    valid = [r for r in beta_results if r["beta"] is not None]
    if not valid:
        return beta_results   # return as-is, de_ratio will be absent

    combined = []
    for r in valid:
        bs       = fetch_balance_sheet(r["name"], r["ticker"])
        de_ratio = bs.get("de_ratio") if "error" not in bs else None
        combined.append({
            "name"    : r["name"],
            "ticker"  : r["ticker"],
            "beta"    : r["beta"],
            "de_ratio": de_ratio,
            "bs"      : bs,
        })
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print("  DEBT-EQUITY ANALYZER  |  Gemini Web Search + Beta from Tracker")
    print("=" * 72)

    company_name = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 \
                   else input("\n  Enter company name: ").strip()

    if not company_name:
        print("  Error: Company name cannot be empty.")
        sys.exit(1)

    # ── Step 1: Run pipeline — single source of company names + betas ─────────
    print(f"\n  Running stock price + beta pipeline for \"{company_name}\"...")
    beta_results = run_pipeline(company_name)

    if not beta_results:
        print("  No results from pipeline. Exiting.")
        sys.exit(1)

    # ── Step 2: Filter — only companies with a valid Beta ─────────────────────
    valid   = [r for r in beta_results if r["beta"] is not None]
    skipped = [r for r in beta_results if r["beta"] is None]

    if skipped:
        names = ", ".join(r["ticker"] for r in skipped)
        print(f"\n  Skipping (no valid Beta): {names}")

    if not valid:
        print("  No companies with valid Beta. Cannot fetch Balance Sheets.")
        sys.exit(1)

    print(f"\n  Companies with valid Beta: "
          f"{', '.join(r['name'] for r in valid)}")

    # ── Step 3: Fetch Balance Sheet via Gemini for each valid company ──────────
    print("\n" + "=" * 72)
    print("  BALANCE SHEET + DEBT-EQUITY  (via Gemini Google Search)")
    print("=" * 72)

    final_results = []
    for r in valid:
        print(f"\n  Searching web for: {r['name']} ({r['ticker']})...")
        bs = fetch_balance_sheet(r["name"], r["ticker"])
        display_balance_sheet(r["beta"], bs)
        final_results.append({
            "name"  : r["name"],
            "ticker": r["ticker"],
            "beta"  : r["beta"],
            "bs"    : bs,
        })

    display_final_summary(final_results)


if __name__ == "__main__":
    main()
