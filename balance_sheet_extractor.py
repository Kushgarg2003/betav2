#!/usr/bin/env python3
"""
Balance Sheet Extractor
────────────────────────
Accepts a balance sheet file and uses Gemini AI to extract
Debt and Equity figures from it.

Supported formats:
    PDF, PNG, JPG/JPEG  — uploaded directly to Gemini (multimodal)
    XLSX, XLS, CSV      — read locally and sent as text

Requirements:
    pip install google-genai pandas openpyxl

Usage:
    python balance_sheet_extractor.py
    python balance_sheet_extractor.py "balance_sheet.pdf"
    python balance_sheet_extractor.py "financials.xlsx"
    python balance_sheet_extractor.py "annual_report.png"
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
    pass

# ── Configuration ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL          = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not set. Add it to a .env file or environment.")
    sys.exit(1)

# File types uploaded to Gemini Files API (multimodal)
UPLOAD_MIME = {
    ".pdf" : "application/pdf",
    ".png" : "image/png",
    ".jpg" : "image/jpeg",
    ".jpeg": "image/jpeg",
}

# File types read locally and sent as text
TEXT_EXTENSIONS = {".xlsx", ".xls", ".csv"}
# ──────────────────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """
You are a financial analyst. Carefully read this balance sheet and extract
all debt and equity figures exactly as they appear.

Extract the following fields (use the same currency/unit as the document):
- Company name (if visible)
- Financial period (e.g. FY2024, Q3FY25)
- Currency and unit (e.g. INR Crores, USD Millions)
- Total Debt (all borrowings combined)
- Long-Term Debt / Long-Term Borrowings
- Short-Term Debt / Short-Term Borrowings / Current Portion of LTD
- Total Shareholders' Equity / Net Worth
- Debt-to-Equity Ratio — compute as Total Debt / Total Equity if not stated
- Total Assets
- Total Liabilities
- Cash and Cash Equivalents
- Retained Earnings

Return ONLY a valid JSON object — no markdown, no code fences, no extra text:
{
    "company"           : "Company name or null",
    "financial_period"  : "FY2024 / Q3FY25 / etc.",
    "currency_unit"     : "INR Crores / USD Millions / etc.",
    "total_debt"        : <number or null>,
    "long_term_debt"    : <number or null>,
    "short_term_debt"   : <number or null>,
    "total_equity"      : <number or null>,
    "de_ratio"          : <number or null>,
    "total_assets"      : <number or null>,
    "total_liabilities" : <number or null>,
    "cash_equivalents"  : <number or null>,
    "retained_earnings" : <number or null>,
    "notes"             : "Any important caveats or observations"
}
"""


# ── File Readers ───────────────────────────────────────────────────────────────

def read_excel_as_text(file_path: str) -> str:
    """Convert all sheets of an Excel file into a plain-text table."""
    try:
        import pandas as pd
    except ImportError:
        print("Missing dependency. Run:  pip install pandas openpyxl")
        sys.exit(1)

    sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    parts = []
    for name, df in sheets.items():
        parts.append(f"=== Sheet: {name} ===")
        parts.append(df.to_string(index=False, header=False, na_rep=""))
    return "\n\n".join(parts)


def read_csv_as_text(file_path: str) -> str:
    """Read a CSV file as raw text."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ── Core Extraction ────────────────────────────────────────────────────────────

def extract_from_file(file_path: str) -> dict:
    """Send the balance sheet file to Gemini and extract debt/equity data."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    ext    = os.path.splitext(file_path)[1].lower()

    all_supported = set(UPLOAD_MIME.keys()) | TEXT_EXTENSIONS
    if ext not in all_supported:
        print(f"\n  Error: Unsupported file type '{ext}'.")
        print(f"  Supported types: {', '.join(sorted(all_supported))}")
        sys.exit(1)

    # ── Text-based files (Excel / CSV) ────────────────────────────────────────
    if ext in TEXT_EXTENSIONS:
        print(f"  Reading {ext.upper()} file...")
        if ext in (".xlsx", ".xls"):
            file_text = read_excel_as_text(file_path)
        else:
            file_text = read_csv_as_text(file_path)

        full_prompt = f"{EXTRACT_PROMPT}\n\nBalance Sheet Data:\n\n{file_text}"
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )

    # ── Binary files (PDF / Images) — upload via Gemini Files API ─────────────
    else:
        mime_type = UPLOAD_MIME[ext]
        print(f"  Uploading {ext.upper()} to Gemini Files API...")
        uploaded = client.files.upload(
            file=file_path,
            config=types.UploadFileConfig(mime_type=mime_type),
        )
        print(f"  Upload complete. Extracting with Gemini...")
        response = client.models.generate_content(
            model=MODEL,
            contents=[uploaded, EXTRACT_PROMPT],
            config=types.GenerateContentConfig(temperature=0.1),
        )

    # ── Parse JSON from response ───────────────────────────────────────────────
    raw     = response.text
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$",         "", cleaned).strip()

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"error": "Could not parse Gemini response", "raw": raw}


# ── Display ────────────────────────────────────────────────────────────────────

def fmt(value, unit: str = "") -> str:
    """Format a number with its unit, return N/A if None."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}  ({unit})" if unit else f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def display_results(data: dict) -> None:
    """Pretty-print extracted balance sheet data."""
    if "error" in data:
        print(f"\n  [!] Extraction failed: {data['error']}")
        if "raw" in data:
            print(f"\n  Raw Gemini response:\n{data['raw']}")
        return

    unit = data.get("currency_unit", "")
    de   = data.get("de_ratio")

    print("\n" + "=" * 68)
    print("  BALANCE SHEET EXTRACTION  —  Debt & Equity")
    print("=" * 68)
    print(f"  Company          : {data.get('company') or 'N/A'}")
    print(f"  Financial Period : {data.get('financial_period') or 'N/A'}")
    print(f"  Currency / Unit  : {unit or 'N/A'}")

    print(f"\n  ── Debt Structure ──────────────────────────────────────────")
    print(f"  Total Debt       : {fmt(data.get('total_debt'), unit)}")
    print(f"  Long-Term Debt   : {fmt(data.get('long_term_debt'), unit)}")
    print(f"  Short-Term Debt  : {fmt(data.get('short_term_debt'), unit)}")

    print(f"\n  ── Equity Structure ────────────────────────────────────────")
    print(f"  Total Equity     : {fmt(data.get('total_equity'), unit)}")
    print(f"  Retained Earnings: {fmt(data.get('retained_earnings'), unit)}")

    print(f"\n  ── Key Ratio ───────────────────────────────────────────────")
    print(f"  D/E Ratio        : {f'{de:.4f}' if de is not None else 'N/A'}")

    print(f"\n  ── Balance Sheet Overview ──────────────────────────────────")
    print(f"  Total Assets     : {fmt(data.get('total_assets'), unit)}")
    print(f"  Total Liabilities: {fmt(data.get('total_liabilities'), unit)}")
    print(f"  Cash & Equiv     : {fmt(data.get('cash_equivalents'), unit)}")

    notes = data.get("notes")
    if notes:
        print(f"\n  ── Notes ───────────────────────────────────────────────────")
        print(f"  {notes}")

    print("=" * 68 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print("  BALANCE SHEET EXTRACTOR  |  Powered by Gemini AI")
    print("=" * 68)

    file_path = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 \
                else input("\n  Enter balance sheet file path: ").strip()

    if not file_path:
        print("  Error: File path cannot be empty.")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"\n  Error: File not found — '{file_path}'")
        sys.exit(1)

    size_kb = os.path.getsize(file_path) / 1024
    print(f"\n  File : {file_path}")
    print(f"  Size : {size_kb:.1f} KB")

    data = extract_from_file(file_path)
    display_results(data)


if __name__ == "__main__":
    main()
