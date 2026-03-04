#!/usr/bin/env python3
"""
Stock Price Tracker + Beta Calculator
──────────────────────────────────────
1. Finds top 3 similar BSE/NSE companies via company_finder.py
2. Fetches 3-month daily closing prices from Yahoo Finance
3. Fetches Nifty 50 (^NSEI) closing prices for the same period
4. Calculates Beta for each company vs Nifty 50

Beta = Cov(stock_returns, nifty_returns) / Var(nifty_returns)

Requirements:
    pip install yfinance numpy

Usage:
    python stock_price_tracker.py
    python stock_price_tracker.py "Infosys"
"""

import sys
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Run:  pip install yfinance")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Missing dependency. Run:  pip install numpy")
    sys.exit(1)

try:
    from company_finder import find_similar_companies
except ImportError:
    print("Error: company_finder.py not found in the same directory.")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}
NIFTY_TICKER    = "^NSEI"
PERIOD_DAYS     = 90
# ──────────────────────────────────────────────────────────────────────────────


def date_range():
    end   = datetime.today()
    start = end - timedelta(days=PERIOD_DAYS)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_yf_ticker(ticker: str, exchange: str) -> str:
    suffix = EXCHANGE_SUFFIX.get(exchange.upper().strip(), ".NS")
    return f"{ticker}{suffix}"


def fetch_closes(yf_ticker: str, start: str, end: str):
    """Return a date-indexed Series of daily closing prices, or None on failure."""
    hist = yf.Ticker(yf_ticker).history(start=start, end=end)
    if hist.empty:
        return None
    hist.index = hist.index.tz_localize(None)
    return hist["Close"].rename(yf_ticker)


def calc_beta(stock_closes, nifty_closes) -> float:
    """
    Align stock and Nifty on common trading dates, compute daily returns,
    and return Beta = Cov(stock, nifty) / Var(nifty).
    """
    import pandas as pd
    combined     = pd.concat([stock_closes, nifty_closes], axis=1).dropna()
    stock_ret    = combined.iloc[:, 0].pct_change().dropna()
    nifty_ret    = combined.iloc[:, 1].pct_change().dropna()
    cov_matrix   = np.cov(stock_ret, nifty_ret)
    beta         = cov_matrix[0, 1] / cov_matrix[1, 1]
    return round(beta, 4)


def display_closes(closes) -> None:
    """Print date → closing price table for a single ticker."""
    print(f"\n  {'Date':<16} {'Close (₹)':>14}")
    print("  " + "-" * 32)
    for date, price in closes.items():
        print(f"  {date.strftime('%d %b %Y'):<16} {round(price, 2):>14,.2f}")
    print("  " + "=" * 32)


def display_beta_summary(beta_results: list[dict]) -> None:
    """Print the final Beta comparison table."""
    print("\n" + "=" * 68)
    print("  BETA vs NIFTY 50  (3-Month Daily Returns)")
    print("=" * 68)
    print(f"\n  {'Company':<35} {'Ticker':<14} {'Beta':>8}  Interpretation")
    print("  " + "-" * 68)

    for r in beta_results:
        beta = r["beta"]
        if beta is None:
            interp = "Data unavailable"
        elif beta > 1.5:
            interp = "Very High Risk / Aggressive"
        elif beta > 1.0:
            interp = "Higher risk than Nifty"
        elif beta == 1.0:
            interp = "Moves with Nifty"
        elif beta > 0.5:
            interp = "Lower risk than Nifty"
        elif beta > 0:
            interp = "Defensive / Low volatility"
        elif beta < 0:
            interp = "Inverse to Nifty"
        else:
            interp = "Uncorrelated"

        beta_str = f"{beta:.4f}" if beta is not None else "  N/A"
        print(f"  {r['name']:<35} {r['ticker']:<14} {beta_str:>8}  {interp}")

    print("\n  Note: Beta > 1 → stock moves MORE than Nifty on average.")
    print("        Beta < 1 → stock moves LESS than Nifty on average.")
    print("        Beta < 0 → stock moves OPPOSITE to Nifty.\n")


# ── Pipeline (importable) ──────────────────────────────────────────────────────

def run_pipeline(company_name: str) -> list[dict]:
    """
    Full pipeline: find similar companies → fetch closes → compute Beta.
    Prints all output and returns a list of dicts:
        [{"ticker": "TCS.NS", "name": "...", "beta": 0.85}, ...]
    beta is None if price data was unavailable for that company.
    """
    start, end = date_range()

    print(f"\n  [1/3] Finding companies similar to \"{company_name}\" on BSE/NSE...")
    result = find_similar_companies(company_name)
    if "error" in result:
        print(f"  Error: {result['error']}")
        return []

    listed_dict: dict = {}
    for c in result.get("similar_companies", []):
        ticker = c.get("ticker", "")
        if ticker:
            listed_dict[ticker] = {
                "name"    : c.get("name"),
                "exchange": c.get("exchange", "NSE"),
            }

    if not listed_dict:
        print("  No companies found.")
        return []

    print(f"  Found: {', '.join(listed_dict.keys())}")

    print(f"\n  [2/3] Fetching Nifty 50 ({NIFTY_TICKER}) closing prices ({start} → {end})...")
    nifty_closes = fetch_closes(NIFTY_TICKER, start, end)
    if nifty_closes is None:
        print("  Error: Could not fetch Nifty 50 data.")
        return []
    print(f"  Nifty 50: {len(nifty_closes)} trading days loaded.")

    print(f"\n  [3/3] Fetching stock closes and computing Beta...\n")
    beta_results = []

    print("=" * 68)
    print(f"  NIFTY 50  ({NIFTY_TICKER})  —  3-Month Daily Closing Prices")
    print("=" * 68)
    display_closes(nifty_closes)

    for ticker, info in listed_dict.items():
        yf_ticker = build_yf_ticker(ticker, info.get("exchange", "NSE"))
        name      = info["name"]

        print(f"\n{'=' * 68}")
        print(f"  {name}  ({yf_ticker})  —  3-Month Daily Closing Prices")
        print("=" * 68)

        closes = fetch_closes(yf_ticker, start, end)
        if closes is None:
            print(f"  [!] No data found for {yf_ticker}")
            beta_results.append({"ticker": yf_ticker, "name": name, "beta": None})
            continue

        display_closes(closes)
        beta = calc_beta(closes, nifty_closes)
        beta_results.append({"ticker": yf_ticker, "name": name, "beta": beta})

    display_beta_summary(beta_results)
    return beta_results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 68)
    print("  STOCK TRACKER + BETA CALCULATOR  |  Gemini AI + Yahoo Finance")
    print("=" * 68)

    if len(sys.argv) > 1:
        company_name = " ".join(sys.argv[1:]).strip()
    else:
        company_name = input("\n  Enter company name: ").strip()

    if not company_name:
        print("  Error: Company name cannot be empty.")
        sys.exit(1)

    run_pipeline(company_name)


if __name__ == "__main__":
    main()
