#!/usr/bin/env python3
"""
Unlevered Beta Calculator
──────────────────────────
Uses beta values from stock_price_tracker.py and D/E ratios from
debt_equity_analyzer.py to compute the Unlevered (Asset) Beta
for each company using the Hamada equation:

    Beta_U = Beta_L / (1 + (1 - t) × D/E)

where:
    Beta_L = Levered beta (from 3-month price regression vs Nifty 50)
    D/E    = Debt-to-Equity ratio (from Gemini web search of balance sheet)
    t      = Corporate tax rate = 22% (India standard rate)

Companies without BOTH a valid Beta AND a valid D/E are skipped.

Usage:
    python unlevered_beta.py
    python unlevered_beta.py "Mergedeck"
"""

import sys

try:
    from debt_equity_analyzer import run_full_pipeline
except ImportError:
    print("Error: debt_equity_analyzer.py not found in the same directory.")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
TAX_RATE = 0.22   # India standard corporate tax rate
# ──────────────────────────────────────────────────────────────────────────────


def calc_unlevered_beta(beta_l: float, de_ratio: float, t: float = TAX_RATE) -> float:
    """
    Hamada Equation:
        Beta_U = Beta_L / (1 + (1 - t) × D/E)
    """
    return round(beta_l / (1 + (1 - t) * de_ratio), 6)


def interpret_beta(b: float) -> str:
    if b > 1.5:   return "Very High systematic risk"
    if b > 1.0:   return "High systematic risk"
    if b > 0.5:   return "Moderate systematic risk"
    if b > 0.0:   return "Low systematic risk"
    if b < 0.0:   return "Inverse to market"
    return "Uncorrelated"


def display_unlevered_results(results: list[dict]) -> None:
    """Print detailed calculation + final comparison table."""

    print("\n" + "=" * 72)
    print("  UNLEVERED BETA CALCULATION  (Hamada Equation, t = 22%)")
    print("=" * 72)
    print(f"  Formula: Beta_U = Beta_L / (1 + (1 - 0.22) × D/E)\n")

    for r in results:
        beta_l   = r["beta_l"]
        de       = r["de_ratio"]
        beta_u   = r["beta_u"]
        adj      = round((1 + (1 - TAX_RATE) * de), 6)

        print(f"  Company  : {r['name']}")
        print(f"  Ticker   : {r['ticker']}")
        print(f"  ── Inputs ────────────────────────────────────────────────")
        print(f"  Beta_L (Levered)     : {beta_l:.4f}")
        print(f"  D/E Ratio            : {de:.4f}")
        print(f"  Tax Rate (t)         : {TAX_RATE*100:.0f}%")
        print(f"  ── Calculation ───────────────────────────────────────────")
        print(f"  1 + (1 - {TAX_RATE}) × {de:.4f}  =  {adj:.6f}")
        print(f"  Beta_U = {beta_l:.4f} / {adj:.6f}  =  {beta_u:.6f}")
        print(f"  ── Result ────────────────────────────────────────────────")
        print(f"  Unlevered Beta       : {beta_u:.6f}")
        print(f"  Interpretation       : {interpret_beta(beta_u)}")
        print("  " + "=" * 60)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY  —  Levered vs Unlevered Beta (t = 22%)")
    print("=" * 72)
    print(f"\n  {'Company':<30} {'Ticker':<16} {'D/E':>7}  {'Beta_L':>8}  {'Beta_U':>8}")
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['name']:<30} {r['ticker']:<16} "
              f"{r['de_ratio']:>7.4f}  {r['beta_l']:>8.4f}  {r['beta_u']:>8.6f}")

    avg_beta_u = sum(r["beta_u"] for r in results) / len(results)
    print("  " + "-" * 72)
    print(f"  {'AVERAGE UNLEVERED BETA':<30} {'':16} {'':>7}  {'':>8}  {avg_beta_u:>8.6f}")

    print(f"\n  Note: Beta_U removes the effect of financial leverage.")
    print(f"        It reflects only the business/operating risk of the company.")
    print(f"        Lower Beta_U = lower inherent business risk.\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print("  UNLEVERED BETA CALCULATOR  |  Hamada Equation  |  t = 22%")
    print("=" * 72)

    company_name = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 \
                   else input("\n  Enter company name: ").strip()

    if not company_name:
        print("  Error: Company name cannot be empty.")
        sys.exit(1)

    # ── Step 1: Run full pipeline (company finder → beta → D/E) ───────────────
    print(f"\n  Running full pipeline for \"{company_name}\"...")
    pipeline_results = run_full_pipeline(company_name)

    if not pipeline_results:
        print("  No results from pipeline. Exiting.")
        sys.exit(1)

    # ── Step 2: Filter — need BOTH beta and D/E to compute Beta_U ─────────────
    eligible = [r for r in pipeline_results
                if r.get("beta") is not None and r.get("de_ratio") is not None]
    skipped  = [r for r in pipeline_results
                if r.get("beta") is None or r.get("de_ratio") is None]

    if skipped:
        print(f"\n  Skipping (missing Beta or D/E): "
              f"{', '.join(r['name'] for r in skipped)}")

    if not eligible:
        print("  No companies have both Beta and D/E. Cannot calculate Beta_U.")
        sys.exit(1)

    # ── Step 3: Calculate Unlevered Beta ──────────────────────────────────────
    results = []
    for r in eligible:
        beta_u = calc_unlevered_beta(r["beta"], r["de_ratio"])
        results.append({
            "name"    : r["name"],
            "ticker"  : r["ticker"],
            "beta_l"  : r["beta"],
            "de_ratio": r["de_ratio"],
            "beta_u"  : beta_u,
        })

    # ── Step 4: Display ───────────────────────────────────────────────────────
    display_unlevered_results(results)


if __name__ == "__main__":
    main()
