#!/usr/bin/env python3
"""
Extract Benchmark CapEx values from 10-K and 10-Q PDFs.
Searches for cash flow statement data containing capital expenditures.
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BM_10K = PROJECT_ROOT / "Benchmark" / "Benchmark_filings" / "10K"
BM_10Q = PROJECT_ROOT / "Benchmark" / "Benchmark_filings" / "10Q"

# Patterns for capital expenditures in thousands (Benchmark reports in thousands)
CAPEX_PATTERNS = [
    # "Capital expenditures  (78,456)" or "Purchases of property  (78,456)"
    r'[Cc]apital\s+expenditures?\s*[\(\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*\)?',
    r'[Pp]urchases?\s+of\s+property[^$\n]*[\(\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*\)?',
    r'[Aa]cquisitions?\s+of\s+property[^$\n]*[\(\$]?\s*([0-9,]+(?:\.[0-9]+)?)\s*\)?',
]

def extract_capex_from_text(text):
    """Find capital expenditure values in PDF text. Returns list of (context, value_thousands)."""
    results = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        if not re.search(r'capital expenditure|purchases? of property|acquisition of property', line, re.IGNORECASE):
            continue
        # Get surrounding context
        start = max(0, i - 2)
        end = min(len(lines), i + 3)
        ctx = ' | '.join(l.strip() for l in lines[start:end] if l.strip())

        # Find numbers in the line (might be in thousands)
        nums = re.findall(r'[\(\$]?\s*([0-9]{2,},?[0-9]{3}(?:\.[0-9]+)?)\s*\)?', ctx)
        for num in nums:
            try:
                val = float(num.replace(',', ''))
                if 10 < val < 1_000_000:  # Reasonable CapEx range in thousands
                    results.append((ctx[:250], val))
            except ValueError:
                pass
    return results


def process_pdf(pdf_path):
    """Extract text from PDF, searching for CapEx values."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            # Focus on cash flow section: typically in last 40% of 10-K
            start_page = max(0, int(total_pages * 0.5))
            all_results = []
            for page_num in range(start_page, total_pages):
                page = pdf.pages[page_num]
                text = page.extract_text() or ''
                if 'cash flow' in text.lower() or 'capital expenditure' in text.lower() or 'purchases of property' in text.lower():
                    results = extract_capex_from_text(text)
                    for ctx, val in results:
                        all_results.append((page_num + 1, ctx, val))
            return all_results
    except Exception as e:
        return [(0, f"ERROR: {e}", 0)]


def main():
    # Process 10-Ks
    print("=== BENCHMARK 10-K CapEx Extraction ===")
    print("Note: Benchmark reports in THOUSANDS. Divide by 1000 for millions.\n")

    for pdf_file in sorted(BM_10K.glob("*.pdf")):
        year = pdf_file.stem.split('_')[0]
        print(f"\n{pdf_file.name} (filing year {year}, covers FY{int(year)-1}):")
        results = process_pdf(pdf_file)
        if not results:
            print("  No CapEx found")
        for pg, ctx, val in results[:3]:
            print(f"  Page {pg}: {val:,.0f} thousands = ${val/1000:.3f}M")
            print(f"    Context: {ctx[:200]}")

    print("\n\n=== BENCHMARK 10-Q CapEx Extraction ===")
    for pdf_file in sorted(BM_10Q.glob("*.pdf")):
        parts = pdf_file.stem.split('_')
        year = parts[0]
        quarter = parts[1] if len(parts) > 1 else '?'
        print(f"\n{pdf_file.name} ({year} {quarter}):")
        results = process_pdf(pdf_file)
        if not results:
            print("  No CapEx found")
        for pg, ctx, val in results[:3]:
            print(f"  Page {pg}: {val:,.0f} thousands = ${val/1000:.3f}M")
            print(f"    Context: {ctx[:200]}")


if __name__ == "__main__":
    main()
