"""
AI-powered CapEx extraction from SEC filings.
Uses targeted section location + LLM extraction for precise numerical answers.
"""
import re
from pathlib import Path
from typing import Optional

from backend.extraction.prompts import CAPEX_EXTRACTION_PROMPT, CAPEX_LABELS
from backend.extraction.json_parser import clean_capex, normalize_units, parse_extraction_response


def _find_cashflow_section(text: str, window: int = 5000) -> str:
    """
    Locate the Cash Flow Statement section in a full filing.

    Scores candidate locations by:
    1. Whether numbers immediately follow the label (table row vs narrative)
    2. Whether Cash Flow Statement context keywords appear nearby

    Returns ~10,000 characters around the best match.
    """
    context_keywords = [
        "investing activities",
        "operating activities",
        "financing activities",
        "cash flows from",
        "statements of cash flows",
    ]

    candidates = []
    text_lower = text.lower()

    for label in CAPEX_LABELS:
        for match in re.finditer(re.escape(label.lower()), text_lower):
            pos = match.start()
            # Check for numbers nearby (within 200 chars after label)
            after = text[pos:pos + 200]
            has_numbers = bool(re.search(r'[\$\(]?\d[\d,]*\.?\d*', after))

            # Check for context keywords nearby (within 2000 chars)
            context_region = text_lower[max(0, pos - 1000):pos + 1000]
            context_score = sum(1 for kw in context_keywords if kw in context_region)

            score = context_score * 2 + (3 if has_numbers else 0)
            candidates.append((score, pos))

    if not candidates:
        # Fallback: search for any cash flow statement header
        for kw in ["consolidated statements of cash flows", "cash flows from investing"]:
            idx = text_lower.find(kw)
            if idx != -1:
                start = max(0, idx - 500)
                end = min(len(text), idx + window)
                return text[start:end]
        # Last resort: return middle section of document
        mid = len(text) // 2
        return text[max(0, mid - window):mid + window]

    # Pick highest-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_pos = candidates[0][1]

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + window)
    return text[start:end]


def _detect_unit_header(section_text: str) -> str:
    """Detect the unit header (thousands, millions, billions) from a financial statement section."""
    patterns = [
        r"\(in\s+(thousands|millions|billions)\)",
        r"\(amounts?\s+in\s+(thousands|millions|billions)\)",
        r"\(expressed\s+in\s+(thousands|millions|billions)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, section_text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return "millions"


def extract_capex(
    document_text: str,
    company: str = "",
    filing_type: str = "",
    fiscal_year: str = "",
    quarter: str = "",
) -> dict:
    """
    Extract CapEx from a full document using targeted section location + LLM.

    Args:
        document_text: Full text of the SEC filing
        company: Company name (for context)
        filing_type: Filing type (10-K, 10-Q, etc.)
        fiscal_year: Fiscal year
        quarter: Quarter (Q1, Q2, etc.)

    Returns:
        Dict with capex_value, unit, raw_value, confidence, etc.
    """
    from backend.core.config import LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY
    from backend.core.llm_client import llm_complete

    active_key = ANTHROPIC_API_KEY if LLM_PROVIDER == "anthropic" else OPENAI_API_KEY
    if not active_key:
        return {"error": f"{LLM_PROVIDER.upper()}_API_KEY not configured"}

    # Locate the Cash Flow Statement section
    section = _find_cashflow_section(document_text)
    unit_header = _detect_unit_header(section)

    # Build context for the LLM
    context_info = []
    if company:
        context_info.append(f"Company: {company}")
    if filing_type:
        context_info.append(f"Filing type: {filing_type}")
    if fiscal_year:
        context_info.append(f"Fiscal year: {fiscal_year}")
    if quarter:
        context_info.append(f"Quarter: {quarter}")
    context_info.append(f"Unit header detected: (in {unit_header})")

    user_prompt = (
        "\n".join(context_info)
        + "\n\n=== FINANCIAL STATEMENT EXCERPT ===\n"
        + section
    )

    try:
        raw_text = llm_complete(
            messages=[{"role": "user", "content": user_prompt}],
            system=CAPEX_EXTRACTION_PROMPT,
            model_key="main",
            max_tokens=500,
        )
        result = parse_extraction_response(raw_text)

        # Normalize units if we got a value
        if result.get("capex_value") is not None:
            raw = clean_capex(result["capex_value"])
            if raw is not None:
                result["capex_value"] = normalize_units(raw, unit_header)
                result["unit"] = "millions"
                result["unit_header_detected"] = unit_header

        return result

    except Exception as e:
        return {"error": f"Extraction failed: {e}"}
