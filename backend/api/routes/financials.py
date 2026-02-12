"""
API routes for financial data extraction.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.analytics.table_extractor import (
    extract_company_financials,
    extract_capex_breakdown,
    compare_company_financials,
)

router = APIRouter()


@router.get("/financials/{company}")
async def get_company_financials(company: str, fiscal_year: Optional[str] = None):
    """
    Get extracted financial data for a company.
    """
    try:
        result = extract_company_financials(company, fiscal_year)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financials/{company}/capex")
async def get_company_capex_breakdown(company: str):
    """
    Get CapEx breakdown by category for a company.
    """
    try:
        return extract_capex_breakdown(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financials/compare/all")
async def compare_financials():
    """
    Compare financial data across all companies.
    """
    try:
        return compare_company_financials()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
