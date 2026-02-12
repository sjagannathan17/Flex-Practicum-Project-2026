"""
API routes for advanced analytics.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.analytics.anomaly import (
    detect_capex_anomalies,
    detect_sentiment_shifts,
    detect_ai_investment_changes,
    get_all_anomalies,
)
from backend.analytics.trends import (
    analyze_company_trends,
    compare_company_trends,
)
from backend.analytics.classifier import (
    classify_company_investments,
    compare_investment_focus,
    get_ai_investment_leaders,
)

router = APIRouter()


# ============== ANOMALY DETECTION ==============

@router.get("/analytics/anomalies")
async def get_anomalies():
    """
    Get all detected anomalies across companies.
    Includes CapEx spikes, sentiment shifts, and AI investment changes.
    """
    try:
        return get_all_anomalies()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/anomalies/capex/{company}")
async def get_company_capex_anomalies(company: str):
    """
    Detect CapEx anomalies for a specific company.
    """
    try:
        return detect_capex_anomalies(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/anomalies/sentiment/{company}")
async def get_company_sentiment_shifts(company: str):
    """
    Detect sentiment shifts for a specific company.
    """
    try:
        return detect_sentiment_shifts(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/anomalies/ai-focus/{company}")
async def get_company_ai_changes(company: str):
    """
    Detect AI investment focus changes for a specific company.
    """
    try:
        return detect_ai_investment_changes(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== TREND ANALYSIS ==============

@router.get("/analytics/trends")
async def get_all_trends():
    """
    Get trend analysis and comparisons across all companies.
    """
    try:
        return compare_company_trends()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/trends/{company}")
async def get_company_trends(company: str):
    """
    Get trend analysis for a specific company.
    Includes CapEx, AI focus, and sentiment trends with forecasts.
    """
    try:
        return analyze_company_trends(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== INVESTMENT CLASSIFICATION ==============

@router.get("/analytics/classification")
async def get_investment_classification():
    """
    Get AI vs Traditional investment classification across all companies.
    """
    try:
        return compare_investment_focus()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/classification/{company}")
async def get_company_classification(company: str, n_docs: int = 50):
    """
    Classify investments for a specific company as AI/Data Center or Traditional.
    """
    try:
        return classify_company_investments(company, n_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/ai-leaders")
async def get_ai_leaders():
    """
    Get companies leading in AI/Data Center investments.
    """
    try:
        return get_ai_investment_leaders()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== DASHBOARD DATA ==============

@router.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """
    Get comprehensive analytics dashboard data.
    """
    try:
        # Get all data
        anomalies = get_all_anomalies()
        trends = compare_company_trends()
        classification = compare_investment_focus()
        leaders = get_ai_investment_leaders()
        
        return {
            "anomalies": {
                "capex_anomalies_count": len(anomalies["capex_anomalies"]),
                "sentiment_shifts_count": len(anomalies["sentiment_shifts"]),
                "companies_with_anomalies": anomalies["summary"]["companies_with_capex_anomalies"],
            },
            "trends": {
                "market_outlook": trends["market_outlook"],
                "companies_growing_ai": trends["rankings"]["ai_focus_growth"],
                "companies_improving_sentiment": trends["rankings"]["sentiment_improvement"],
            },
            "classification": {
                "industry_average_ai_focus": classification["industry_average_ai_focus"],
                "most_ai_focused": classification["rankings"]["ai_focused"][0]["company"] if classification["rankings"]["ai_focused"] else None,
                "most_traditional": classification["rankings"]["traditional_focused"][0]["company"] if classification["rankings"]["traditional_focused"] else None,
            },
            "leaders": leaders,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
