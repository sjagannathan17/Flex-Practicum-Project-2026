"""
Optimized dashboard API endpoint.
Provides pre-aggregated data for fast dashboard loading.
"""
from fastapi import APIRouter
from datetime import datetime

from backend.core.config import COMPANIES
from backend.core.cache import api_cache, cached
from backend.core.database import get_collection_stats
from backend.analytics.sentiment import analyze_company_sentiment
from backend.analytics.classifier import classify_company_investments, compare_investment_focus
from backend.analytics.trends import analyze_company_trends, compare_company_trends
from backend.analytics.geographic import get_company_facilities
from backend.analytics.anomaly import get_all_anomalies

router = APIRouter()


@router.get("/dashboard/quick")
async def get_quick_dashboard():
    """
    Fast dashboard endpoint with minimal data.
    Use for initial page load.
    """
    cache_key = "dashboard_quick"
    cached_data = api_cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Get basic stats only
    try:
        stats = get_collection_stats()
    except:
        stats = {"total_documents": 0, "companies": {}}
    
    data = {
        "stats": {
            "total_documents": stats.get("total_documents", 0),
            "companies_tracked": len(COMPANIES),
        },
        "companies": [
            {
                "name": config["name"].split()[0],
                "ticker": ticker,
            }
            for ticker, config in COMPANIES.items()
        ],
        "generated_at": datetime.now().isoformat(),
    }
    
    api_cache.set(cache_key, data)
    return data


@router.get("/dashboard/full")
async def get_full_dashboard():
    """
    Full dashboard data with analytics.
    Use for complete dashboard view.
    """
    cache_key = "dashboard_full"
    cached_data = api_cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Get all analytics data
    try:
        stats = get_collection_stats()
    except:
        stats = {"total_documents": 0, "companies": {}}
    
    # Build company summaries
    company_summaries = []
    for ticker, config in COMPANIES.items():
        company_name = config["name"].split()[0]
        
        try:
            sentiment = analyze_company_sentiment(company_name)
            classification = classify_company_investments(company_name)
            trends = analyze_company_trends(company_name)
            facilities = get_company_facilities(company_name)
            
            company_summaries.append({
                "name": company_name,
                "ticker": ticker,
                "sentiment_score": sentiment.get("sentiment_score", 0),
                "sentiment_label": sentiment.get("overall_sentiment", "neutral"),
                "ai_focus": classification.get("overall_ai_focus_percentage", 0),
                "investment_focus": classification.get("investment_focus", "balanced"),
                "trend_outlook": trends.get("overall_outlook", "neutral"),
                "facility_count": facilities.get("total_facilities", 0),
                "headquarters": facilities.get("headquarters", {}).get("city", "N/A"),
            })
        except Exception as e:
            company_summaries.append({
                "name": company_name,
                "ticker": ticker,
                "error": str(e),
            })
    
    # Get comparison data
    try:
        investment_comparison = compare_investment_focus()
        trends_comparison = compare_company_trends()
    except:
        investment_comparison = {}
        trends_comparison = {}
    
    # Get anomalies summary
    try:
        anomalies = get_all_anomalies()
        total_anomalies = sum(
            len(a) for company_data in anomalies.values() 
            for a in company_data.values() if isinstance(a, list)
        )
    except:
        total_anomalies = 0
    
    data = {
        "stats": {
            "total_documents": stats.get("total_documents", 0),
            "companies_tracked": len(COMPANIES),
            "total_anomalies": total_anomalies,
        },
        "companies": company_summaries,
        "investment_comparison": {
            "ai_leaders": investment_comparison.get("ai_leaders", []),
            "traditional_leaders": investment_comparison.get("traditional_leaders", []),
        },
        "trends_comparison": {
            "positive_outlook": [
                c for c, d in trends_comparison.get("companies", {}).items()
                if d.get("outlook") == "positive"
            ],
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    api_cache.set(cache_key, data)
    return data


@router.get("/dashboard/company/{company}")
async def get_company_dashboard(company: str):
    """
    Single company dashboard data.
    """
    cache_key = f"dashboard_company_{company.lower()}"
    cached_data = api_cache.get(cache_key)
    if cached_data:
        return cached_data
    
    company_title = company.title()
    
    try:
        sentiment = analyze_company_sentiment(company_title)
        classification = classify_company_investments(company_title)
        trends = analyze_company_trends(company_title)
        facilities = get_company_facilities(company_title)
        
        data = {
            "company": company_title,
            "sentiment": {
                "score": sentiment.get("sentiment_score", 0),
                "label": sentiment.get("overall_sentiment", "neutral"),
                "positive_count": sentiment.get("positive_count", 0),
                "negative_count": sentiment.get("negative_count", 0),
            },
            "investment": {
                "ai_focus": classification.get("overall_ai_focus_percentage", 0),
                "focus_label": classification.get("investment_focus", "balanced"),
                "total_documents": classification.get("total_documents", 0),
            },
            "trends": {
                "outlook": trends.get("overall_outlook", "neutral"),
                "capex_direction": trends.get("capex_trend", {}).get("direction", "stable"),
                "ai_focus_direction": trends.get("ai_focus_trend", {}).get("direction", "stable"),
            },
            "geographic": {
                "facility_count": facilities.get("total_facilities", 0),
                "headquarters": facilities.get("headquarters", {}),
            },
            "generated_at": datetime.now().isoformat(),
        }
        
        api_cache.set(cache_key, data)
        return data
        
    except Exception as e:
        return {
            "company": company_title,
            "error": str(e),
        }


@router.delete("/dashboard/cache")
async def clear_dashboard_cache():
    """Clear dashboard cache to force refresh."""
    api_cache.clear()
    return {"success": True, "message": "Dashboard cache cleared"}


@router.post("/dashboard/warmup")
async def warmup_cache():
    """
    Pre-populate cache with common data.
    Call this on app startup or periodically.
    """
    warmed = []
    
    # Warm up company data
    for ticker, config in COMPANIES.items():
        company_name = config["name"].split()[0]
        try:
            analyze_company_sentiment(company_name)
            warmed.append(f"sentiment:{company_name}")
        except:
            pass
        
        try:
            classify_company_investments(company_name)
            warmed.append(f"classification:{company_name}")
        except:
            pass
        
        try:
            analyze_company_trends(company_name)
            warmed.append(f"trends:{company_name}")
        except:
            pass
    
    # Warm up comparison data
    try:
        compare_investment_focus()
        warmed.append("investment_comparison")
    except:
        pass
    
    try:
        compare_company_trends()
        warmed.append("trends_comparison")
    except:
        pass
    
    return {
        "success": True,
        "warmed_items": len(warmed),
        "items": warmed,
    }
