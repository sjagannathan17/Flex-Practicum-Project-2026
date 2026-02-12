"""
API routes for earnings transcripts and calendar.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.ingestion.earnings_scraper import (
    EarningsDataManager,
    EarningsCalendar,
    search_earnings_transcripts,
    get_earnings_calendar,
    get_upcoming_earnings as get_upcoming,
    scrape_earnings_highlights,
)

router = APIRouter()


@router.get("/earnings/calendar")
async def get_calendar():
    """
    Get the earnings calendar for all tracked companies.
    """
    calendar = await get_earnings_calendar()
    return {
        "calendar": calendar,
        "note": "These are typical earnings reporting periods based on fiscal year ends",
    }


@router.get("/earnings/company/{ticker}")
async def get_company_earnings(ticker: str):
    """
    Get earnings information for a specific company.
    Includes fiscal year schedule and recent transcript sources.
    """
    manager = EarningsDataManager()
    
    try:
        info = await manager.get_company_earnings_info(ticker.upper())
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/earnings/transcripts/{ticker}")
async def search_transcripts(ticker: str):
    """
    Search for earnings transcripts for a company.
    Returns links to potential transcript sources found via web search.
    """
    try:
        transcripts = await search_earnings_transcripts(ticker.upper())
        return {
            "ticker": ticker.upper(),
            "transcript_sources": transcripts,
            "count": len(transcripts),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/earnings/all")
async def get_all_earnings():
    """
    Get earnings information for all tracked companies.
    """
    manager = EarningsDataManager()
    
    try:
        all_info = await manager.get_all_earnings_info()
        return {
            "companies": all_info,
            "count": len(all_info),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/earnings/upcoming")
async def get_upcoming_earnings():
    """
    Get upcoming earnings dates based on typical schedule.
    Uses smart filtering to show only near-term earnings.
    """
    try:
        upcoming = await get_upcoming()
        return {
            "upcoming_earnings": upcoming,
            "count": len(upcoming),
        }
    except Exception as e:
        # Fallback to simple calendar if new function fails
        from datetime import datetime
        
        calendar = EarningsCalendar.get_all_schedules()
        current_month = datetime.now().strftime("%B")
        
        upcoming = []
        for ticker, schedule in calendar.items():
            for quarter in schedule.get("typical_quarters", []):
                parts = quarter.split()
                if len(parts) >= 2:
                    quarter_name = parts[0]
                    month = parts[1]
                    
                    upcoming.append({
                        "ticker": ticker,
                        "quarter": quarter_name,
                        "expected_month": month,
                        "fiscal_year_end": schedule.get("fiscal_year_end"),
                    })
        
        return {
            "upcoming_earnings": upcoming,
            "current_month": current_month,
        }


@router.get("/earnings/highlights/{ticker}")
async def get_earnings_highlights(ticker: str):
    """
    Get earnings highlights for a company from documents and web search.
    Combines internal document data with recent web news.
    """
    try:
        highlights = await scrape_earnings_highlights(ticker.upper())
        return highlights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
