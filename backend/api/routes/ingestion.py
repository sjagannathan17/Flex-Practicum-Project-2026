"""
API routes for data ingestion management.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional
from pydantic import BaseModel

from backend.ingestion.sec_downloader import SECDownloader
from backend.ingestion.scheduler import (
    start_scheduler,
    stop_scheduler,
    get_scheduler_status,
    run_manual_check,
)
from backend.ingestion.processor import process_new_filings

router = APIRouter()


class FilingCheckRequest(BaseModel):
    """Request body for filing check."""
    days_back: int = 30
    filing_types: list[str] = ["10-K", "10-Q", "8-K"]


@router.get("/ingestion/status")
async def get_ingestion_status():
    """Get current ingestion scheduler status."""
    scheduler_status = get_scheduler_status()
    
    downloader = SECDownloader()
    download_stats = downloader.get_download_stats()
    
    return {
        "scheduler": scheduler_status,
        "downloads": download_stats,
    }


@router.post("/ingestion/start-scheduler")
async def api_start_scheduler():
    """Start the automated ingestion scheduler."""
    try:
        start_scheduler()
        return {
            "status": "started",
            "scheduler": get_scheduler_status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingestion/stop-scheduler")
async def api_stop_scheduler():
    """Stop the automated ingestion scheduler."""
    try:
        stop_scheduler()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingestion/check-filings")
async def check_filings(
    background_tasks: BackgroundTasks,
    request: FilingCheckRequest,
):
    """
    Manually trigger a check for new SEC filings.
    Returns immediately and processes in background.
    """
    async def check_and_process():
        downloader = SECDownloader()
        new_filings = await downloader.check_and_download_new_filings(
            filing_types=request.filing_types,
            days_back=request.days_back,
        )
        
        if new_filings:
            await process_new_filings(new_filings)
    
    background_tasks.add_task(check_and_process)
    
    return {
        "status": "checking",
        "message": f"Checking for filings from last {request.days_back} days",
        "filing_types": request.filing_types,
    }


@router.get("/ingestion/filings")
async def get_available_filings(
    ticker: Optional[str] = None,
    days_back: int = 90,
):
    """
    Get list of available SEC filings (without downloading).
    """
    downloader = SECDownloader()
    
    if ticker:
        filings = await downloader.get_company_filings(ticker.upper(), days_back=days_back)
        return {"ticker": ticker, "filings": filings}
    
    # Get for all companies
    all_filings = {}
    from backend.core.config import COMPANIES
    
    for company_ticker in COMPANIES.keys():
        filings = await downloader.get_company_filings(company_ticker, days_back=days_back)
        all_filings[company_ticker] = filings
    
    return {"filings": all_filings}


@router.post("/ingestion/download-filing")
async def download_specific_filing(
    ticker: str,
    form: str,
    filing_date: str,
    background_tasks: BackgroundTasks,
):
    """
    Download a specific filing.
    """
    downloader = SECDownloader()
    
    # Find the filing
    filings = await downloader.get_company_filings(ticker.upper(), days_back=365)
    
    target = None
    for filing in filings:
        if filing["form"] == form and filing["filing_date"] == filing_date:
            target = filing
            break
    
    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Filing not found: {ticker} {form} {filing_date}",
        )
    
    if target["already_downloaded"]:
        return {
            "status": "already_downloaded",
            "filing": target,
        }
    
    # Download in background
    async def download_and_process():
        path = await downloader.download_filing(target)
        if path:
            target["local_path"] = str(path)
            await process_new_filings([target])
    
    background_tasks.add_task(download_and_process)
    
    return {
        "status": "downloading",
        "filing": target,
    }
