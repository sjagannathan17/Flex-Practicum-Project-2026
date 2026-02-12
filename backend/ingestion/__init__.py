"""
Ingestion module for automated data collection and processing.
"""
from .sec_downloader import SECDownloader, check_new_filings_sync
from .scheduler import start_scheduler, stop_scheduler, get_scheduler_status
from .processor import process_new_filings, process_filing
from .earnings_scraper import (
    EarningsCalendar,
    TranscriptScraper,
    EarningsDataManager,
    search_earnings_transcripts,
    get_earnings_calendar,
)

__all__ = [
    "SECDownloader",
    "check_new_filings_sync",
    "start_scheduler",
    "stop_scheduler",
    "get_scheduler_status",
    "process_new_filings",
    "process_filing",
    "EarningsCalendar",
    "TranscriptScraper",
    "EarningsDataManager",
    "search_earnings_transcripts",
    "get_earnings_calendar",
]
