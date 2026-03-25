"""
Automated scheduler for data ingestion tasks.
Uses APScheduler to run periodic data updates.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.core.config import INGESTION_SCHEDULE
from backend.ingestion.sec_downloader import SECDownloader
from backend.ingestion.processor import process_new_filings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = AsyncIOScheduler()

# ---------------------------------------------------------------------------
# JOB METADATA (friendly descriptions + last-run tracking)
# ---------------------------------------------------------------------------

# Human-readable schedule description keyed by job ID
_JOB_FRIENDLY_SCHEDULE: dict[str, str] = {
    "sec_filing_check":            "Weekdays at 4:00 PM",
    "sec_8k_check":                "Every 6 hours",
    "weekly_transcript_ingestion": "Sundays at 2:00 AM",
    "daily_ir_check":              "Weekdays at 8:00 AM",
}

# Last-run timestamps keyed by job ID (updated when each job finishes)
_job_last_run: dict[str, Optional[str]] = {
    "sec_filing_check":            None,
    "sec_8k_check":                None,
    "weekly_transcript_ingestion": None,
    "daily_ir_check":              None,
}


def _mark_done(job_id: str) -> None:
    _job_last_run[job_id] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# JOB IMPLEMENTATIONS
# ---------------------------------------------------------------------------

async def scheduled_sec_check():
    """Daily + 6-hourly task: check for new 10-K/10-Q/8-K filings."""
    logger.info(f"[{datetime.now()}] Starting scheduled SEC filing check...")
    try:
        downloader = SECDownloader()
        new_filings = await downloader.check_and_download_new_filings(
            filing_types=["10-K", "10-Q", "8-K"],
            days_back=7,
        )
        if new_filings:
            logger.info(f"Found {len(new_filings)} new filings. Processing...")
            processed = await process_new_filings(new_filings)
            logger.info(f"Processed {processed} new documents into ChromaDB")
        else:
            logger.info("No new filings found.")
    except Exception as e:
        logger.error(f"Error in scheduled SEC check: {e}")


async def scheduled_sec_filing_check():
    """Weekday 4 PM ET — full SEC filing check."""
    await scheduled_sec_check()
    _mark_done("sec_filing_check")


async def scheduled_sec_8k_check():
    """Every 6 hours — quick 8-K check."""
    await scheduled_sec_check()
    _mark_done("sec_8k_check")


def _run_transcript_ingestion_sync():
    """Sync wrapper for the full weekly transcript pipeline."""
    try:
        from backend.ingestion.transcript_scraper import run_transcript_ingestion
        run_transcript_ingestion()
    except Exception as e:
        logger.error(f"[weekly_transcript_ingestion] Error: {e}")
    finally:
        _mark_done("weekly_transcript_ingestion")


def _run_ir_check_sync():
    """Sync wrapper for the daily IR press-release check."""
    try:
        from backend.ingestion.transcript_scraper import run_ir_only_ingestion
        run_ir_only_ingestion()
    except Exception as e:
        logger.error(f"[daily_ir_check] Error: {e}")
    finally:
        _mark_done("daily_ir_check")


async def scheduled_web_update():
    """Placeholder for future web content update tasks."""
    logger.info(f"[{datetime.now()}] Starting web content update...")


# ---------------------------------------------------------------------------
# SCHEDULER LIFECYCLE
# ---------------------------------------------------------------------------

def start_scheduler():
    """Start the background scheduler with all ingestion jobs."""
    if scheduler.running:
        logger.info("Scheduler already running")
        return

    # ── Existing jobs ──────────────────────────────────────────────────────
    scheduler.add_job(
        scheduled_sec_filing_check,
        CronTrigger.from_crontab(INGESTION_SCHEDULE),
        id="sec_filing_check",
        name="Check for new SEC filings",
        replace_existing=True,
    )
    scheduler.add_job(
        scheduled_sec_8k_check,
        IntervalTrigger(hours=6),
        id="sec_8k_check",
        name="Quick check for 8-K filings",
        replace_existing=True,
    )

    # ── New jobs ───────────────────────────────────────────────────────────
    # Weekly full transcript + IR ingestion — Sundays at 2 AM
    scheduler.add_job(
        _run_transcript_ingestion_sync,
        CronTrigger(day_of_week="sun", hour=2, minute=0),
        id="weekly_transcript_ingestion",
        name="Weekly transcript and IR ingestion",
        replace_existing=True,
    )
    # Daily IR press release check — weekdays at 8 AM
    scheduler.add_job(
        _run_ir_check_sync,
        CronTrigger(day_of_week="1-5", hour=8, minute=0),
        id="daily_ir_check",
        name="Daily IR press release check",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")


def stop_scheduler():
    """Stop the background scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


def get_scheduler_status() -> dict:
    """
    Return scheduler status with human-readable schedule descriptions,
    last-run timestamps, and per-job status badges.
    """
    jobs = []
    for job in scheduler.get_jobs():
        job_id: str = job.id
        next_run: Optional[str] = (
            job.next_run_time.isoformat() if job.next_run_time else None
        )
        last_run: Optional[str] = _job_last_run.get(job_id)

        # Derive status
        if not scheduler.running:
            status = "stopped"
        elif next_run:
            status = "pending"
        else:
            status = "idle"

        jobs.append(
            {
                "id": job_id,
                "name": job.name,
                "trigger": str(job.trigger),
                "friendly_schedule": _JOB_FRIENDLY_SCHEDULE.get(job_id, ""),
                "next_run": next_run,
                "last_run": last_run,
                "status": status,
            }
        )

    return {
        "running": scheduler.running,
        "jobs": jobs,
    }


async def run_manual_check():
    """Manually trigger a full SEC filing check."""
    logger.info("Manual SEC filing check triggered")
    await scheduled_sec_check()
